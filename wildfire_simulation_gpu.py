#!/usr/bin/env python3
"""GPU-accelerated synthetic wildfire/forest-management simulator.

Model: two-layer probabilistic cellular automaton (surface + canopy fire).
Backend: PyTorch; automatically uses NVIDIA CUDA, Apple MPS, or CPU.

This is an educational, semi-empirical research model, NOT a full Rothermel,
FARSITE, WRF-SFIRE, combustion-chemistry, or operational fire-behavior model.
Its coefficients are explicit hypotheses that require calibration and
independent validation before interpreting results as real-world predictions.

Install PyTorch for your platform first (see README_AI_AND_CITATION.md), then:
    python -m pip install pandas numpy
    python wildfire_simulation_gpu.py --device auto

Quick correctness test:
    python wildfire_simulation_gpu.py --device cpu --quick
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DEFAULT EXPERIMENTAL DESIGN
# ---------------------------------------------------------------------------

MODEL_VERSION = "2.0"
DEFAULT_SEED = 20261117
DEFAULT_GRID_SIZE = 48
DEFAULT_CELL_SIZE_M = 2.0
DEFAULT_MAX_STEPS = 180        # one simulation step represents one model minute
DEFAULT_BATCH_SIZE = 64
DEFAULT_REPLICATES = 2

TREE_DENSITIES_TREES_HA = [250, 500, 750]
TREE_ARRANGEMENTS = ["random", "regular", "clustered"]
WIND_SPEEDS_KMH = [5, 15, 25]
# Direction convention: 0=east/right, 90=north/up, 180=west/left.
WIND_DIRECTIONS_DEG = [0, 90, 180]
FUEL_MOISTURES_PERCENT = [5, 15, 25]
SLOPES_DEG = [0, 20]
# The slope aspect is the direction pointing uphill; held fixed in this design.
SLOPE_ASPECT_DEG = 0.0
MANAGEMENT_STRATEGIES = [
    "unmanaged",
    "thin_from_below",
    "surface_fuel_reduction",
    "combined_thinning_and_fuel_reduction",
]

# ---------------------------------------------------------------------------
# MODEL PARAMETERS (explicitly tunable; not universal constants)
# ---------------------------------------------------------------------------

# Moisture damping follows the cubic functional form documented for the
# Rothermel model, but this simulator does NOT implement the full Rothermel
# spread equation. The 30% extinction moisture is a configurable assumption.
MOISTURE_OF_EXTINCTION_PERCENT = 30.0

# Thinning removes small trees and leaves a small amount of slash on site.
THIN_SIZE_CUTOFF = 0.30
THIN_SLASH_MULTIPLIER = 1.08
SURFACE_FUEL_REMAINING_AFTER_TREATMENT = 0.65

# Relative / dimensionless model coefficients; tune against observed fires.
HEAT_TRANSFER_COEFFICIENT = 0.30
SURFACE_IGNITION_ENERGY_INDEX = 1.00
CROWN_IGNITION_ENERGY_INDEX = 1.70
CROWN_HEAT_MULTIPLIER = 1.25
WIND_COEFFICIENT = 0.23
SLOPE_COEFFICIENT = 1.40
MIDFLAME_WIND_ADJUSTMENT = 0.40

# Dimensionless thermal-field update coefficients. These are model assumptions,
# not measured diffusivities or temperatures in physical units.
THERMAL_DIFFUSIVITY = 0.10
THERMAL_HEAT_LOSS_PER_STEP = 0.25
THERMAL_SOURCE_GAIN = 0.20
WIND_ADVECTION_RATE = 0.08
THERMAL_FIELD_IGNITION_WEIGHT = 0.40
THERMAL_FIELD_STOP_THRESHOLD = 0.04
BURN_DURATION_STEPS = 2

# The ember module is deliberately short-range and probabilistic. This is a
# sensitivity parameter, not a validated spotting-distance distribution.
SPOTTING_ENABLED_BY_DEFAULT = True
SPOTTING_BASE_PROBABILITY = 0.045
SPOTTING_OFFSET_CELLS = 3

# Tree mortality is a separate exposure-response proxy. Tree-size is used as
# a bark/resistance proxy; it is NOT a species-specific mortality model.
TREE_RESISTANCE_BASE = 0.70
TREE_RESISTANCE_SIZE_COEFFICIENT = 1.10
MORTALITY_DOSE_SCALE = 0.30


# ---------------------------------------------------------------------------
# REPRODUCIBILITY AND DEVICE SETUP
# ---------------------------------------------------------------------------

def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but PyTorch cannot see a CUDA GPU. "
                "Install a CUDA-enabled PyTorch build for your system."
            )
        return torch.device("cuda")
    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {requested}")


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    # Use a generator on the same device as the tensors being sampled.
    generator_device = device.type if device.type in ("cpu", "cuda") else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator


def rand_tensor(
    shape: tuple[int, ...],
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    # MPS random-generator support varies by PyTorch version; sample on CPU
    # then transfer in that case. CUDA and CPU sample directly on-device.
    sample_device = torch.device("cpu") if device.type == "mps" else device
    return torch.rand(shape, device=sample_device, generator=generator).to(device)


def paired_rand_tensor(
    batch: int,
    spatial_shape: tuple[int, ...],
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """Use common random numbers within each four-treatment comparison block."""
    treatments = len(MANAGEMENT_STRATEGIES)
    if batch % treatments != 0:
        raise ValueError("Batch must be divisible by the number of treatments.")
    base = rand_tensor((batch // treatments, *spatial_shape), device, generator)
    return base.repeat_interleave(treatments, dim=0)


# ---------------------------------------------------------------------------
# SCENARIO DESIGN
# ---------------------------------------------------------------------------

def build_scenarios(replicates: int) -> list[dict[str, Any]]:
    """Build four-treatment blocks sharing identical environmental settings."""
    scenarios: list[dict[str, Any]] = []
    scenario_id = 0
    for values in itertools.product(
        TREE_DENSITIES_TREES_HA,
        TREE_ARRANGEMENTS,
        WIND_SPEEDS_KMH,
        WIND_DIRECTIONS_DEG,
        FUEL_MOISTURES_PERCENT,
        SLOPES_DEG,
        range(1, replicates + 1),
    ):
        density, arrangement, wind, wind_dir, moisture, slope, replicate = values
        # The four treatments are deliberately adjacent in each block.
        for management in MANAGEMENT_STRATEGIES:
            scenario_id += 1
            scenarios.append({
                "scenario_id": scenario_id,
                "replicate": replicate,
                "baseline_tree_density_trees_ha": density,
                "tree_arrangement": arrangement,
                "wind_speed_kmh": wind,
                "wind_direction_deg": wind_dir,
                "fuel_moisture_percent": moisture,
                "slope_degrees": slope,
                "slope_aspect_deg": SLOPE_ASPECT_DEG,
                "management": management,
            })
    return scenarios


def quick_scenarios(replicates: int) -> list[dict[str, Any]]:
    """Small factorial subset used only to smoke-test the program."""
    scenarios: list[dict[str, Any]] = []
    scenario_id = 0
    for values in itertools.product(
        [500],
        ["random", "regular", "clustered"],
        [5, 25],
        [0, 90],
        [5, 20],
        [0, 20],
        range(1, replicates + 1),
    ):
        density, arrangement, wind, wind_dir, moisture, slope, rep = values
        for management in MANAGEMENT_STRATEGIES:
            scenario_id += 1
            scenarios.append({
                "scenario_id": scenario_id,
                "replicate": rep,
                "baseline_tree_density_trees_ha": density,
                "tree_arrangement": arrangement,
                "wind_speed_kmh": wind,
                "wind_direction_deg": wind_dir,
                "fuel_moisture_percent": moisture,
                "slope_degrees": slope,
                "slope_aspect_deg": SLOPE_ASPECT_DEG,
                "management": management,
            })
    return scenarios


# ---------------------------------------------------------------------------
# GRID UTILITIES AND FOREST INITIALIZATION
# ---------------------------------------------------------------------------

def shift_source_to_target(field: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Move each source cell to the neighbor (dy, dx), with no wraparound.

    field has shape [batch, height, width]. A zero-filled output is used at
    the grid boundaries so fire cannot wrap from one edge to the opposite edge.
    """
    _, height, width = field.shape
    out = torch.zeros_like(field)

    src_y0 = max(0, -dy)
    src_y1 = min(height, height - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(width, width - dx)

    dst_y0 = max(0, dy)
    dst_y1 = min(height, height + dy)
    dst_x0 = max(0, dx)
    dst_x1 = min(width, width + dx)

    out[:, dst_y0:dst_y1, dst_x0:dst_x1] = field[:, src_y0:src_y1, src_x0:src_x1]
    return out


def smooth_random_field(
    batch: int,
    grid_size: int,
    device: torch.device,
    generator: torch.Generator,
    kernel_size: int = 9,
) -> torch.Tensor:
    noise = rand_tensor((batch, 1, grid_size, grid_size), device, generator)
    smoothed = F.avg_pool2d(noise, kernel_size, stride=1, padding=kernel_size // 2)
    smoothed = smoothed[:, 0]
    low = smoothed.amin(dim=(1, 2), keepdim=True)
    high = smoothed.amax(dim=(1, 2), keepdim=True)
    return (smoothed - low) / (high - low + 1e-7)


def create_forest_layers(
    scenario_batch: list[dict[str, Any]],
    grid_size: int,
    cell_size_m: float,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Create paired forest layouts and then apply four management treatments.

    Each contiguous four-row block contains the same base forest/environment
    under the four strategies. Base trees, tree sizes, and surface fuels are
    generated once per block and repeated across treatments. This avoids
    confounding treatment effects with different starting forest realizations.
    The fire and tree-mortality random numbers are also paired across treatments.
    """
    batch = len(scenario_batch)
    if batch % len(MANAGEMENT_STRATEGIES) != 0:
        raise ValueError(
            "Scenario batches must contain complete four-strategy blocks. "
            "Use a --batch-size divisible by 4."
        )
    treatments_per_block = len(MANAGEMENT_STRATEGIES)
    groups = batch // treatments_per_block

    # The first row of every block contains the shared base-condition values.
    block_rows = [scenario_batch[i * treatments_per_block] for i in range(groups)]
    density_group = torch.tensor(
        [s["baseline_tree_density_trees_ha"] for s in block_rows],
        dtype=torch.float32, device=device,
    )
    arrangement_group = torch.tensor(
        [{"random": 0, "regular": 1, "clustered": 2}[s["tree_arrangement"]]
         for s in block_rows],
        dtype=torch.int64, device=device,
    )
    management = torch.tensor(
        [{"unmanaged": 0, "thin_from_below": 1,
          "surface_fuel_reduction": 2,
          "combined_thinning_and_fuel_reduction": 3}[s["management"]]
         for s in scenario_batch],
        dtype=torch.int64, device=device,
    )

    domain_area_m2 = (grid_size * cell_size_m) ** 2
    desired_count_group = torch.round(
        density_group * domain_area_m2 / 10000.0
    ).to(torch.int64)
    max_count = int(desired_count_group.max().item())
    cell_count = grid_size * grid_size
    if max_count > cell_count:
        raise ValueError("Requested density exceeds one tree per grid cell.")

    # Generate one base spatial arrangement per paired treatment block.
    random_scores = rand_tensor((groups, grid_size, grid_size), device, generator)
    clustered_scores = smooth_random_field(groups, grid_size, device, generator)

    yy, xx = torch.meshgrid(
        torch.arange(grid_size, device=device, dtype=torch.float32),
        torch.arange(grid_size, device=device, dtype=torch.float32),
        indexing="ij",
    )
    spacing_cells = torch.sqrt(10000.0 / density_group) / cell_size_m
    offset_x = rand_tensor((groups, 1, 1), device, generator) * spacing_cells[:, None, None]
    offset_y = rand_tensor((groups, 1, 1), device, generator) * spacing_cells[:, None, None]
    dx = torch.remainder(
        xx[None] - offset_x + spacing_cells[:, None, None] / 2,
        spacing_cells[:, None, None],
    ) - spacing_cells[:, None, None] / 2
    dy = torch.remainder(
        yy[None] - offset_y + spacing_cells[:, None, None] / 2,
        spacing_cells[:, None, None],
    ) - spacing_cells[:, None, None] / 2
    regular_scores = -torch.sqrt(dx.square() + dy.square())
    regular_scores += 0.015 * rand_tensor((groups, grid_size, grid_size), device, generator)

    scores = torch.where(
        (arrangement_group == 0)[:, None, None], random_scores,
        torch.where((arrangement_group == 1)[:, None, None], regular_scores, clustered_scores),
    )
    selected = torch.topk(scores.reshape(groups, -1), k=max_count, dim=1).indices
    ranks = torch.arange(max_count, device=device)[None, :] < desired_count_group[:, None]
    tree_flat = torch.zeros((groups, cell_count), dtype=torch.bool, device=device)
    tree_flat.scatter_(1, selected, ranks)
    base_trees = tree_flat.reshape(groups, grid_size, grid_size)

    base_tree_size_raw = rand_tensor(
        (groups, grid_size, grid_size), device, generator
    )
    base_tree_size = torch.where(
        base_trees, base_tree_size_raw, torch.zeros_like(base_tree_size_raw)
    )
    base_surface_noise = smooth_random_field(
        groups, grid_size, device, generator, kernel_size=7
    )
    base_surface_fuel = 0.35 + 0.90 * base_surface_noise

    # Copy exactly the same base forest into all four treatment rows.
    trees_base = base_trees.repeat_interleave(treatments_per_block, dim=0)
    tree_size_base = base_tree_size.repeat_interleave(treatments_per_block, dim=0)
    surface_fuel_base = base_surface_fuel.repeat_interleave(treatments_per_block, dim=0)
    desired_count = desired_count_group.repeat_interleave(treatments_per_block)

    thinning_applied = (management == 1) | (management == 3)
    trees = trees_base & (
        ~thinning_applied[:, None, None]
        | (tree_size_base >= THIN_SIZE_CUTOFF)
    )
    tree_size = torch.where(trees, tree_size_base, torch.zeros_like(tree_size_base))

    slash_multiplier = torch.where(
        thinning_applied,
        torch.full((batch,), THIN_SLASH_MULTIPLIER, device=device),
        torch.ones((batch,), device=device),
    )
    surface_fuel = surface_fuel_base * slash_multiplier[:, None, None]

    fuel_treatment_applied = (management == 2) | (management == 3)
    fuel_multiplier = torch.where(
        fuel_treatment_applied,
        torch.full((batch,), SURFACE_FUEL_REMAINING_AFTER_TREATMENT, device=device),
        torch.ones((batch,), device=device),
    )
    surface_fuel = surface_fuel * fuel_multiplier[:, None, None]

    canopy_fuel = trees.float() * (0.35 + 0.65 * tree_size)
    canopy_local = F.avg_pool2d(
        canopy_fuel[:, None], kernel_size=5, stride=1, padding=2
    )[:, 0] * 25.0
    canopy_base_height_m = torch.where(
        trees, 2.0 + 3.0 * tree_size, torch.zeros_like(tree_size)
    )

    return {
        "trees": trees,
        "tree_size": tree_size,
        "surface_fuel": surface_fuel,
        "canopy_fuel": canopy_fuel,
        "canopy_local": canopy_local,
        "canopy_base_height_m": canopy_base_height_m,
        "trees_before_management": desired_count,
        "trees_after_management": trees.sum(dim=(1, 2)),
        "fuel_multiplier": fuel_multiplier,
        "slash_multiplier": slash_multiplier,
    }


# ---------------------------------------------------------------------------
# FIRE MODEL
# ---------------------------------------------------------------------------

def rothermel_form_moisture_damping(
    moisture_percent: torch.Tensor,
) -> torch.Tensor:
    """Cubic moisture-damping form; not the full Rothermel fire model."""
    ratio = (moisture_percent / MOISTURE_OF_EXTINCTION_PERCENT).clamp(0.0, 1.0)
    damping = 1.0 - 2.59 * ratio + 5.11 * ratio.square() - 3.52 * ratio.pow(3)
    return damping.clamp(0.0, 1.0)


def simulate_batch(
    scenario_batch: list[dict[str, Any]],
    grid_size: int,
    cell_size_m: float,
    max_steps: int,
    device: torch.device,
    generator: torch.Generator,
    spotting_enabled: bool,
) -> dict[str, np.ndarray]:
    """Simulate a batch of scenarios in parallel on the selected device."""
    batch = len(scenario_batch)
    layers = create_forest_layers(
        scenario_batch, grid_size, cell_size_m, device, generator
    )
    trees = layers["trees"]
    tree_size = layers["tree_size"]
    surface_fuel = layers["surface_fuel"]
    canopy_fuel = layers["canopy_fuel"]
    canopy_local = layers["canopy_local"]
    canopy_base_height = layers["canopy_base_height_m"]

    wind_kmh = torch.tensor([s["wind_speed_kmh"] for s in scenario_batch],
                            dtype=torch.float32, device=device)
    wind_deg = torch.tensor([s["wind_direction_deg"] for s in scenario_batch],
                            dtype=torch.float32, device=device)
    moisture = torch.tensor([s["fuel_moisture_percent"] for s in scenario_batch],
                            dtype=torch.float32, device=device)
    slope_deg = torch.tensor([s["slope_degrees"] for s in scenario_batch],
                             dtype=torch.float32, device=device)
    slope_aspect = torch.tensor([s["slope_aspect_deg"] for s in scenario_batch],
                                dtype=torch.float32, device=device)

    wind_rad = torch.deg2rad(wind_deg)
    slope_rad = torch.deg2rad(slope_aspect)
    midflame_wind_ms = (wind_kmh / 3.6) * MIDFLAME_WIND_ADJUSTMENT
    moisture_damping = rothermel_form_moisture_damping(moisture)
    slope_tangent = torch.tan(torch.deg2rad(slope_deg))

    # Surface and canopy flames have separate timers and burned-state masks.
    surface_timer = torch.zeros((batch, grid_size, grid_size), dtype=torch.int16, device=device)
    crown_timer = torch.zeros_like(surface_timer)
    surface_burned = torch.zeros((batch, grid_size, grid_size), dtype=torch.bool, device=device)
    crown_burned = torch.zeros_like(surface_burned)
    thermal_dose = torch.zeros((batch, grid_size, grid_size), dtype=torch.float32, device=device)
    thermal_field = torch.zeros((batch, grid_size, grid_size), dtype=torch.float32, device=device)
    spotlight_count = torch.zeros((batch,), dtype=torch.int32, device=device)
    duration_steps = torch.zeros((batch,), dtype=torch.int32, device=device)

    # Point/line ignition near the western edge, centered vertically.
    y0 = max(0, grid_size // 2 - 1)
    y1 = min(grid_size, grid_size // 2 + 2)
    ignition_mask = surface_fuel[:, y0:y1, 0:2] > 0.05
    surface_timer[:, y0:y1, 0:2] = torch.where(
        ignition_mask,
        torch.full_like(surface_timer[:, y0:y1, 0:2], BURN_DURATION_STEPS),
        torch.zeros_like(surface_timer[:, y0:y1, 0:2]),
    )
    surface_burned[:, y0:y1, 0:2] = ignition_mask

    # Eight adjacent offsets for heat transfer.
    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),            (0, 1),
        (1, -1),  (1, 0),   (1, 1),
    ]

    # Eight downwind-capable spotting offsets, at a longer fixed range.
    spot_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),            (0, 1),
        (1, -1),  (1, 0),   (1, 1),
    ]

    # Precompute scenario-wise directional weights.
    neighbor_weights: list[torch.Tensor] = []
    advection_weights: list[torch.Tensor] = []
    for dy, dx in neighbor_offsets:
        distance = math.sqrt(dx * dx + dy * dy)
        ux = dx / distance
        uy = -dy / distance  # positive y points north in the directional math
        wind_alignment = ux * torch.cos(wind_rad) + uy * torch.sin(wind_rad)
        uphill_alignment = ux * torch.cos(slope_rad) + uy * torch.sin(slope_rad)
        wind_exponent = WIND_COEFFICIENT * midflame_wind_ms * wind_alignment
        slope_exponent = SLOPE_COEFFICIENT * slope_tangent * uphill_alignment
        directional_gain = torch.exp(
            wind_exponent.clamp(-1.5, 1.5) + slope_exponent.clamp(-1.5, 1.5)
        )
        neighbor_weights.append(
            HEAT_TRANSFER_COEFFICIENT * directional_gain / distance
        )
        advection_weights.append(wind_alignment.clamp(0.0, 1.0))

    advection_weight_sum = torch.stack(advection_weights, dim=0).sum(dim=0).clamp(min=1e-6)
    wind_advection_strength = WIND_ADVECTION_RATE * (
        midflame_wind_ms / 5.0
    ).clamp(0.0, 1.0)

    spot_weights: list[torch.Tensor] = []
    if spotting_enabled:
        wind_ratio_sq = (wind_kmh / 25.0).clamp(0.0, 2.0).square()
        for dy_unit, dx_unit in spot_offsets:
            distance_unit = math.sqrt(dx_unit * dx_unit + dy_unit * dy_unit)
            ux = dx_unit / distance_unit
            uy = -dy_unit / distance_unit
            alignment = (ux * torch.cos(wind_rad) + uy * torch.sin(wind_rad)).clamp(0.0, 1.0)
            probability = (
                SPOTTING_BASE_PROBABILITY
                * wind_ratio_sq
                * alignment.pow(6)
                * math.exp(-SPOTTING_OFFSET_CELLS / 4.0)
            )
            spot_weights.append(probability.clamp(0.0, 0.25))

    truncated = torch.zeros((batch,), dtype=torch.bool, device=device)

    for _step in range(max_steps):
        surface_active = surface_timer > 0
        crown_active = crown_timer > 0
        any_active_grid = surface_active | crown_active
        active_scenarios = any_active_grid.flatten(1).any(dim=1)
        pending_heat = thermal_field.flatten(1).amax(dim=1) > THERMAL_FIELD_STOP_THRESHOLD
        running_scenarios = active_scenarios | pending_heat
        if not bool(running_scenarios.any().item()):
            break
        duration_steps += running_scenarios.to(torch.int32)

        # Source flame power combines surface and crown contributions.
        source_power = (
            surface_active.float() * surface_fuel
            + crown_active.float() * canopy_fuel * CROWN_HEAT_MULTIPLIER
        )

        # Directional heat transferred from actively burning source cells.
        heat_flux = torch.zeros_like(surface_fuel)
        for (dy, dx), weight in zip(neighbor_offsets, neighbor_weights):
            heat_flux = heat_flux + shift_source_to_target(source_power, dy, dx) * weight[:, None, None]

        # Evolve a dimensionless thermal field using a finite-difference
        # diffusion, downwind-advection, heat-loss, and combustion-source update.
        cardinal_heat = (
            shift_source_to_target(thermal_field, -1, 0)
            + shift_source_to_target(thermal_field, 1, 0)
            + shift_source_to_target(thermal_field, 0, -1)
            + shift_source_to_target(thermal_field, 0, 1)
        )
        laplacian = cardinal_heat - 4.0 * thermal_field

        advected_heat = torch.zeros_like(thermal_field)
        for (dy, dx), weight in zip(neighbor_offsets, advection_weights):
            advected_heat = advected_heat + (
                shift_source_to_target(thermal_field, dy, dx) * weight[:, None, None]
            )
        advected_heat = advected_heat / advection_weight_sum[:, None, None]

        thermal_field = (
            thermal_field
            + THERMAL_DIFFUSIVITY * laplacian
            + wind_advection_strength[:, None, None] * (advected_heat - thermal_field)
            - THERMAL_HEAT_LOSS_PER_STEP * thermal_field
            + THERMAL_SOURCE_GAIN * source_power
        ).clamp(min=0.0)
        heat_for_ignition = heat_flux + THERMAL_FIELD_IGNITION_WEIGHT * thermal_field

        # Optional ember spotting: a wind-aligned, short-range stochastic jump.
        spot_pressure = torch.zeros_like(surface_fuel)
        if spotting_enabled:
            for (dy_unit, dx_unit), spot_weight in zip(spot_offsets, spot_weights):
                dy = dy_unit * SPOTTING_OFFSET_CELLS
                dx = dx_unit * SPOTTING_OFFSET_CELLS
                shifted = shift_source_to_target((source_power > 0).float(), dy, dx)
                spot_pressure = spot_pressure + shifted * spot_weight[:, None, None]

        # Available target fuel and moisture reduce or enhance ignition likelihood.
        surface_prob_local = 1.0 - torch.exp(
            -heat_for_ignition * moisture_damping[:, None, None]
            * surface_fuel / SURFACE_IGNITION_ENERGY_INDEX
        )
        surface_prob_spot = 1.0 - torch.exp(
            -spot_pressure * moisture_damping[:, None, None]
            * surface_fuel / max(SURFACE_IGNITION_ENERGY_INDEX, 1e-6)
        )
        surface_prob_total = 1.0 - (1.0 - surface_prob_local) * (1.0 - surface_prob_spot)
        surface_prob_total = surface_prob_total.clamp(0.0, 0.98)

        # Crown ignition requires tree canopy fuel and local canopy continuity.
        canopy_continuity_factor = (canopy_local / 1.5).clamp(0.0, 1.5)
        crown_heat = heat_for_ignition + spot_pressure * 0.65
        canopy_base_height_factor = 1.0 / (1.0 + 0.20 * canopy_base_height)
        crown_prob = 1.0 - torch.exp(
            -crown_heat * moisture_damping[:, None, None]
            * canopy_fuel * canopy_continuity_factor * canopy_base_height_factor
            / CROWN_IGNITION_ENERGY_INDEX
        )
        crown_prob = crown_prob.clamp(0.0, 0.90)

        rand_surface = paired_rand_tensor(
            batch, (grid_size, grid_size), device, generator
        )
        rand_crown = paired_rand_tensor(
            batch, (grid_size, grid_size), device, generator
        )
        surface_new = (
            (~surface_burned)
            & (surface_fuel > 0.05)
            & (rand_surface < surface_prob_total)
        )
        crown_new = (
            trees
            & (~crown_burned)
            & (canopy_fuel > 0.05)
            & (rand_crown < crown_prob)
        )

        # Count spread ignitions that came from the spotting component rather
        # than the adjacent-cell heat-transfer component.
        spotted_new = surface_new & (rand_surface >= surface_prob_local)
        spotlight_count += spotted_new.sum(dim=(1, 2)).to(torch.int32)

        # Exposure accumulates direct flame and neighbor heat; it drives a
        # simplified probabilistic tree-injury/mortality response at the end.
        thermal_dose = thermal_dose + (
            0.22 * heat_for_ignition
            + 0.24 * surface_active.float() * surface_fuel
            + 0.34 * crown_active.float() * canopy_fuel
        )

        # Burned cells do not reignite. Newly ignited cells receive a residence
        # timer, producing a short-lived flame front.
        surface_timer = torch.where(
            surface_timer > 0, surface_timer - 1, torch.zeros_like(surface_timer)
        )
        crown_timer = torch.where(
            crown_timer > 0, crown_timer - 1, torch.zeros_like(crown_timer)
        )
        surface_timer = torch.where(
            surface_new,
            torch.full_like(surface_timer, BURN_DURATION_STEPS),
            surface_timer,
        )
        crown_timer = torch.where(
            crown_new,
            torch.full_like(crown_timer, BURN_DURATION_STEPS),
            crown_timer,
        )
        surface_burned |= surface_new
        crown_burned |= crown_new

    still_active = (surface_timer > 0) | (crown_timer > 0)
    still_flaming = still_active.flatten(1).any(dim=1)
    heat_pending_at_end = (
        thermal_field.flatten(1).amax(dim=1) > THERMAL_FIELD_STOP_THRESHOLD
    )
    truncated = still_flaming | heat_pending_at_end

    # Tree mortality is an exposure-response proxy. Larger trees have a higher
    # resistance parameter, representing (very approximately) thicker bark.
    tree_resistance = TREE_RESISTANCE_BASE + TREE_RESISTANCE_SIZE_COEFFICIENT * tree_size
    mortality_logit = (thermal_dose - tree_resistance) / MORTALITY_DOSE_SCALE
    mortality_probability = torch.where(
        thermal_dose > 0.01,
        torch.sigmoid(mortality_logit),
        torch.zeros_like(thermal_dose),
    ).clamp(0.0, 1.0)
    survival_draw = paired_rand_tensor(
        batch, (grid_size, grid_size), device, generator
    )
    trees_surviving = trees & (survival_draw >= mortality_probability)
    trees_killed = trees & (~trees_surviving)

    trees_before = layers["trees_before_management"]
    trees_after = trees.sum(dim=(1, 2))
    trees_survive_count = trees_surviving.sum(dim=(1, 2))
    trees_killed_count = trees_killed.sum(dim=(1, 2))
    burned_cells = (surface_burned | crown_burned).sum(dim=(1, 2))
    crown_cells = crown_burned.sum(dim=(1, 2))
    burned_trees = (trees & (surface_burned | crown_burned)).sum(dim=(1, 2))
    tree_exposed = (trees & (thermal_dose > 0.05)).sum(dim=(1, 2))
    area_per_cell = cell_size_m ** 2
    burned_area = burned_cells.float() * area_per_cell
    domain_area = (grid_size * cell_size_m) ** 2
    max_x_index = torch.where(
        surface_burned | crown_burned,
        torch.arange(grid_size, device=device)[None, None, :],
        torch.zeros((batch, grid_size, grid_size), device=device),
    ).amax(dim=(1, 2))
    spread_distance = (max_x_index.float() + 1.0) * cell_size_m
    duration_minutes = duration_steps.float()
    spread_speed_proxy = spread_distance / duration_minutes.clamp(min=1.0)

    def cpu_array(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().to("cpu").numpy()

    return {
        "trees_before_management": cpu_array(trees_before),
        "trees_after_management": cpu_array(trees_after),
        "trees_removed_by_management": cpu_array(trees_before - trees_after),
        "trees_surviving_after_fire": cpu_array(trees_survive_count),
        "trees_killed_by_fire": cpu_array(trees_killed_count),
        "tree_survival_pct_of_post_treatment_stock": cpu_array(
            100.0 * trees_survive_count.float() / trees_after.float().clamp(min=1.0)
        ),
        "living_trees_pct_of_original_stock": cpu_array(
            100.0 * trees_survive_count.float() / trees_before.float().clamp(min=1.0)
        ),
        "trees_in_burned_cells": cpu_array(burned_trees),
        "trees_thermally_exposed": cpu_array(tree_exposed),
        "burned_cells": cpu_array(burned_cells),
        "burned_area_m2": cpu_array(burned_area),
        "burned_area_pct_of_grid": cpu_array(100.0 * burned_area / domain_area),
        "crown_ignited_tree_cells_proxy": cpu_array(crown_cells),
        "spotting_ignitions": cpu_array(spotlight_count),
        "fire_duration_minutes": cpu_array(duration_minutes),
        "max_eastward_extent_from_west_edge_m": cpu_array(spread_distance),
        "mean_eastward_extent_speed_proxy_m_min": cpu_array(spread_speed_proxy),
        "mean_tree_thermal_dose": cpu_array(
            (thermal_dose * trees.float()).sum(dim=(1, 2)) / trees.sum(dim=(1, 2)).float().clamp(min=1.0)
        ),
        "maximum_thermal_field_index": cpu_array(
            thermal_field.flatten(1).amax(dim=1)
        ),
        "simulation_truncated_at_max_steps": cpu_array(truncated),
        "actual_tree_density_after_management_trees_ha": cpu_array(
            trees_after.float() / domain_area * 10000.0
        ),
    }


# ---------------------------------------------------------------------------
# DATASET GENERATION AND REPORTS
# ---------------------------------------------------------------------------

def batched(items: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def build_output_rows(
    scenarios: list[dict[str, Any]],
    output_arrays: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, scenario in enumerate(scenarios):
        row = dict(scenario)
        for column, values in output_arrays.items():
            value = values[i]
            if isinstance(value, (np.bool_, bool)):
                value = bool(value)
            elif isinstance(value, (np.integer, int)):
                value = int(value)
            elif isinstance(value, (np.floating, float)):
                value = float(value)
            row[column] = value
        rows.append(row)
    return rows


def write_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    scenario_count: int,
    elapsed_seconds: float,
) -> None:
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        device_name = "Apple Metal Performance Shaders (MPS)"
    else:
        device_name = platform.processor() or "CPU"

    metadata = {
        "dataset_title": "Synthetic wildfire forest-management scenario dataset",
        "model_version": MODEL_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "data_type": "synthetic simulation-generated data; not observed wildfire measurements",
        "generation_method": "stochastic two-layer cellular-automaton simulation implemented in AI-assisted Python code",
        "ai_tool": "ChatGPT by OpenAI",
        "ai_model_reported_in_this_chat": "GPT-6",
        "ai_assistance_date": "2026-11-17",
        "ai_role": [
            "brainstorming the research framework",
            "drafting and revising Python simulation code",
            "explaining model assumptions and AI-use citation practices",
        ],
        "researcher_responsibilities": [
            "review and edit the generated code",
            "choose and justify parameter values",
            "calibrate against appropriate real-world data",
            "validate on separate held-out fires",
            "report limitations and preserve the exact final code used",
        ],
        "hardware": {
            "requested_device": args.device,
            "actual_device": str(device),
            "device_name": device_name,
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        },
        "experiment": {
            "seed": args.seed,
            "grid_size_cells": args.grid_size,
            "cell_size_m": args.cell_size_m,
            "domain_width_m": args.grid_size * args.cell_size_m,
            "domain_height_m": args.grid_size * args.cell_size_m,
            "time_step_minutes": 1,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "replicates_per_parameter_combination": args.replicates,
            "scenario_count": scenario_count,
            "quick_smoke_test_mode": args.quick,
            "paired_treatment_design": True,
            "common_random_numbers_across_the_four_treatments": True,
            "spotting_enabled": args.spotting,
        },
        "model_parameters": {
            "moisture_of_extinction_percent_assumption": MOISTURE_OF_EXTINCTION_PERCENT,
            "thin_size_cutoff": THIN_SIZE_CUTOFF,
            "thin_slash_multiplier": THIN_SLASH_MULTIPLIER,
            "surface_fuel_remaining_after_treatment": SURFACE_FUEL_REMAINING_AFTER_TREATMENT,
            "heat_transfer_coefficient": HEAT_TRANSFER_COEFFICIENT,
            "surface_ignition_energy_index": SURFACE_IGNITION_ENERGY_INDEX,
            "crown_ignition_energy_index": CROWN_IGNITION_ENERGY_INDEX,
            "crown_heat_multiplier": CROWN_HEAT_MULTIPLIER,
            "wind_coefficient": WIND_COEFFICIENT,
            "slope_coefficient": SLOPE_COEFFICIENT,
            "midflame_wind_adjustment": MIDFLAME_WIND_ADJUSTMENT,
            "thermal_diffusivity_dimensionless": THERMAL_DIFFUSIVITY,
            "thermal_heat_loss_per_step": THERMAL_HEAT_LOSS_PER_STEP,
            "thermal_source_gain": THERMAL_SOURCE_GAIN,
            "wind_advection_rate": WIND_ADVECTION_RATE,
            "thermal_field_ignition_weight": THERMAL_FIELD_IGNITION_WEIGHT,
            "thermal_field_stop_threshold": THERMAL_FIELD_STOP_THRESHOLD,
            "burn_duration_steps": BURN_DURATION_STEPS,
            "spotting_base_probability": SPOTTING_BASE_PROBABILITY,
            "spotting_offset_cells": SPOTTING_OFFSET_CELLS,
            "tree_resistance_base": TREE_RESISTANCE_BASE,
            "tree_resistance_size_coefficient": TREE_RESISTANCE_SIZE_COEFFICIENT,
            "mortality_dose_scale": MORTALITY_DOSE_SCALE,
        },
        "important_model_limitations": [
            "This is not a full implementation of the Rothermel surface-fire spread model.",
            "The moisture damping term uses a Rothermel-form cubic relationship, with a chosen 30% extinction-moisture assumption.",
            "The evolving thermal field is a dimensionless finite-difference diffusion/advection/heat-loss proxy, not a calibrated temperature field in degrees or a combustion-chemistry calculation.",
            "Directional wind, slope, heat-transfer, crown ignition (including the canopy-base-height modifier), spotting, and mortality relationships are simplified tunable proxies.",
            "Tree size is only a crude resistance proxy; species, bark thickness, crown base height, and post-fire mortality are not empirically resolved.",
            "The grid represents a small idealized forest patch with spatially simplified terrain and weather.",
            "All treatment effects and coefficients must be calibrated and independently validated before real-world inference.",
        ],
        "calibration_recommendation": {
            "fuels_and_canopy_inputs": "LANDFIRE fuel and vegetation products",
            "burned_area_and_severity_comparison": "Monitoring Trends in Burn Severity (MTBS)",
            "warning": "MTBS perimeter/severity agreement does not by itself validate individual-tree survival or treatment effectiveness.",
        },
        "runtime_seconds": elapsed_seconds,
        "citation_note": "Use the companion README_AI_AND_CITATION.md and replace its chat-URL placeholder with a share link generated by the user.",
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU-accelerated synthetic wildfire forest-management simulation."
    )
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--cell-size-m", type=float, default=DEFAULT_CELL_SIZE_M)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=str, default="wildfire_output_gpu")
    parser.add_argument("--quick", action="store_true", help="Run a small smoke-test experiment.")
    parser.add_argument("--no-spotting", dest="spotting", action="store_false")
    parser.add_argument("--spotting", dest="spotting", action="store_true")
    parser.set_defaults(spotting=SPOTTING_ENABLED_BY_DEFAULT)
    args = parser.parse_args()

    if args.grid_size < 12:
        parser.error("--grid-size must be at least 12.")
    if args.cell_size_m <= 0:
        parser.error("--cell-size-m must be positive.")
    if args.max_steps < 1 or args.batch_size < 1 or args.replicates < 1:
        parser.error("--max-steps, --batch-size, and --replicates must be positive.")
    if args.batch_size % len(MANAGEMENT_STRATEGIES) != 0:
        parser.error("--batch-size must be divisible by 4 so treatment blocks stay paired.")

    if args.quick:
        args.grid_size = 24
        args.max_steps = 180
        args.batch_size = 16
        args.replicates = 1

    device = choose_device(args.device)
    # Fixed seeds for repeatability of the same software/device/configuration.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    generator = make_generator(device, args.seed)
    scenarios = quick_scenarios(args.replicates) if args.quick else build_scenarios(args.replicates)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Model: {MODEL_VERSION}")
    print(f"Grid: {args.grid_size} x {args.grid_size} cells; {args.cell_size_m:g} m/cell")
    print(f"Scenarios: {len(scenarios):,}; batch size: {args.batch_size}; spotting: {args.spotting}")

    all_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    total_batches = math.ceil(len(scenarios) / args.batch_size)

    for batch_number, scenario_batch in enumerate(batched(scenarios, args.batch_size), start=1):
        arrays = simulate_batch(
            scenario_batch=scenario_batch,
            grid_size=args.grid_size,
            cell_size_m=args.cell_size_m,
            max_steps=args.max_steps,
            device=device,
            generator=generator,
            spotting_enabled=args.spotting,
        )
        all_rows.extend(build_output_rows(scenario_batch, arrays))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        print(
            f"Batch {batch_number:>3}/{total_batches} complete "
            f"({min(batch_number * args.batch_size, len(scenarios)):,}/{len(scenarios):,} scenarios); "
            f"elapsed {elapsed:.1f}s"
        )

    elapsed_seconds = time.perf_counter() - started
    df = pd.DataFrame(all_rows)
    csv_path = output_dir / "wildfire_scenarios.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")

    summary = (
        df.groupby("management", as_index=False)
        .agg(
            mean_living_trees_pct_of_original_stock=("living_trees_pct_of_original_stock", "mean"),
            sd_living_trees_pct_of_original_stock=("living_trees_pct_of_original_stock", "std"),
            mean_tree_survival_pct_of_post_treatment_stock=("tree_survival_pct_of_post_treatment_stock", "mean"),
            mean_burned_area_pct=("burned_area_pct_of_grid", "mean"),
            mean_trees_removed_by_management=("trees_removed_by_management", "mean"),
            mean_crown_ignited_tree_cells_proxy=("crown_ignited_tree_cells_proxy", "mean"),
            scenarios=("scenario_id", "count"),
        )
        .sort_values("mean_living_trees_pct_of_original_stock", ascending=False)
    )
    summary_path = output_dir / "management_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.6f")

    # A stratified table matters more than one overall winner: management may
    # perform differently under different wind/moisture/density combinations.
    by_conditions = (
        df.groupby([
            "baseline_tree_density_trees_ha", "tree_arrangement", "wind_speed_kmh",
            "wind_direction_deg", "fuel_moisture_percent", "slope_degrees", "management",
        ], as_index=False)
        .agg(
            mean_living_trees_pct_of_original_stock=("living_trees_pct_of_original_stock", "mean"),
            mean_burned_area_pct=("burned_area_pct_of_grid", "mean"),
            replicates=("scenario_id", "count"),
        )
    )
    conditions_path = output_dir / "management_by_conditions.csv"
    by_conditions.to_csv(conditions_path, index=False, float_format="%.6f")

    write_metadata(output_dir, args, device, len(df), elapsed_seconds)

    print("\nFinished.")
    print(f"Dataset: {csv_path.resolve()}")
    print(f"Overall summary: {summary_path.resolve()}")
    print(f"Condition-specific summary: {conditions_path.resolve()}")
    print(f"Metadata: {(output_dir / 'run_metadata.json').resolve()}")
    print("\nMean living trees retained relative to the original, pre-treatment forest:")
    print(summary[["management", "mean_living_trees_pct_of_original_stock"]].to_string(index=False))
    if df["simulation_truncated_at_max_steps"].any():
        n_truncated = int(df["simulation_truncated_at_max_steps"].sum())
        print(
            f"\nWARNING: {n_truncated} scenario(s) still had active flames at the "
            "maximum step. Increase --max-steps and rerun before analysis."
        )


if __name__ == "__main__":
    main()
