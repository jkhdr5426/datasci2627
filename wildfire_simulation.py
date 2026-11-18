"""
Wildfire forest-management simulation
-------------------------------------
Creates a synthetic, simulation-generated dataset for comparing forest
management strategies under different wildfire conditions.

Install dependencies:
    python -m pip install numpy pandas

Run:
    python wildfire_simulation.py

Outputs (in a folder named wildfire_output):
    wildfire_scenarios.csv  - one row per simulated wildfire
    run_metadata.txt        - model assumptions and parameter settings

IMPORTANT:
This is an educational, probabilistic cellular-automaton model. It is not
a validated operational wildfire-prediction tool. Its outputs are simulated,
not observed real-world measurements.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd


# --------------------------- USER SETTINGS ---------------------------

SEED = 42
GRID_SIZE = 40             # 40 x 40 cells
CELL_SIZE_M = 2.0          # each cell represents a 2 m x 2 m area
MAX_STEPS = 100            # maximum simulated time steps per fire
REPLICATES = 2             # increase to 10-30 for a larger dataset

TREE_DENSITIES = [250, 500, 750]       # trees per hectare, before management
ARRANGEMENTS = ["random", "regular"]
WIND_SPEEDS = [5, 15, 25]              # km/h
FUEL_MOISTURES = [5, 15]               # percent
SLOPES = [0, 20]                       # degrees
MANAGEMENT_STRATEGIES = [
    "untreated",
    "tree_thinning",
    "ground_fuel_removal",
    "combined",
]

# Model calibration/tuning parameters.
# These are dimensionless model coefficients, not universal physical constants.
BASE_SPREAD_PROBABILITY = 0.16
TREE_FUEL_CONTRIBUTION = 0.80
GROUND_FUEL_BASELINE = 1.00
THINNING_FRACTION = 0.30
GROUND_FUEL_REMOVAL_FRACTION = 0.65

OUTPUT_DIR = Path("wildfire_output")


# --------------------------- FOREST CREATION ---------------------------

def create_forest(density_per_ha, arrangement, rng):
    """Return a Boolean grid indicating which cells contain living trees."""
    n = GRID_SIZE
    cell_area_m2 = CELL_SIZE_M ** 2
    target_probability = density_per_ha * cell_area_m2 / 10000.0

    if arrangement == "random":
        trees = rng.random((n, n)) < target_probability

    elif arrangement == "regular":
        # Approximate equal spacing from the requested trees-per-hectare value.
        spacing_m = math.sqrt(10000.0 / density_per_ha)
        spacing_cells = max(1, round(spacing_m / CELL_SIZE_M))
        row_offset = int(rng.integers(0, spacing_cells))
        col_offset = int(rng.integers(0, spacing_cells))
        trees = np.zeros((n, n), dtype=bool)
        trees[row_offset::spacing_cells, col_offset::spacing_cells] = True

    else:
        raise ValueError(f"Unknown arrangement: {arrangement}")

    return trees


# --------------------------- FIRE SIMULATION ---------------------------

def shifted_slices(n, dr, dc):
    """Slices mapping a source cell to its neighbor (dr, dc) away."""
    source_r = slice(max(0, -dr), min(n, n - dr))
    source_c = slice(max(0, -dc), min(n, n - dc))
    target_r = slice(max(0, dr), min(n, n + dr))
    target_c = slice(max(0, dc), min(n, n + dc))
    return source_r, source_c, target_r, target_c


def simulate_fire(
    trees,
    wind_speed,
    wind_direction_deg,
    fuel_moisture,
    slope_deg,
    slope_aspect_deg,
    ground_fuel_fraction,
    rng,
):
    """
    Simulate a fire ignited along the left edge of the grid.

    Cell states:
        0 = unburned
        1 = actively burning
        2 = burned

    Fire spreads probabilistically to neighboring cells. The probability
    depends on target-cell fuel, moisture, wind alignment, and uphill direction.
    """
    n = GRID_SIZE
    state = np.zeros((n, n), dtype=np.uint8)

    # Fuel available in each cell. Trees add combustible material.
    ground_fuel = GROUND_FUEL_BASELINE * ground_fuel_fraction
    fuel = ground_fuel + trees.astype(float) * TREE_FUEL_CONTRIBUTION

    # Ignite combustible cells along the left boundary.
    state[:, 0] = np.where(fuel[:, 0] > 0.05, 1, 0)

    # Environmental multipliers. These are simplified empirical relationships.
    moisture_factor = math.exp(-0.035 * (fuel_moisture - 10.0))

    wind_rad = math.radians(wind_direction_deg)
    slope_rad = math.radians(slope_aspect_deg)

    # Eight neighboring directions.
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    steps = 0

    for step in range(MAX_STEPS):
        burning = state == 1
        if not burning.any():
            break

        new_fire = np.zeros((n, n), dtype=bool)

        for dr, dc in directions:
            sr, sc, tr, tc = shifted_slices(n, dr, dc)

            # Direction vector from source to target.
            distance = math.sqrt(dr * dr + dc * dc)
            direction_x = dc / distance
            direction_y = -dr / distance

            # Wind direction is the direction the wind is blowing TOWARD.
            wind_alignment = (
                direction_x * math.cos(wind_rad)
                + direction_y * math.sin(wind_rad)
            )
            wind_factor = math.exp(0.045 * wind_speed * wind_alignment)
            wind_factor = min(wind_factor, 3.0)

            # Slope aspect is the direction pointing uphill.
            uphill_alignment = (
                direction_x * math.cos(slope_rad)
                + direction_y * math.sin(slope_rad)
            )
            slope_factor = math.exp(
                0.025 * slope_deg * uphill_alignment
            )

            # Diagonal spread is slightly reduced because the cells are farther apart.
            distance_factor = 1.0 / distance

            probability = (
                BASE_SPREAD_PROBABILITY
                * fuel[tr, tc]
                * moisture_factor
                * wind_factor
                * slope_factor
                * distance_factor
            )
            probability = np.clip(probability, 0.0, 0.95)

            source_is_burning = burning[sr, sc]
            target_is_unburned = state[tr, tc] == 0
            random_draw = rng.random(source_is_burning.shape)

            successful_spread = (
                source_is_burning
                & target_is_unburned
                & (random_draw < probability)
                & (fuel[tr, tc] > 0.05)
            )

            new_fire[tr, tc] |= successful_spread

        # Current fire cells finish burning; newly ignited cells burn next step.
        state[burning] = 2
        state[new_fire] = 1
        steps = step + 1

    burned = state > 0
    trees_burned = int(np.count_nonzero(trees & burned))
    initial_trees = int(np.count_nonzero(trees))
    surviving_trees = initial_trees - trees_burned

    return {
        "initial_trees": initial_trees,
        "trees_burned": trees_burned,
        "surviving_trees": surviving_trees,
        "tree_survival_percent": (
            100.0 * surviving_trees / initial_trees
            if initial_trees else 0.0
        ),
        "burned_cells": int(np.count_nonzero(burned)),
        "burned_area_m2": float(np.count_nonzero(burned) * CELL_SIZE_M**2),
        "burned_area_percent": 100.0 * np.mean(burned),
        "fire_duration_steps": steps,
    }


# --------------------------- EXPERIMENT DESIGN ---------------------------

def apply_management(base_density, strategy):
    """Return managed tree density and ground-fuel fraction."""
    density = float(base_density)
    ground_fuel_fraction = 1.0

    if strategy in ("tree_thinning", "combined"):
        density *= (1.0 - THINNING_FRACTION)

    if strategy in ("ground_fuel_removal", "combined"):
        ground_fuel_fraction *= (1.0 - GROUND_FUEL_REMOVAL_FRACTION)

    return density, ground_fuel_fraction


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    master_rng = np.random.default_rng(SEED)

    records = []
    scenario_id = 0

    total_runs = (
        len(TREE_DENSITIES)
        * len(ARRANGEMENTS)
        * len(WIND_SPEEDS)
        * len(FUEL_MOISTURES)
        * len(SLOPES)
        * len(MANAGEMENT_STRATEGIES)
        * REPLICATES
    )
    print(f"Generating {total_runs} simulated wildfire scenarios...")

    for base_density in TREE_DENSITIES:
        for arrangement in ARRANGEMENTS:
            for wind_speed in WIND_SPEEDS:
                for moisture in FUEL_MOISTURES:
                    for slope in SLOPES:
                        for strategy in MANAGEMENT_STRATEGIES:
                            managed_density, ground_fuel_fraction = apply_management(
                                base_density, strategy
                            )

                            for replicate in range(1, REPLICATES + 1):
                                scenario_id += 1
                                # Independent reproducible random stream per scenario.
                                rng = np.random.default_rng(
                                    int(master_rng.integers(0, 2**32 - 1))
                                )

                                trees = create_forest(
                                    managed_density, arrangement, rng
                                )

                                # Wind and uphill direction are varied between scenarios.
                                wind_direction = 0.0  # fire spreads generally left-to-right
                                slope_aspect = 0.0    # uphill is left-to-right

                                results = simulate_fire(
                                    trees=trees,
                                    wind_speed=wind_speed,
                                    wind_direction_deg=wind_direction,
                                    fuel_moisture=moisture,
                                    slope_deg=slope,
                                    slope_aspect_deg=slope_aspect,
                                    ground_fuel_fraction=ground_fuel_fraction,
                                    rng=rng,
                                )

                                records.append({
                                    "scenario_id": scenario_id,
                                    "replicate": replicate,
                                    "management": strategy,
                                    "tree_arrangement": arrangement,
                                    "baseline_tree_density_trees_ha": base_density,
                                    "managed_target_density_trees_ha": managed_density,
                                    "actual_initial_tree_count": results["initial_trees"],
                                    "wind_speed_kmh": wind_speed,
                                    "wind_direction_deg": wind_direction,
                                    "fuel_moisture_percent": moisture,
                                    "slope_degrees": slope,
                                    "ground_fuel_remaining_fraction": ground_fuel_fraction,
                                    **results,
                                })

        print(f"Completed density level: {base_density} trees/ha")

    df = pd.DataFrame(records)
    csv_path = OUTPUT_DIR / "wildfire_scenarios.csv"
    df.to_csv(csv_path, index=False)

    metadata = f"""WILDFIRE SIMULATION DATASET METADATA

Dataset type:
    Synthetic, simulation-generated data. These are not observed wildfire events.

AI disclosure:
    Record the AI tool, model/version, date, and exact prompts used to help
    create or revise this code. The simulation code generates the dataset.

Random seed:
    {SEED}

Grid:
    {GRID_SIZE} x {GRID_SIZE} cells
    Cell width and height: {CELL_SIZE_M} m
    Total simulated area: {GRID_SIZE * CELL_SIZE_M} m x {GRID_SIZE * CELL_SIZE_M} m

Experimental settings:
    Tree densities (trees/ha): {TREE_DENSITIES}
    Tree arrangements: {ARRANGEMENTS}
    Wind speeds (km/h): {WIND_SPEEDS}
    Fuel moisture (%): {FUEL_MOISTURES}
    Slopes (degrees): {SLOPES}
    Management strategies: {MANAGEMENT_STRATEGIES}
    Replicates per parameter combination: {REPLICATES}
    Maximum fire steps: {MAX_STEPS}

Management assumptions:
    Tree thinning removes {THINNING_FRACTION:.0%} of target tree density.
    Ground-fuel removal removes {GROUND_FUEL_REMOVAL_FRACTION:.0%} of ground fuel.

Model limitations:
    This is a simplified probabilistic cellular automaton, not a full
    computational-fluid-dynamics or combustion-chemistry model. Wind, moisture,
    slope, and fuel effects use tunable simplified relationships. The model
    should be calibrated and validated against suitable real-world datasets
    before making claims about actual forest outcomes.
"""
    (OUTPUT_DIR / "run_metadata.txt").write_text(metadata, encoding="utf-8")

    print("\nDone.")
    print(f"Dataset saved to: {csv_path.resolve()}")
    print(f"Metadata saved to: {(OUTPUT_DIR / 'run_metadata.txt').resolve()}")
    print(f"Rows generated: {len(df)}")

    print("\nAverage tree survival by management strategy:")
    summary = (
        df.groupby("management")["tree_survival_percent"]
        .mean()
        .sort_values(ascending=False)
        .round(2)
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
