"""
Graph wildfire simulation results.

Install:
    python -m pip install pandas matplotlib numpy

Run:
    python plot_wildfire_results.py
Or:
    python plot_wildfire_results.py --input wildfire_output/wildfire_scenarios.csv

Creates PNG graphs in ./wildfire_graphs/.
"""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def first_col(df, options, required=True):
    for col in options:
        if col in df.columns:
            return col
    if required:
        raise ValueError(
            f"Missing expected column. Tried {options}. "
            f"CSV columns: {list(df.columns)}"
        )
    return None


def save(fig, out, filename):
    out.mkdir(parents=True, exist_ok=True)
    target = out / filename
    fig.savefig(target, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {target}")


def mean_ci(df, groups, metric):
    s = df.groupby(groups)[metric].agg(["mean", "std", "count"]).reset_index()
    s["ci"] = 1.96 * s["std"].fillna(0) / np.sqrt(s["count"].clip(lower=1))
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("wildfire_graphs"))
    args = parser.parse_args()

    if args.input is None:
        candidates = [
            Path("wildfire_output/wildfire_scenarios.csv"),
            Path("wildfire_scenarios.csv"),
        ]
        csv_path = next((p for p in candidates if p.exists()), None)
    else:
        csv_path = args.input

    if csv_path is None or not csv_path.exists():
        sys.exit(
            "Could not find wildfire_scenarios.csv. Specify it with "
            "--input path\\to\\wildfire_scenarios.csv"
        )

    df = pd.read_csv(csv_path)
    if df.empty:
        sys.exit("The CSV has no rows.")

    management = first_col(df, ["management", "management_strategy"])
    survival = first_col(df, [
        "living_trees_pct_of_original_stock",
        "tree_survival_pct_of_post_treatment_stock",
        "tree_survival_percent",
        "tree_survival_pct",
    ])
    burned = first_col(df, ["burned_area_m2", "burned_area", "burned_area_cells"])
    wind = first_col(df, ["wind_speed_kmh", "wind_speed"], required=False)
    moisture = first_col(df, ["fuel_moisture_percent", "fuel_moisture"], required=False)
    arrangement = first_col(df, ["tree_arrangement", "arrangement"], required=False)
    density = first_col(df, [
        "baseline_tree_density_trees_ha", "tree_density_trees_ha", "tree_density"
    ], required=False)

    df = df.copy()
    df["_survival"] = pd.to_numeric(df[survival], errors="coerce")
    df["_burned"] = pd.to_numeric(df[burned], errors="coerce")
    # Convert proportions to percentages if needed.
    if df["_survival"].max(skipna=True) <= 1:
        df["_survival"] *= 100

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(df):,} rows from {csv_path.resolve()}")

    # 1. Mean survival by management, with approximate 95% CI.
    stats = mean_ci(df, [management], "_survival").sort_values("mean", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(stats))
    ax.bar(x, stats["mean"], yerr=stats["ci"], capsize=5, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v).replace("_", " ").title() for v in stats[management]],
                       rotation=15, ha="right")
    ax.set_ylabel("Tree survival (%)")
    ax.set_title("Tree survival by forest-management strategy")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    save(fig, out, "01_management_survival.png")

    # 2. Burned area by management.
    stats = mean_ci(df, [management], "_burned").sort_values("mean")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(stats))
    ax.barh(y, stats["mean"], xerr=stats["ci"], capsize=5, edgecolor="black")
    ax.set_yticks(y)
    ax.set_yticklabels([str(v).replace("_", " ").title() for v in stats[management]])
    ax.set_xlabel("Burned area (m²)")
    ax.set_title("Burned area by forest-management strategy")
    ax.grid(axis="x", alpha=.25)
    fig.tight_layout()
    save(fig, out, "02_management_burned_area.png")

    # 3. Weather heatmap: average survival at each wind/moisture combination.
    if wind and moisture:
        table = df.pivot_table(index=moisture, columns=wind, values="_survival", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(8, 5.5))
        im = ax.imshow(table.values, origin="lower", aspect="auto")
        ax.set_xticks(range(len(table.columns)), labels=[str(v) for v in table.columns])
        ax.set_yticks(range(len(table.index)), labels=[str(v) for v in table.index])
        ax.set_xlabel("Wind speed (km/h)")
        ax.set_ylabel("Fuel moisture (%)")
        ax.set_title("Mean tree survival across weather conditions")
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("Tree survival (%)")
        for i in range(len(table.index)):
            for j in range(len(table.columns)):
                val = table.iloc[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center")
        fig.tight_layout()
        save(fig, out, "03_survival_weather_heatmap.png")

    # 4. Arrangement x management comparison.
    if arrangement:
        table = df.pivot_table(index=arrangement, columns=management,
                               values="_survival", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(10, 5.5))
        table.plot(kind="bar", ax=ax)
        ax.set_xlabel("Forest arrangement")
        ax.set_ylabel("Tree survival (%)")
        ax.set_title("Forest geometry and management strategy")
        ax.set_xticklabels([str(v).replace("_", " ").title() for v in table.index],
                           rotation=0)
        ax.legend(title="Management", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        save(fig, out, "04_arrangement_comparison.png")

    # 5. Survival as a function of original tree density.
    if density:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for strategy, group in df.groupby(management):
            means = group.groupby(density)["_survival"].mean().sort_index()
            ax.plot(means.index, means.values, marker="o", label=str(strategy).replace("_", " ").title())
        ax.set_xlabel("Baseline tree density (trees/ha)")
        ax.set_ylabel("Tree survival (%)")
        ax.set_title("Tree density and forest preservation")
        ax.legend(title="Management")
        ax.grid(alpha=.25)
        fig.tight_layout()
        save(fig, out, "05_survival_vs_density.png")

    # 6. Distribution of burned area.
    strategies = list(df[management].dropna().unique())
    values = [df.loc[df[management] == s, "_burned"].dropna().values for s in strategies]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.boxplot(values, tick_labels=[str(s).replace("_", " ").title() for s in strategies],
               showfliers=False)
    ax.set_ylabel("Burned area (m²)")
    ax.set_title("Burned-area distribution by management strategy")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    save(fig, out, "06_burned_area_distribution.png")

    print(f"\nFinished. Graphs are in: {out.resolve()}")
    if survival == "tree_survival_pct_of_post_treatment_stock":
        print("Note: your CSV measures survival relative to post-treatment trees,")
        print("not the original forest stock. Interpret this distinction carefully.")


if __name__ == "__main__":
    main()
