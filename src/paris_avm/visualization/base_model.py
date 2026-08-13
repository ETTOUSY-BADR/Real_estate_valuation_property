"""Create honest, data-driven visuals for the valuation model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from paris_avm.paths import PROJECT_ROOT


NAVY = "#0B2E59"
BLUE = "#2B6CB0"
CORAL = "#EF5B5B"
PALE = "#EAF1F8"
GRAY = "#667085"


def euro_thousands(values: np.ndarray | pd.Series) -> np.ndarray:
    return np.asarray(values) / 1_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", type=Path, default=PROJECT_ROOT / "reports/test_predictions.csv"
    )
    parser.add_argument("--metrics", type=Path, default=PROJECT_ROOT / "reports/metrics.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "visuals")
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    report = json.loads(args.metrics.read_text(encoding="utf-8"))
    predictions["signed_percentage_error"] = (
        (predictions["predicted_value"] - predictions["valeur_fonciere"])
        / predictions["valeur_fonciere"]
        * 100
    )
    predictions["absolute_percentage_error"] = predictions[
        "signed_percentage_error"
    ].abs()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelcolor": NAVY,
            "axes.edgecolor": "#CBD5E1",
            "xtick.color": GRAY,
            "ytick.color": GRAY,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    fig.suptitle(
        "Paris Apartment Valuation Model — Honest Holdout Performance",
        fontsize=20,
        color=NAVY,
    )
    fig.text(
        0.5,
        0.955,
        f"{report['test_rows']:,} unseen sales · {report['date_range']['test_start']} to {report['date_range']['test_end']}",
        ha="center",
        color=GRAY,
        fontsize=11,
    )

    # Actual vs predicted price.
    ax = axes[0, 0]
    display_limit = float(
        np.quantile(
            np.concatenate(
                [
                    predictions["valeur_fonciere"].to_numpy(),
                    predictions["predicted_value"].to_numpy(),
                ]
            ),
            0.985,
        )
    )
    visible = predictions.loc[
        predictions["valeur_fonciere"].le(display_limit)
        & predictions["predicted_value"].le(display_limit)
    ]
    ax.scatter(
        euro_thousands(visible["valeur_fonciere"]),
        euro_thousands(visible["predicted_value"]),
        s=12,
        alpha=0.28,
        color=BLUE,
        edgecolors="none",
    )
    limit_k = display_limit / 1_000
    ax.plot(
        [0, limit_k],
        [0, limit_k],
        color=CORAL,
        linewidth=2,
        label="Perfect prediction",
    )
    ax.set(
        xlim=(0, limit_k),
        ylim=(0, limit_k),
        xlabel="Actual sale price (€000)",
        ylabel="Predicted price (€000)",
    )
    ax.set_title("Actual vs predicted price")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.18)
    metrics = report["model_metrics"]
    ax.text(
        0.97,
        0.05,
        f"R²  {metrics['r2']:.3f}\nMedian error  {metrics['median_absolute_percentage_error']:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=NAVY,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=PALE, edgecolor="none"),
    )

    # Error distribution.
    ax = axes[0, 1]
    clipped_ape = predictions["absolute_percentage_error"].clip(upper=80)
    ax.hist(clipped_ape, bins=np.arange(0, 82, 2), color=BLUE, alpha=0.85)
    ax.axvline(10, color=NAVY, linestyle="--", linewidth=1.6)
    ax.axvline(20, color=CORAL, linestyle="--", linewidth=1.6)
    ax.set(
        xlabel="Absolute percentage error (values above 80% clipped)",
        ylabel="Number of sales",
    )
    ax.set_title("How far predictions miss")
    ax.grid(axis="y", alpha=0.18)
    ax.text(
        0.97,
        0.92,
        f"Within 10%: {metrics['within_10_percent']:.1%}\nWithin 20%: {metrics['within_20_percent']:.1%}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=NAVY,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=PALE, edgecolor="none"),
    )

    # Error by arrondissement.
    ax = axes[1, 0]
    by_postcode = (
        predictions.groupby("code_postal")
        .agg(
            median_ape=("absolute_percentage_error", "median"),
            sales=("id_mutation", "size"),
        )
        .sort_index()
    )
    positions = np.arange(len(by_postcode))
    bars = ax.bar(positions, by_postcode["median_ape"], color=BLUE)
    ax.set_xticks(positions, [str(int(code))[-2:] for code in by_postcode.index])
    ax.set(
        xlabel="Arrondissement",
        ylabel="Median absolute percentage error",
    )
    ax.set_title("Typical error by arrondissement")
    ax.grid(axis="y", alpha=0.18)
    for bar, count in zip(bars, by_postcode["sales"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.35,
            f"n={count}",
            ha="center",
            va="bottom",
            fontsize=7,
            color=GRAY,
            rotation=90,
        )

    # Permutation importance.
    ax = axes[1, 1]
    importance = pd.Series(
        report["feature_importance_mae_increase_eur"]
    ).sort_values().tail(9)
    friendly_names = {
        "surface_reelle_bati": "Surface",
        "nombre_pieces_principales": "Rooms",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "adresse_numero": "Street number",
        "nombre_lots": "Lots",
        "month": "Month",
        "day_of_year": "Day of year",
        "year": "Year",
        "months_since_2021": "Time index",
        "code_postal": "Arrondissement",
    }
    labels = [friendly_names.get(name, name) for name in importance.index]
    colors = [CORAL if value == importance.max() else BLUE for value in importance.values]
    ax.barh(labels, importance.values / 1_000, color=colors)
    ax.set(xlabel="Increase in holdout MAE when shuffled (€000)")
    ax.set_title("What information the model relies on")
    ax.grid(axis="x", alpha=0.18)

    dashboard_path = args.output_dir / "model_performance_dashboard.png"
    fig.savefig(dashboard_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Geographic pattern of over- and under-valuation.
    fig, ax = plt.subplots(figsize=(11, 10), constrained_layout=True)
    signed_error = predictions["signed_percentage_error"].clip(-50, 50)
    scatter = ax.scatter(
        predictions["longitude"],
        predictions["latitude"],
        c=signed_error,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-50, vcenter=0, vmax=50),
        s=14,
        alpha=0.62,
        edgecolors="none",
    )
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.78, pad=0.02)
    colorbar.set_label("Prediction error (%) · blue = under · red = over")
    ax.set(
        xlabel="Longitude",
        ylabel="Latitude",
        title="Where the model overvalues and undervalues Paris apartments",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.12)
    ax.text(
        0.01,
        0.01,
        "Each point is an unseen 2025 sale · colors clipped to ±50%",
        transform=ax.transAxes,
        color=GRAY,
        fontsize=10,
    )
    map_path = args.output_dir / "prediction_error_map.png"
    fig.savefig(map_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {dashboard_path}")
    print(f"Saved: {map_path}")


if __name__ == "__main__":
    main()
