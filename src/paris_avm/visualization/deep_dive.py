"""Generate deeper segment, bias, market, and uncertainty diagnostics."""

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
LIGHT_BLUE = "#8CB9E5"
CORAL = "#EF5B5B"
GREEN = "#2A9D8F"
GRAY = "#667085"
PALE = "#EDF3F8"


def postcode_label(value: object) -> str:
    return str(value)[-2:]


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    spread = z * np.sqrt(
        proportion * (1 - proportion) / total + z**2 / (4 * total**2)
    ) / denominator
    return center - spread, center + spread


def load_data(predictions_path: Path) -> pd.DataFrame:
    data = pd.read_csv(predictions_path, dtype={"code_postal": "string"})
    data["date_mutation"] = pd.to_datetime(data["date_mutation"])
    data = data.sort_values(["date_mutation", "id_mutation"]).reset_index(drop=True)
    data["actual_price_per_m2"] = (
        data["valeur_fonciere"] / data["surface_reelle_bati"]
    )
    data["predicted_price_per_m2"] = (
        data["predicted_value"] / data["surface_reelle_bati"]
    )
    data["signed_error_percent"] = (
        (data["predicted_value"] - data["valeur_fonciere"])
        / data["valeur_fonciere"]
        * 100
    )
    data["absolute_percentage_error"] = data["signed_error_percent"].abs()
    data["month"] = data["date_mutation"].dt.to_period("M").astype(str)
    return data


def save_arrondissement_market(data: pd.DataFrame, output_dir: Path) -> Path:
    market = (
        data.groupby("code_postal")
        .agg(
            actual_median=("actual_price_per_m2", "median"),
            predicted_median=("predicted_price_per_m2", "median"),
            median_ape=("absolute_percentage_error", "median"),
            sales=("id_mutation", "size"),
        )
        .sort_index()
    )
    positions = np.arange(len(market))
    width = 0.38
    fig, ax = plt.subplots(figsize=(15, 8), constrained_layout=True)
    ax.bar(
        positions - width / 2,
        market["actual_median"],
        width,
        color=NAVY,
        label="Actual median",
    )
    ax.bar(
        positions + width / 2,
        market["predicted_median"],
        width,
        color=LIGHT_BLUE,
        label="Predicted median",
    )
    ax.set_xticks(positions, [postcode_label(code) for code in market.index])
    ax.set(
        xlabel="Arrondissement",
        ylabel="Median transaction value per m² (€)",
        title="Actual and Predicted Apartment Prices by Arrondissement",
    )
    ax.set_ylim(0, max(market["actual_median"].max(), market["predicted_median"].max()) * 1.18)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, ncol=2)
    for x, (_, row) in zip(positions, market.iterrows()):
        ax.text(
            x,
            max(row["actual_median"], row["predicted_median"]) + 130,
            f"{row['median_ape']:.1f}%\nn={int(row['sales'])}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=GRAY,
        )
    ax.text(
        0.01,
        0.98,
        "Labels show median absolute error and holdout sale count",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=GRAY,
    )
    path = output_dir / "arrondissement_market_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_bias_dashboard(data: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    fig.suptitle("Model Bias and Stability Deep Dive", fontsize=20, color=NAVY)

    # Bias by surface segment.
    ax = axes[0, 0]
    surface_labels = ["9–24", "25–39", "40–59", "60–79", "80–119", "120–300"]
    data = data.copy()
    data["surface_band"] = pd.cut(
        data["surface_reelle_bati"],
        bins=[9, 25, 40, 60, 80, 120, 301],
        labels=surface_labels,
        right=False,
    )
    surface = data.groupby("surface_band", observed=True).agg(
        median_bias=("signed_error_percent", "median"),
        median_ape=("absolute_percentage_error", "median"),
        sales=("id_mutation", "size"),
    )
    colors = [CORAL if value > 0 else BLUE for value in surface["median_bias"]]
    bars = ax.bar(surface.index.astype(str), surface["median_bias"], color=colors)
    ax.axhline(0, color=NAVY, linewidth=1)
    ax.set(
        xlabel="Surface band (m²)",
        ylabel="Median signed error (%)",
        title="Does size create systematic bias?",
    )
    ax.grid(axis="y", alpha=0.18)
    for bar, (_, row) in zip(bars, surface.iterrows()):
        offset = 0.25 if row["median_bias"] >= 0 else -0.25
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{row['median_bias']:+.1f}%\nn={int(row['sales'])}",
            ha="center",
            va="bottom" if row["median_bias"] >= 0 else "top",
            fontsize=8,
            color=GRAY,
        )

    # Calibration by predicted-price decile avoids defining groups from the target.
    ax = axes[0, 1]
    data["predicted_decile"] = pd.qcut(
        data["predicted_value"], q=10, labels=False, duplicates="drop"
    )
    deciles = data.groupby("predicted_decile").agg(
        predicted_median=("predicted_value", "median"),
        actual_median=("valeur_fonciere", "median"),
        sales=("id_mutation", "size"),
    )
    maximum = max(deciles["predicted_median"].max(), deciles["actual_median"].max()) / 1_000
    ax.plot([0, maximum], [0, maximum], color=GRAY, linestyle="--", label="Perfect calibration")
    scatter = ax.scatter(
        deciles["predicted_median"] / 1_000,
        deciles["actual_median"] / 1_000,
        c=np.arange(len(deciles)),
        cmap="viridis",
        s=75,
        zorder=3,
    )
    for decile, row in deciles.iterrows():
        ax.annotate(
            f"D{int(decile) + 1}",
            (row["predicted_median"] / 1_000, row["actual_median"] / 1_000),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set(
        xlim=(0, maximum),
        ylim=(0, maximum),
        xlabel="Median predicted price (€000)",
        ylabel="Median actual price (€000)",
        title="Calibration across predicted-price deciles",
    )
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)

    # Error spread by predicted price.
    ax = axes[1, 0]
    plot_sample = data.sample(n=min(4_000, len(data)), random_state=42)
    ax.scatter(
        plot_sample["predicted_value"] / 1_000,
        plot_sample["signed_error_percent"].clip(-80, 80),
        s=11,
        alpha=0.22,
        color=BLUE,
        edgecolors="none",
    )
    ax.axhline(0, color=CORAL, linewidth=2)
    ax.set(
        xlabel="Predicted price (€000)",
        ylabel="Signed error (%) · clipped to ±80%",
        title="Residual pattern and heteroskedasticity",
    )
    ax.grid(alpha=0.18)

    # Monthly market stability.
    ax = axes[1, 1]
    monthly = data.groupby("month").agg(
        actual_median=("actual_price_per_m2", "median"),
        predicted_median=("predicted_price_per_m2", "median"),
        sales=("id_mutation", "size"),
    )
    x = np.arange(len(monthly))
    ax.plot(x, monthly["actual_median"], marker="o", color=NAVY, linewidth=2.5, label="Actual €/m²")
    ax.plot(
        x,
        monthly["predicted_median"],
        marker="o",
        color=CORAL,
        linewidth=2.5,
        label="Predicted €/m²",
    )
    ax.set_xticks(x, [month[-2:] for month in monthly.index])
    ax.set(
        xlabel="Month in temporal holdout",
        ylabel="Median price per m² (€)",
        title="Does the model follow late-2025 market movement?",
    )
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    for position, count in zip(x, monthly["sales"]):
        ax.text(
            position,
            0.03,
            f"n={count}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            color=GRAY,
            fontsize=8,
        )

    path = output_dir / "bias_and_stability_dashboard.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def coverage_summary(group: pd.DataFrame) -> pd.Series:
    successes = int(group["covered_90"].sum())
    total = int(len(group))
    lower, upper = wilson_interval(successes, total)
    return pd.Series(
        {
            "coverage": successes / total,
            "lower": lower,
            "upper": upper,
            "sales": total,
        }
    )


def coverage_bars(ax: plt.Axes, table: pd.DataFrame, labels: list[str], title: str) -> None:
    positions = np.arange(len(table))
    coverage = table["coverage"].to_numpy() * 100
    lower_error = (table["coverage"] - table["lower"]).to_numpy() * 100
    upper_error = (table["upper"] - table["coverage"]).to_numpy() * 100
    colors = [GREEN if value >= 88 else CORAL for value in coverage]
    ax.bar(
        positions,
        coverage,
        color=colors,
        yerr=np.vstack([lower_error, upper_error]),
        capsize=3,
        alpha=0.9,
    )
    ax.axhline(90, color=NAVY, linestyle="--", linewidth=1.8, label="90% target")
    ax.set_xticks(positions, labels, rotation=0)
    ax.set(ylim=(65, 101), ylabel="Observed coverage (%)", title=title)
    ax.grid(axis="y", alpha=0.18)
    for x, (_, row) in zip(positions, table.iterrows()):
        ax.text(
            x,
            66,
            f"n={int(row['sales'])}",
            ha="center",
            va="bottom",
            fontsize=7,
            color=GRAY,
            rotation=90 if len(table) > 10 else 0,
        )


def save_coverage_dashboard(
    data: pd.DataFrame, report: dict, output_dir: Path
) -> Path:
    # Interval ratios come from the separate 2024 validation file, so every
    # row here belongs to the untouched 2025 coverage test.
    evaluation = data.copy()
    interval = report["prediction_interval"]
    evaluation["interval_lower"] = (
        evaluation["predicted_value"] * interval["lower_price_ratio"]
    )
    evaluation["interval_upper"] = (
        evaluation["predicted_value"] * interval["upper_price_ratio"]
    )
    evaluation["covered_90"] = evaluation["valeur_fonciere"].between(
        evaluation["interval_lower"], evaluation["interval_upper"]
    )
    evaluation["surface_band"] = pd.cut(
        evaluation["surface_reelle_bati"],
        bins=[9, 25, 40, 60, 80, 120, 301],
        labels=["9–24", "25–39", "40–59", "60–79", "80–119", "120–300"],
        right=False,
    )
    evaluation["predicted_price_band"] = pd.qcut(
        evaluation["predicted_value"],
        q=5,
        labels=["Lowest", "Low-mid", "Middle", "High-mid", "Highest"],
        duplicates="drop",
    )

    by_postcode = evaluation.groupby("code_postal").apply(
        coverage_summary, include_groups=False
    )
    by_surface = evaluation.groupby("surface_band", observed=True).apply(
        coverage_summary, include_groups=False
    )
    by_price = evaluation.groupby("predicted_price_band", observed=True).apply(
        coverage_summary, include_groups=False
    )

    fig, axes = plt.subplots(3, 1, figsize=(15, 15), constrained_layout=True)
    fig.suptitle(
        "Where Is the 90% Prediction Interval Reliable?",
        fontsize=20,
        color=NAVY,
    )
    coverage_bars(
        axes[0],
        by_postcode,
        [postcode_label(code) for code in by_postcode.index],
        "Coverage by arrondissement",
    )
    coverage_bars(
        axes[1],
        by_surface,
        [str(label) for label in by_surface.index],
        "Coverage by surface band (m²)",
    )
    coverage_bars(
        axes[2],
        by_price,
        [str(label) for label in by_price.index],
        "Coverage by predicted-price quintile",
    )
    axes[0].legend(frameon=False, loc="lower right")
    axes[2].set_xlabel("Segment")
    fig.text(
        0.5,
        0.01,
        "Error bars are 95% Wilson intervals · evaluated only on sales after the calibration period",
        ha="center",
        color=GRAY,
    )
    path = output_dir / "uncertainty_coverage_by_segment.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_error_matrix(data: pd.DataFrame, output_dir: Path) -> Path:
    matrix_data = data.copy()
    matrix_data["surface_band"] = pd.cut(
        matrix_data["surface_reelle_bati"],
        bins=[9, 25, 40, 60, 80, 120, 301],
        labels=["9–24", "25–39", "40–59", "60–79", "80–119", "120–300"],
        right=False,
    )
    matrix = matrix_data.pivot_table(
        index="surface_band",
        columns="code_postal",
        values="signed_error_percent",
        aggfunc="median",
        observed=True,
    )
    counts = matrix_data.pivot_table(
        index="surface_band",
        columns="code_postal",
        values="id_mutation",
        aggfunc="size",
        observed=True,
    )
    fig, ax = plt.subplots(figsize=(16, 7), constrained_layout=True)
    values = matrix.to_numpy()
    count_values = counts.to_numpy()
    display_values = values.copy()
    display_values[count_values < 20] = np.nan
    colormap = plt.get_cmap("RdBu_r").copy()
    colormap.set_bad("#E5E7EB")
    image = ax.imshow(
        display_values,
        cmap=colormap,
        norm=TwoSlopeNorm(vmin=-15, vcenter=0, vmax=15),
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(matrix.columns)), [postcode_label(code) for code in matrix.columns])
    ax.set_yticks(np.arange(len(matrix.index)), [str(label) for label in matrix.index])
    ax.set(
        xlabel="Arrondissement",
        ylabel="Surface band (m²)",
        title="Median Signed Error by Apartment Segment",
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            count = count_values[row, column]
            if np.isfinite(value):
                if count < 20:
                    ax.text(
                        column,
                        row,
                        f"low n\nn={int(count)}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=GRAY,
                    )
                    continue
                color = "white" if abs(value) >= 8 else NAVY
                ax.text(
                    column,
                    row,
                    f"{value:+.1f}%\nn={int(count)}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color,
                )
    colorbar = fig.colorbar(image, ax=ax, shrink=0.85, pad=0.02)
    colorbar.set_label("Median signed error (%) · blue = under · red = over")
    ax.text(
        0.01,
        -0.14,
        "Gray cells are suppressed because they contain fewer than 20 holdout sales.",
        transform=ax.transAxes,
        color=GRAY,
    )
    path = output_dir / "segment_error_heatmap.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_price_map(data: pd.DataFrame, output_dir: Path) -> Path:
    lower, upper = np.quantile(data["actual_price_per_m2"], [0.02, 0.98])
    fig, ax = plt.subplots(figsize=(11, 10), constrained_layout=True)
    scatter = ax.scatter(
        data["longitude"],
        data["latitude"],
        c=data["actual_price_per_m2"].clip(lower, upper),
        cmap="viridis",
        s=15,
        alpha=0.66,
        edgecolors="none",
    )
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.78, pad=0.02)
    colorbar.set_label("Actual transaction value per m² (€) · colors clipped at 2nd/98th percentiles")
    ax.set(
        xlabel="Longitude",
        ylabel="Latitude",
        title="Geography of Apartment Prices in the Unseen Holdout",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.12)
    path = output_dir / "holdout_price_per_m2_map.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", type=Path, default=PROJECT_ROOT / "reports/test_predictions.csv"
    )
    parser.add_argument("--metrics", type=Path, default=PROJECT_ROOT / "reports/metrics.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "visuals")
    args = parser.parse_args()

    data = load_data(args.predictions)
    report = json.loads(args.metrics.read_text(encoding="utf-8"))
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

    paths = [
        save_arrondissement_market(data, args.output_dir),
        save_bias_dashboard(data, args.output_dir),
        save_coverage_dashboard(data, report, args.output_dir),
        save_error_matrix(data, args.output_dir),
        save_price_map(data, args.output_dir),
    ]
    for path in paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
