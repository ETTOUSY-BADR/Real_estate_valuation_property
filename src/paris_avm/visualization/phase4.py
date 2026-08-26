"""Generate the detailed Phase 4 nationwide report and publication figures.

The module never estimates missing metrics.  With ``--allow-incomplete`` it
publishes the methodology, architecture, artifact inventory, and execution
status before the large national run.  Once the Gold table and benchmark
outputs exist, the same command adds empirical coverage, performance,
diagnostic, regional, feature-importance, and comparable-density figures.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np
import pandas as pd

from paris_avm.artifacts import write_atomic
from paris_avm.features.phase4 import (
    PHASE4_CATEGORICAL_FEATURES,
    PHASE4_MODEL_FEATURES,
    PHASE4_NUMERIC_FEATURES,
)
from paris_avm.geography import KNOWN_DVF_GAPS, REGION_NAMES
from paris_avm.paths import PROJECT_ROOT


NAVY = "#102A43"
BLUE = "#1677B8"
CYAN = "#38A3A5"
GREEN = "#2A9D6F"
GOLD = "#E9A23B"
RED = "#D1495B"
PURPLE = "#7158A6"
INK = "#243B53"
MUTED = "#627D98"
GRID = "#D9E2EC"
LIGHT = "#F4F7FA"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK,
            "axes.edgecolor": GRID,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 220,
        }
    )


def finish(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")
    return path


def clean_axis(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.75)
        ax.set_axisbelow(True)


def euro(value: float, _: int | None = None) -> str:
    return f"EUR {value / 1_000:,.0f}k"


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def draw_architecture(output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(14.2, 8.0))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    fig.suptitle(
        "Phase 4 architecture: one leakage-aware model across nationwide Geo-DVF",
        fontsize=18,
        fontweight="bold",
        color=NAVY,
        y=0.975,
    )
    ax.text(
        0.35,
        7.25,
        "Chunked national ingestion · global spatial representation · 90-day information lag · frozen 2025 test",
        color=MUTED,
        fontsize=11,
    )

    def box(
        x: float, y: float, width: float, height: float, title: str, body: str, color: str
    ) -> None:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.025,rounding_size=0.12",
            linewidth=1.5,
            edgecolor=color,
            facecolor="white",
        )
        patch.set_path_effects(
            [patheffects.SimplePatchShadow(offset=(1.5, -1.5), alpha=0.12), patheffects.Normal()]
        )
        ax.add_patch(patch)
        ax.add_patch(
            FancyBboxPatch(
                (x, y + height - 0.38),
                width,
                0.38,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                facecolor=color,
                edgecolor=color,
            )
        )
        ax.text(x + 0.16, y + height - 0.2, title, color="white", fontweight="bold", va="center")
        ax.text(x + 0.16, y + height - 0.62, body, fontsize=8.8, va="top", linespacing=1.35)

    box(0.35, 5.15, 2.65, 1.5, "SOURCE", "Geo-DVF full France\n2021 · 2022 · 2023\n2024 · 2025", BLUE)
    box(3.75, 5.15, 2.65, 1.5, "CLEANING", "One apartment per sale\nQuality and value bounds\nChunk-boundary safety", CYAN)
    box(7.15, 5.15, 2.65, 1.5, "GEOGRAPHY", "Region · department\ncommune · postcode · H3\nglobal Cartesian position", PURPLE)
    box(10.55, 5.15, 2.95, 1.5, "COMPARABLES", "250–1,000 m radii\n180–730 day windows\nminimum age: 90 days", GREEN)
    box(1.3, 2.55, 3.2, 1.55, "MODEL SELECTION", "2021–2023 development\n2024 CatBoost selection\n2024 interval calibration", GOLD)
    box(5.4, 2.55, 3.2, 1.55, "FROZEN EVALUATION", "Refit through 2024\nUntouched 2025 test\nregional + department audit", RED)
    box(9.5, 2.55, 3.2, 1.55, "PRODUCTION", "Refit on 2021–2025\nregional empirical interval\nartifact checksum contract", NAVY)
    box(3.35, 0.45, 3.25, 1.25, "REPORTING", "Six empirical dashboards\nterritorial performance tables\nartifact inventory", BLUE)
    box(7.4, 0.45, 3.25, 1.25, "INTERPRETATION", "Automated indication only\nknown DVF territorial gaps\nno causal feature claims", MUTED)

    arrows = [
        ((3.02, 5.9), (3.72, 5.9)),
        ((6.42, 5.9), (7.12, 5.9)),
        ((9.82, 5.9), (10.52, 5.9)),
        ((12.0, 5.12), (3.0, 4.13)),
        ((4.52, 3.32), (5.37, 3.32)),
        ((8.62, 3.32), (9.47, 3.32)),
        ((11.1, 2.52), (9.0, 1.72)),
        ((5.35, 1.72), (5.0, 1.7)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=15,
                linewidth=1.7,
                color=MUTED,
            )
        )
    return finish(fig, output_dir, "phase4_national_pipeline.png")


def draw_coverage(gold: pd.DataFrame, output_dir: Path) -> Path:
    data = gold.copy()
    data["date_mutation"] = pd.to_datetime(data["date_mutation"])
    if "year" not in data:
        data["year"] = data["date_mutation"].dt.year
    region_year = data.groupby(["code_region", "year"], observed=True).size().unstack(fill_value=0)
    region_year.index = [REGION_NAMES.get(str(code), str(code)) for code in region_year.index]
    region_year = region_year.loc[region_year.sum(axis=1).sort_values().index]

    fig = plt.figure(figsize=(14.2, 9.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], width_ratios=[1.18, 1], hspace=0.34, wspace=0.28)
    fig.suptitle("Phase 4 national data coverage", fontsize=18, fontweight="bold", color=NAVY, y=0.985)

    ax = fig.add_subplot(gs[:, 0])
    sample = data.sample(min(150_000, len(data)), random_state=42)
    ax.scatter(sample["longitude"], sample["latitude"], s=1.2, alpha=0.18, color=BLUE, linewidths=0)
    ax.set_xlim(-6, 11)
    ax.set_ylim(41, 52)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Metropolitan transaction footprint (sample)", loc="left")
    ax.text(
        0.02,
        0.02,
        "Overseas departments are included in the model\nbut omitted from this metropolitan viewport.",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": GRID},
    )
    clean_axis(ax, None)

    ax = fig.add_subplot(gs[0, 1])
    left = np.zeros(len(region_year))
    palette = [BLUE, CYAN, GREEN, GOLD, PURPLE]
    for position, year in enumerate(region_year.columns):
        values = region_year[year].to_numpy()
        ax.barh(region_year.index, values, left=left, color=palette[position % len(palette)], label=str(year))
        left += values
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1_000:,.0f}k"))
    ax.set_title("Retained sales by region and year", loc="left")
    ax.set_xlabel("Clean apartment sales")
    ax.legend(frameon=False, ncol=min(5, len(region_year.columns)), fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    clean_axis(ax, "x")

    ax = fig.add_subplot(gs[1, 1])
    annual = data.groupby("year").size()
    bars = ax.bar(annual.index.astype(str), annual.values, color=[BLUE, CYAN, GREEN, GOLD, PURPLE][: len(annual)])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1_000:,.0f}k"))
    ax.set_ylabel("Retained sales")
    ax.set_title("Chronological evaluation cohorts", loc="left")
    for bar, value in zip(bars, annual.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}", ha="center", va="bottom", fontsize=8)
    clean_axis(ax)
    return finish(fig, output_dir, "phase4_national_coverage.png")


def draw_performance(
    benchmark: dict[str, Any], region_performance: pd.DataFrame, output_dir: Path
) -> Path:
    model = benchmark["test_metrics"]
    baseline = benchmark["baseline_test_metrics"]
    metrics = ["mae_eur", "rmse_eur", "median_absolute_percentage_error", "within_20_percent"]
    labels = ["MAE", "RMSE", "Median APE", "Within ±20%"]

    fig = plt.figure(figsize=(14.0, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.25], hspace=0.38, wspace=0.3)
    fig.suptitle("Phase 4 frozen-test performance", fontsize=18, fontweight="bold", color=NAVY, y=0.985)
    fig.text(0.5, 0.95, f"Untouched {benchmark['protocol']['untouched_test']} national apartment cohort", ha="center", color=MUTED)

    cards = [
        ("National MAE", f"EUR {model['mae_eur']:,.0f}", GREEN),
        ("Baseline MAE", f"EUR {baseline['mae_eur']:,.0f}", MUTED),
        ("MAE reduction", f"EUR {benchmark['mae_reduction_vs_baseline_eur']:,.0f}", BLUE),
        ("90% interval coverage", percent(benchmark["prediction_interval"]["observed_test_coverage"]), PURPLE),
    ]
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 2.4)
    ax.axis("off")
    for index, (title, value, color) in enumerate(cards):
        x = 0.25 + index * 3.45
        patch = FancyBboxPatch((x, 0.35), 3.05, 1.55, boxstyle="round,pad=0.04,rounding_size=0.12", facecolor=LIGHT, edgecolor=color, linewidth=1.5)
        ax.add_patch(patch)
        ax.text(x + 0.18, 1.58, title, color=MUTED, fontsize=9.5, fontweight="bold")
        ax.text(x + 0.18, 0.82, value, color=color, fontsize=18, fontweight="bold")

    ax = fig.add_subplot(gs[1, 0])
    comparison = pd.DataFrame(
        {
            "metric": labels,
            "model": [model[name] for name in metrics],
            "baseline": [baseline[name] for name in metrics],
        }
    )
    # Scale the heterogeneous metrics relative to the baseline for a compact operational view.
    comparison["model_index"] = comparison["model"] / comparison["baseline"] * 100
    y = np.arange(len(comparison))
    ax.barh(y + 0.18, np.full(len(y), 100), height=0.34, color=MUTED, label="Geographic median baseline")
    ax.barh(y - 0.18, comparison["model_index"], height=0.34, color=GREEN, label="Phase 4 CatBoost")
    ax.axvline(100, color=NAVY, linewidth=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Index (baseline = 100; direction differs by metric)")
    ax.set_title("Model versus hierarchical baseline", loc="left")
    ax.legend(frameon=False, fontsize=8)
    clean_axis(ax, "x")

    ax = fig.add_subplot(gs[1, 1])
    regions = region_performance.sort_values("mae_eur").copy()
    names = regions.get("region_name", regions["code_region"].map(lambda value: REGION_NAMES.get(str(value), value)))
    colors = np.where(regions["mae_eur"] <= model["mae_eur"], GREEN, GOLD)
    ax.barh(names, regions["mae_eur"], color=colors)
    ax.axvline(model["mae_eur"], color=RED, linestyle="--", label="National MAE")
    ax.xaxis.set_major_formatter(FuncFormatter(euro))
    ax.set_xlabel("Regional MAE")
    ax.set_title("Territorial error heterogeneity", loc="left")
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(frameon=False)
    clean_axis(ax, "x")
    return finish(fig, output_dir, "phase4_performance_dashboard.png")


def draw_diagnostics(predictions: pd.DataFrame, output_dir: Path) -> Path:
    data = predictions.copy()
    actual = data["valeur_fonciere"].to_numpy(float)
    predicted = data["predicted_value"].to_numpy(float)
    error = predicted - actual
    absolute_percentage = np.abs(error) / actual * 100

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.4))
    fig.suptitle("Phase 4 out-of-time prediction diagnostics", fontsize=18, fontweight="bold", color=NAVY, y=0.99)
    ax = axes[0, 0]
    hb = ax.hexbin(actual, predicted, gridsize=60, xscale="log", yscale="log", bins="log", mincnt=1, cmap="viridis")
    limits = [max(1_000, np.nanpercentile(actual, 0.2)), np.nanpercentile(actual, 99.8)]
    ax.plot(limits, limits, color=RED, linestyle="--", linewidth=1.5)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Observed value (EUR, log)")
    ax.set_ylabel("Predicted value (EUR, log)")
    ax.set_title("Observed versus predicted", loc="left")
    fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.03, label="log density")
    clean_axis(ax, None)

    ax = axes[0, 1]
    decile = pd.qcut(actual, 10, labels=False, duplicates="drop")
    calibration = pd.DataFrame({"actual": actual, "predicted": predicted, "decile": decile}).groupby("decile").median()
    ax.plot(calibration["actual"], calibration["predicted"], marker="o", color=BLUE, linewidth=2.2)
    extrema = [calibration[["actual", "predicted"]].min().min(), calibration[["actual", "predicted"]].max().max()]
    ax.plot(extrema, extrema, color=RED, linestyle="--")
    ax.xaxis.set_major_formatter(FuncFormatter(euro))
    ax.yaxis.set_major_formatter(FuncFormatter(euro))
    ax.set_xlabel("Median observed value")
    ax.set_ylabel("Median predicted value")
    ax.set_title("Calibration by observed-value decile", loc="left")
    clean_axis(ax)

    ax = axes[1, 0]
    clipped = np.clip(error, np.nanpercentile(error, 1), np.nanpercentile(error, 99))
    ax.hist(clipped, bins=70, color=BLUE, alpha=0.86, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color=RED, linewidth=1.4)
    ax.axvline(np.median(error), color=GREEN, linewidth=2, label=f"Median bias: EUR {np.median(error):,.0f}")
    ax.xaxis.set_major_formatter(FuncFormatter(euro))
    ax.set_xlabel("Prediction error (predicted − observed; 1–99% clipped)")
    ax.set_ylabel("Transactions")
    ax.set_title("Signed residual distribution", loc="left")
    ax.legend(frameon=False)
    clean_axis(ax)

    ax = axes[1, 1]
    bands = pd.cut(actual, bins=[0, 100_000, 200_000, 350_000, 500_000, 750_000, 1_000_000, 2_000_000, np.inf])
    by_value = pd.DataFrame({"band": bands, "ape": absolute_percentage}).groupby("band", observed=True)["ape"].median()
    bars = ax.bar(range(len(by_value)), by_value, color=np.where(by_value <= 20, GREEN, GOLD))
    ax.set_xticks(range(len(by_value)), [str(value).replace(", ", "–") for value in by_value.index], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Median absolute percentage error")
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_title("Relative error across value bands", loc="left")
    for bar, value in zip(bars, by_value):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)
    clean_axis(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return finish(fig, output_dir, "phase4_prediction_diagnostics.png")


def draw_regional_performance(region: pd.DataFrame, department: pd.DataFrame, output_dir: Path) -> Path:
    regions = region.sort_values("mae_eur").copy()
    regions["name"] = regions.get("region_name", regions["code_region"].map(lambda value: REGION_NAMES.get(str(value), value)))
    departments = department.sort_values("mae_eur")
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 8.3), gridspec_kw={"width_ratios": [1, 1.05], "wspace": 0.38})
    fig.suptitle("Territorial robustness audit", fontsize=18, fontweight="bold", color=NAVY, y=0.985)

    ax = axes[0]
    size = np.clip(np.sqrt(regions["rows"]) * 1.6, 30, 450)
    scatter = ax.scatter(regions["rows"], regions["mae_eur"], s=size, c=regions["median_absolute_percentage_error"], cmap="viridis", alpha=0.8, edgecolors="white")
    for _, row in regions.iterrows():
        ax.annotate(str(row["name"]), (row["rows"], row["mae_eur"]), xytext=(4, 3), textcoords="offset points", fontsize=7.5)
    ax.set_xscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(euro))
    ax.set_xlabel("2025 regional test rows (log scale)")
    ax.set_ylabel("MAE")
    ax.set_title("Error versus regional evidence", loc="left")
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03, label="Median APE (%)")
    clean_axis(ax)

    ax = axes[1]
    worst = pd.concat([departments.head(10), departments.tail(10)]).drop_duplicates().sort_values("mae_eur")
    colors = [GREEN] * min(10, len(worst)) + [GOLD] * max(0, len(worst) - 10)
    ax.barh(worst["code_departement"].astype(str), worst["mae_eur"], color=colors)
    ax.xaxis.set_major_formatter(FuncFormatter(euro))
    ax.set_xlabel("Department MAE")
    ax.set_title("Best and worst sufficiently sized departments", loc="left")
    clean_axis(ax, "x")
    fig.text(0.5, 0.015, "Only segments with at least 100 frozen-test transactions are reported; differences combine market mix and model error.", ha="center", color=MUTED, fontsize=9)
    return finish(fig, output_dir, "phase4_regional_performance.png")


def feature_family(feature: str) -> str:
    if feature in PHASE4_CATEGORICAL_FEATURES or feature.startswith("h3_"):
        return "Territorial identity"
    if feature.startswith(("local_", "weighted_", "similar_", "effective_", "nearest_", "median_comparable")):
        return "Historical comparables"
    if feature in {"latitude", "longitude", "geo_x", "geo_y", "geo_z"}:
        return "Global location"
    if feature in {"month", "day_of_year", "year", "months_since_2021"}:
        return "Time"
    return "Property"


def draw_feature_importance(importance: pd.DataFrame, output_dir: Path) -> Path:
    data = importance.copy()
    data["family"] = data["feature"].map(feature_family)
    family = data.groupby("family", as_index=False)["importance"].sum().sort_values("importance")
    top = data.nlargest(20, "importance").sort_values("importance")
    palette = {
        "Territorial identity": PURPLE,
        "Historical comparables": BLUE,
        "Global location": GREEN,
        "Time": GOLD,
        "Property": MUTED,
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 8.3), gridspec_kw={"width_ratios": [0.75, 1.5], "wspace": 0.48})
    fig.suptitle("Phase 4 model information usage", fontsize=18, fontweight="bold", color=NAVY, y=0.985)
    ax = axes[0]
    ax.barh(family["family"], family["importance"], color=[palette[name] for name in family["family"]])
    ax.set_xlabel("Summed CatBoost importance")
    ax.set_title("Feature-family contribution", loc="left")
    clean_axis(ax, "x")
    ax = axes[1]
    labels = [value.replace("_", " ").capitalize() for value in top["feature"]]
    ax.barh(labels, top["importance"], color=[palette[name] for name in top["family"]])
    ax.tick_params(axis="y", labelsize=8.4)
    ax.set_xlabel("CatBoost prediction-value-change importance")
    ax.set_title("Top individual predictors", loc="left")
    clean_axis(ax, "x")
    fig.text(0.5, 0.018, "Importance measures predictive usage, not causal value or willingness to pay.", ha="center", color=MUTED, fontsize=9.5)
    return finish(fig, output_dir, "phase4_feature_importance.png")


def draw_comparable_coverage(gold: pd.DataFrame, output_dir: Path) -> Path:
    data = gold.copy()
    data["region_name"] = data["code_region"].map(lambda value: REGION_NAMES.get(str(value), str(value)))
    region = data.groupby("region_name", observed=True).agg(
        rows=("id_mutation", "size"),
        weighted_coverage=("weighted_comp_ppm2", lambda values: values.notna().mean()),
        similar_coverage=("similar_weighted_comp_ppm2", lambda values: values.notna().mean()),
        median_count=("local_365d_500m_count", "median"),
    ).sort_values("weighted_coverage")
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 8.1), gridspec_kw={"wspace": 0.42})
    fig.suptitle("Historical comparable evidence across France", fontsize=18, fontweight="bold", color=NAVY, y=0.985)
    ax = axes[0]
    y = np.arange(len(region))
    ax.barh(y + 0.17, region["weighted_coverage"] * 100, height=0.32, color=BLUE, label="Any weighted comparable")
    ax.barh(y - 0.17, region["similar_coverage"] * 100, height=0.32, color=GREEN, label="Any similar comparable")
    ax.set_yticks(y, region.index)
    ax.tick_params(axis="y", labelsize=8)
    ax.xaxis.set_major_formatter(PercentFormatter())
    ax.set_xlabel("Coverage")
    ax.set_title("Availability by region", loc="left")
    ax.legend(frameon=False, fontsize=8)
    clean_axis(ax, "x")
    ax = axes[1]
    ax.scatter(region["median_count"], region["weighted_coverage"] * 100, s=np.clip(np.sqrt(region["rows"]) * 1.3, 30, 400), color=PURPLE, alpha=0.75, edgecolors="white")
    for name, row in region.iterrows():
        ax.annotate(name, (row["median_count"], row["weighted_coverage"] * 100), xytext=(4, 3), textcoords="offset points", fontsize=7.5)
    ax.set_xlabel("Median 365-day comparable count within 500 m")
    ax.set_ylabel("Weighted-comparable coverage")
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_title("Density versus coverage", loc="left")
    clean_axis(ax)
    return finish(fig, output_dir, "phase4_comparable_coverage.png")


def markdown_table(frame: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    if frame.empty:
        return "_No rows available._"
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|"]
    for _, row in frame.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if column.endswith("_eur"):
                values.append(f"EUR {float(value):,.0f}")
            elif column.startswith("within_"):
                values.append(f"{float(value):.2%}")
            elif "percent" in column:
                values.append(f"{float(value):.2f}%")
            elif column == "r2":
                values.append(f"{float(value):.4f}")
            elif column == "rows":
                values.append(f"{int(value):,}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def artifact_inventory(paths: dict[str, Path]) -> tuple[str, list[str]]:
    lines = ["| Artifact | Status | Size | SHA-256 |", "|---|---|---:|---|"]
    missing: list[str] = []
    for name, path in paths.items():
        if path.exists():
            lines.append(f"| `{name}` | available | {path.stat().st_size:,} bytes | `{file_sha256(path)[:16]}…` |")
        else:
            lines.append(f"| `{name}` | **not generated** | — | — |")
            missing.append(name)
    return "\n".join(lines), missing


def render_report(
    report_path: Path,
    output_dir: Path,
    paths: dict[str, Path],
    generated_figures: list[Path],
    benchmark: dict[str, Any] | None,
    quality: dict[str, Any] | None,
    region: pd.DataFrame | None,
    comparison: pd.DataFrame | None,
    validation_trials: pd.DataFrame | None,
) -> None:
    inventory, missing = artifact_inventory(paths)
    ready = not missing
    figure_links = "\n".join(
        f"![{path.stem.replace('_', ' ').title()}](../../visuals/phase4/{path.name})"
        for path in generated_figures
    )
    if benchmark:
        model = benchmark["test_metrics"]
        baseline = benchmark["baseline_test_metrics"]
        outcome = f"""The frozen 2025 test contains **{benchmark['rows']['test']:,}** apartment sales. The selected
CatBoost model reaches **EUR {model['mae_eur']:,.0f} MAE**, **{model['median_absolute_percentage_error']:.2f}%
median absolute percentage error**, **{model['within_20_percent']:.2%} within ±20%**, and **R² =
{model['r2']:.4f}**. Against the training-only hierarchical geographic median baseline (EUR
{baseline['mae_eur']:,.0f} MAE), the reduction is **EUR
{benchmark['mae_reduction_vs_baseline_eur']:,.0f}**. The empirical 90% interval covers
**{benchmark['prediction_interval']['observed_test_coverage']:.2%}** of test transactions."""
        status = "**Execution status: complete empirical report.**"
    else:
        outcome = """The national acquisition, feature build, and benchmark have not yet been executed in this
workspace. Consequently, this report contains **no claimed Phase 4 accuracy figures**. It documents the
complete protocol and renders the architecture now; empirical sections and six additional dashboards are
populated automatically by rerunning the reporting command after the national pipeline completes."""
        status = "**Execution status: methodology report; empirical results pending.**"

    if quality:
        coverage_text = (
            f"The Gold table contains **{quality['row_count']:,}** retained sales from **{quality['date_min']}** "
            f"through **{quality['date_max']}**, spanning **{len(quality['observed_departments'])} departments** "
            f"and **{len(quality['observed_regions'])} regions** represented upstream."
        )
    else:
        coverage_text = "Territorial row counts and comparable coverage will be inserted from `feature_quality.json`."

    if region is not None and not region.empty:
        table_frame = region.copy()
        if "region_name" not in table_frame:
            table_frame["region_name"] = table_frame["code_region"].map(lambda value: REGION_NAMES.get(str(value), value))
        regional_table = markdown_table(
            table_frame.sort_values("mae_eur"),
            ["region_name", "rows", "mae_eur", "median_absolute_percentage_error", "within_20_percent", "r2"],
            ["Region", "Rows", "MAE", "MdAPE", "Within ±20%", "R²"],
        )
    else:
        regional_table = "_Pending `region_performance.csv` from the frozen test._"

    if comparison is not None and not comparison.empty:
        benchmark_table = markdown_table(
            comparison,
            ["method", "mae_eur", "rmse_eur", "median_absolute_percentage_error", "within_20_percent", "r2"],
            ["Method", "MAE", "RMSE", "MdAPE", "Within ±20%", "R²"],
        )
    else:
        benchmark_table = "_Pending `benchmark_comparison.csv` from the frozen test._"
    if validation_trials is not None and not validation_trials.empty:
        validation_table = markdown_table(
            validation_trials.sort_values("mae_eur"),
            ["candidate", "configuration", "mae_eur", "median_absolute_percentage_error", "within_20_percent", "r2"],
            ["Candidate", "Configuration", "2024 MAE", "2024 MdAPE", "Within ±20%", "R²"],
        )
    else:
        validation_table = "_Pending `validation_trials.csv` from model selection._"

    feature_groups = pd.DataFrame(
        [
            ("Property", 7, "Surface, rooms, address number, lots, terrain, log surface, room density"),
            ("Calendar", 4, "Month, day of year, year, months since January 2021"),
            ("Global location", 5, "Latitude, longitude and unit-sphere Cartesian x/y/z"),
            ("Territorial identity", len(PHASE4_CATEGORICAL_FEATURES), "Region, department, commune, postcode and H3 r7/r8/r9"),
            ("Historical comparables", len(PHASE4_MODEL_FEATURES) - 16 - len(PHASE4_CATEGORICAL_FEATURES), "Lagged counts, medians, dispersion, similarity, distance, age and momentum"),
        ],
        columns=["group", "features", "interpretation"],
    )
    feature_table = markdown_table(feature_groups, ["group", "features", "interpretation"], ["Family", "Count", "Purpose"])
    figures_section = figure_links or "_No figures generated._"
    missing_text = ", ".join(f"`{name}`" for name in missing) if missing else "none"

    report = f"""# Phase 4 — Nationwide France apartment valuation

Generated at {datetime.now(timezone.utc).isoformat()}.

{status}

## Executive summary

{outcome}

Phase 4 transfers the chronological, leakage-aware research protocol from Paris to a single national
CatBoost model. It does **not** transfer Paris-only transport, school, park, noise, BAN/BDNB, or DPE
fields where a nationally consistent snapshot has not been established. This choice makes the national
comparison narrower but auditable.

## Scope and territorial coverage

{coverage_text}

Geo-DVF does not publish open observations for Moselle (57), Bas-Rhin (67), Haut-Rhin (68), or Mayotte
(976). The phrase “nationwide” therefore means every department represented by the upstream Geo-DVF
source, not universal coverage of all French territory. Corsica and represented overseas departments are
handled explicitly through alphanumeric/three-digit department codes and spherical geometry.

## Research questions

1. Can one model learn heterogeneous apartment markets across French regions without collapsing local
   price structure into a national average?
2. Do region, department, commune, postcode, H3 and continuous global-location features provide enough
   spatial hierarchy for previously unseen annual transactions?
3. Do strictly lagged local comparable sales improve on a training-only geographic median baseline?
4. How uneven are error, calibration and comparable coverage across regions and departments?
5. Does the empirical uncertainty interval retain useful coverage under the 2024→2025 time shift?

## Data source and cohort construction

The input is the normalized, geolocated Geo-DVF annual `full.csv.gz` archive for each year from 2021
through 2025. Files are streamed in bounded chunks. A carry-over buffer prevents a mutation split at a
chunk boundary from being classified before all its property rows are available.

The cohort keeps `Vente` mutations containing exactly one apartment (`code_type_local = 2`) and no house
or commercial property. Dependencies such as parking or cellars may remain bundled because DVF reports
mutation-level consideration. Required values are transaction date, value, surface, commune and valid
WGS84 coordinates. Fixed admissibility bounds are 9–300 m², EUR 10,000–10,000,000, and EUR
100–30,000/m². These broad national bounds remove obvious corruption while retaining lower-price markets.

## Feature contract

{feature_table}

The current transaction's `valeur_fonciere` and derived `price_per_m2` are explicitly excluded from model
inputs. `price_per_m2` may only contribute through older eligible transactions when comparable features
are computed.

### Global geography

Lambert-93 is not valid for all overseas departments. Phase 4 therefore represents position using WGS84,
H3 cells and unit-sphere Cartesian coordinates. Nearby-sale distances use BallTree haversine distance and
are processed independently per department. This prevents mainland-only projection assumptions and avoids
cross-border neighbor contamination.

### Historical comparable sales

For every target sale, a source transaction must be at least 90 days older and no more than 730 days old.
The feature builder evaluates 250 m, 500 m and 1,000 m radii over 180, 365 and 730-day windows. It records
counts, median EUR/m², local median absolute deviation, exponentially weighted EUR/m², effective sample
size, nearest distance, median age, similar-apartment evidence and short-versus-long momentum. No current
or future transaction can enter its own features.

## Model and chronological protocol

- **Development:** 2021–2023.
- **Selection/calibration:** two predeclared CatBoost configurations evaluated on 2024.
- **Frozen test:** the selected configuration is refitted through 2024 and evaluated once on 2025.
- **Production artifact:** after reporting test performance, the chosen model is refitted on 2021–2025.
- **Target:** `log1p(valeur_fonciere)`, transformed back to non-negative euros.
- **Baseline:** postcode median EUR/m² from prior training years, falling back to commune, department,
  region and finally national median.

CatBoost is retained from Phase 3 because it handles nonlinear numeric effects, missing values and nested
high-cardinality territorial categories without one-hot expansion. Configuration selection uses 2024 only;
regional and department tables are diagnostics, not additional tuning targets.

### 2024 candidate benchmark

{validation_table}

### Frozen 2025 model-versus-baseline benchmark

{benchmark_table}

The benchmark is paired: both methods predict the identical test transactions. Phase 4 therefore reports
the transaction-level difference between baseline and model absolute error, together with a deterministic
paired bootstrap 95% interval. Bootstrap computation is capped at 250,000 test rows to bound memory; the
sample size and repetition count are recorded in `benchmark.json`.

## Evaluation framework

The national report records MAE, RMSE, R², MAPE, median absolute percentage error and hit rates within
±10% and ±20%. The same metrics are calculated for every region and for departments with at least 100
test observations. A 90% empirical interval is calibrated from 2024 target/prediction ratios, with regional
ratios used when at least 500 calibration observations exist and the national ratio used otherwise.

## Regional frozen-test results

{regional_table}

Regional comparisons must not be read as rankings of model quality alone. Sale-value mix, apartment
surface, urban density, sample size, geocoding completeness and comparable availability all alter raw MAE.
MdAPE and hit rates provide necessary scale-adjusted context.

## Visual diagnostics

{figures_section}

The coverage map is a sampled point cloud rather than a political boundary map. It demonstrates the
transaction footprint without introducing an unversioned shapefile. Feature importance measures how often
and how strongly CatBoost uses a predictor; it is not a causal estimate of an amenity or location premium.

## Reproducibility and artifact integrity

{inventory}

Missing empirical artifacts in this rendering: {missing_text}.

Every downloaded archive, Gold table and benchmark input is hashed with SHA-256. Gold Parquet, JSON, CSV,
model and prediction files are written through sibling `.part` files and promoted only after successful
serialization. The production bundle records the exact feature-table checksum and feature lists.

## Limitations and responsible use

- DVF omits property condition, renovation quality, apartment floor, elevator, view, occupancy and many
  transaction-specific rights or bundled movable elements.
- Departments 57, 67, 68 and 976 are absent upstream; the model must reject unsupported inference there.
- Sparse rural and overseas markets can have few nearby comparable sales and wider error dispersion.
- H3 and administrative identifiers encode location but do not explain causal neighborhood effects.
- The 2025 holdout measures one annual temporal transfer. It does not guarantee 2026+ performance.
- Empirical prediction intervals are marginal coverage diagnostics, not certified conditional guarantees.
- Output is an automated statistical indication, not a notarial valuation, certified appraisal, lending
  decision, tax assessment or investment recommendation.

## Reproduction

```powershell
$env:PYTHONPATH = '.\\src'
python -m paris_avm.data.acquire_phase4
python -m paris_avm.features.phase4
python -m paris_avm.modeling.benchmark_phase4
python -m paris_avm.visualization.phase4
```

The complete workflow is also available through:

```powershell
powershell -ExecutionPolicy Bypass -File .\\scripts\\run_phase4.ps1
```

For a methodology-only rendering before the large empirical run:

```powershell
python -m paris_avm.visualization.phase4 --allow-incomplete
```
"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(report_path, lambda temporary: temporary.write_text(report, encoding="utf-8"))
    print(f"Saved {report_path}")


def build_parser() -> argparse.ArgumentParser:
    root = PROJECT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=root / "data/gold/phase4_france_sale_features.parquet")
    parser.add_argument("--quality", type=Path, default=root / "reports/phase4/feature_quality.json")
    parser.add_argument("--benchmark", type=Path, default=root / "reports/phase4/benchmark.json")
    parser.add_argument("--predictions", type=Path, default=root / "reports/phase4/test_predictions.parquet")
    parser.add_argument("--region-performance", type=Path, default=root / "reports/phase4/region_performance.csv")
    parser.add_argument("--department-performance", type=Path, default=root / "reports/phase4/department_performance.csv")
    parser.add_argument("--feature-importance", type=Path, default=root / "reports/phase4/feature_importance.csv")
    parser.add_argument("--benchmark-comparison", type=Path, default=root / "reports/phase4/benchmark_comparison.csv")
    parser.add_argument("--validation-trials", type=Path, default=root / "reports/phase4/validation_trials.csv")
    parser.add_argument("--source-manifest", type=Path, default=root / "data/raw/dvf_france/manifest.json")
    parser.add_argument("--model", type=Path, default=root / "models/phase4/france_catboost.joblib")
    parser.add_argument("--output-dir", type=Path, default=root / "visuals/phase4")
    parser.add_argument("--report", type=Path, default=root / "reports/phase4/PHASE4_REPORT.md")
    parser.add_argument("--allow-incomplete", action="store_true", help="Render methodology and available figures without inventing missing results.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    paths = {
        "gold": args.gold,
        "feature_quality": args.quality,
        "benchmark": args.benchmark,
        "test_predictions": args.predictions,
        "region_performance": args.region_performance,
        "department_performance": args.department_performance,
        "feature_importance": args.feature_importance,
        "benchmark_comparison": args.benchmark_comparison,
        "validation_trials": args.validation_trials,
        "source_manifest": args.source_manifest,
        "production_model": args.model,
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing and not args.allow_incomplete:
        raise SystemExit(
            "Missing Phase 4 artifacts: " + ", ".join(missing) + ". Run the national pipeline or use --allow-incomplete."
        )
    setup_style()
    figures = [draw_architecture(args.output_dir)]
    gold = pd.read_parquet(args.gold) if args.gold.exists() else None
    quality = json.loads(args.quality.read_text(encoding="utf-8")) if args.quality.exists() else None
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8")) if args.benchmark.exists() else None
    predictions = pd.read_parquet(args.predictions) if args.predictions.exists() else None
    region = pd.read_csv(args.region_performance, dtype={"code_region": "string"}) if args.region_performance.exists() else None
    department = pd.read_csv(args.department_performance, dtype={"code_departement": "string"}) if args.department_performance.exists() else None
    importance = pd.read_csv(args.feature_importance) if args.feature_importance.exists() else None
    comparison = pd.read_csv(args.benchmark_comparison) if args.benchmark_comparison.exists() else None
    validation_trials = pd.read_csv(args.validation_trials) if args.validation_trials.exists() else None
    if gold is not None:
        figures.append(draw_coverage(gold, args.output_dir))
        figures.append(draw_comparable_coverage(gold, args.output_dir))
    if benchmark is not None and region is not None:
        figures.append(draw_performance(benchmark, region, args.output_dir))
    if predictions is not None:
        figures.append(draw_diagnostics(predictions, args.output_dir))
    if region is not None and department is not None:
        figures.append(draw_regional_performance(region, department, args.output_dir))
    if importance is not None:
        figures.append(draw_feature_importance(importance, args.output_dir))
    render_report(
        args.report,
        args.output_dir,
        paths,
        figures,
        benchmark,
        quality,
        region,
        comparison,
        validation_trials,
    )


if __name__ == "__main__":
    main()
