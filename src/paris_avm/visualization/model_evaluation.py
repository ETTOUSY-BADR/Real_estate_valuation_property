"""Generate tables and figures for the preprocessing/model evaluation report.

All predictive diagnostics use the untouched 2025 test export.  The script is
read-only with respect to models: it never fits, tunes or selects a candidate.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np
import pandas as pd
import joblib

from paris_avm.artifacts import write_csv_atomic
from paris_avm.paths import PROJECT_ROOT
from paris_avm.visualization.phase3 import (
    BLUE,
    CYAN,
    GOLD,
    GREEN,
    GRID,
    INK,
    LIGHT,
    MUTED,
    NAVY,
    PURPLE,
    RED,
    clean_axis,
    feature_family,
    pretty_feature,
    setup_style,
)


PHASE2_PREDICTION = "prediction_phase2_comparable_correction"
PHASE3_PREDICTION = "prediction_catboost_full_identity"
REPORT_DIR = PROJECT_ROOT / "reports" / "model_evaluation"
VISUAL_DIR = PROJECT_ROOT / "visuals" / "model_evaluation"
PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "phase3" / "phase3_test_predictions.csv"
GOLD_PATH = PROJECT_ROOT / "data" / "gold" / "phase3_sale_features.parquet"
COMPARISON_PATH = PROJECT_ROOT / "reports" / "phase3" / "phase3_model_comparison.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "phase3" / "phase3_selected_model.joblib"


def add_error_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Return row-level paired error columns without modifying the input."""
    output = data.copy()
    actual = output["valeur_fonciere"].to_numpy(dtype=float)
    for label, column in (("phase2", PHASE2_PREDICTION), ("phase3", PHASE3_PREDICTION)):
        residual = output[column].to_numpy(dtype=float) - actual
        output[f"{label}_residual"] = residual
        output[f"{label}_absolute_error"] = np.abs(residual)
        output[f"{label}_absolute_percentage_error"] = np.abs(residual) / actual * 100
    output["paired_improvement_eur"] = (
        output["phase2_absolute_error"] - output["phase3_absolute_error"]
    )
    return output


def summarize_segments(data: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Summarize paired model performance for one precomputed segment column."""
    rows: list[dict[str, object]] = []
    for segment, group in data.groupby(dimension, observed=False, sort=True):
        if not len(group):
            continue
        phase2_mae = float(group["phase2_absolute_error"].mean())
        phase3_mae = float(group["phase3_absolute_error"].mean())
        reduction = phase2_mae - phase3_mae
        rows.append(
            {
                "dimension": dimension,
                "segment": str(segment),
                "transactions": len(group),
                "phase2_mae_eur": round(phase2_mae, 2),
                "phase3_mae_eur": round(phase3_mae, 2),
                "mae_reduction_eur": round(reduction, 2),
                "mae_reduction_percent": round(reduction / phase2_mae * 100, 3),
                "phase3_median_ape_percent": round(
                    float(group["phase3_absolute_percentage_error"].median()), 3
                ),
                "phase3_within_20_percent": round(
                    float((group["phase3_absolute_percentage_error"] <= 20).mean()), 5
                ),
                "phase3_mean_bias_eur": round(float(group["phase3_residual"].mean()), 2),
            }
        )
    return pd.DataFrame(rows)


def build_segment_metrics(predictions: pd.DataFrame, gold: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build detailed test diagnostics across time, place, size and coverage."""
    context_columns = [
        "id_mutation",
        "local_365d_500m_count",
        "dpe_available_at_sale",
    ]
    data = predictions.merge(
        gold[context_columns], on="id_mutation", how="left", validate="one_to_one"
    )
    data["date_mutation"] = pd.to_datetime(data["date_mutation"])
    data = add_error_columns(data)
    data["month"] = pd.Categorical(
        data["date_mutation"].dt.month_name().str[:3],
        categories=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        ordered=True,
    )
    data["arrondissement"] = pd.Categorical(
        data["code_postal"].astype("string").str[-2:].str.zfill(2),
        categories=[f"{value:02d}" for value in range(1, 21)],
        ordered=True,
    )
    data["surface_band"] = pd.cut(
        data["surface_reelle_bati"],
        bins=[0, 20, 40, 60, 80, 120, np.inf],
        labels=["<20", "20-40", "40-60", "60-80", "80-120", "120+"],
        right=False,
    )
    data["value_band"] = pd.cut(
        data["valeur_fonciere"],
        bins=[0, 250_000, 500_000, 750_000, 1_000_000, 2_000_000, np.inf],
        labels=["<250k", "250-500k", "500-750k", "750k-1m", "1-2m", "2m+"],
        right=False,
    )
    data["comparable_density"] = pd.cut(
        data["local_365d_500m_count"],
        bins=[-1, 0, 50, 200, 500, np.inf],
        labels=["0", "1-49", "50-199", "200-499", "500+"],
    )
    data["dpe_status"] = pd.Categorical(
        np.where(data["dpe_available_at_sale"].eq(1), "Available by sale", "Unavailable"),
        categories=["Available by sale", "Unavailable"],
        ordered=True,
    )

    dimensions = [
        "month",
        "arrondissement",
        "surface_band",
        "value_band",
        "comparable_density",
        "dpe_status",
    ]
    segments = pd.concat(
        [summarize_segments(data, dimension) for dimension in dimensions],
        ignore_index=True,
    )

    data["value_decile"] = pd.qcut(
        data["valeur_fonciere"], 10, labels=[str(index) for index in range(1, 11)]
    )
    calibration = (
        data.groupby("value_decile", observed=False)
        .agg(
            transactions=("id_mutation", "size"),
            mean_actual_eur=("valeur_fonciere", "mean"),
            mean_prediction_eur=(PHASE3_PREDICTION, "mean"),
            mean_bias_eur=("phase3_residual", "mean"),
            mae_eur=("phase3_absolute_error", "mean"),
            median_ape_percent=("phase3_absolute_percentage_error", "median"),
        )
        .reset_index()
    )
    for column in calibration.columns[2:]:
        calibration[column] = calibration[column].round(2)
    return segments, calibration


def save_figure(fig: plt.Figure, filename: str) -> None:
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    path = VISUAL_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path.relative_to(PROJECT_ROOT)}")


def segment_frame(metrics: pd.DataFrame, dimension: str) -> pd.DataFrame:
    return metrics.loc[metrics["dimension"].eq(dimension)].reset_index(drop=True)


def draw_segment_dashboard(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.2))
    fig.suptitle(
        "Frozen 2025 performance across operational segments",
        fontsize=19,
        fontweight="bold",
        color=NAVY,
        y=0.99,
    )
    fig.text(
        0.5,
        0.955,
        "Every panel compares predictions on the same transactions; lower MAE is better",
        ha="center",
        color=MUTED,
        fontsize=11,
    )

    month = segment_frame(metrics, "month")
    x = np.arange(len(month))
    axes[0, 0].plot(x, month["phase2_mae_eur"] / 1_000, marker="o", color=MUTED, label="Phase 2")
    axes[0, 0].plot(x, month["phase3_mae_eur"] / 1_000, marker="o", color=GREEN, linewidth=2.4, label="Phase 3")
    axes[0, 0].set_xticks(x, month["segment"])
    axes[0, 0].set_ylabel("MAE (EUR thousands)")
    axes[0, 0].set_title("Monthly stability", loc="left")
    axes[0, 0].legend(frameon=False, ncol=2)
    clean_axis(axes[0, 0])

    arrondissement = segment_frame(metrics, "arrondissement").sort_values("segment")
    colors = np.where(arrondissement["mae_reduction_eur"] >= 0, GREEN, RED)
    axes[0, 1].bar(
        arrondissement["segment"], arrondissement["mae_reduction_eur"] / 1_000, color=colors
    )
    axes[0, 1].axhline(0, color=INK, linewidth=0.8)
    axes[0, 1].set_xlabel("Arrondissement")
    axes[0, 1].set_ylabel("MAE reduction (EUR thousands)")
    axes[0, 1].set_title("Improvement by arrondissement", loc="left")
    clean_axis(axes[0, 1])

    surface = segment_frame(metrics, "surface_band")
    x = np.arange(len(surface))
    width = 0.36
    axes[1, 0].bar(x - width / 2, surface["phase2_mae_eur"] / 1_000, width, color=MUTED, label="Phase 2")
    axes[1, 0].bar(x + width / 2, surface["phase3_mae_eur"] / 1_000, width, color=GREEN, label="Phase 3")
    axes[1, 0].set_xticks(x, surface["segment"])
    axes[1, 0].set_xlabel("Apartment surface (m2)")
    axes[1, 0].set_ylabel("MAE (EUR thousands)")
    axes[1, 0].set_title("Error scales with property size", loc="left")
    axes[1, 0].legend(frameon=False, ncol=2)
    clean_axis(axes[1, 0])

    value = segment_frame(metrics, "value_band")
    x = np.arange(len(value))
    axes[1, 1].bar(x - width / 2, value["phase2_mae_eur"] / 1_000, width, color=MUTED, label="Phase 2")
    axes[1, 1].bar(x + width / 2, value["phase3_mae_eur"] / 1_000, width, color=GREEN, label="Phase 3")
    axes[1, 1].set_xticks(x, value["segment"], rotation=20)
    axes[1, 1].set_xlabel("Transaction value")
    axes[1, 1].set_ylabel("MAE (EUR thousands)")
    axes[1, 1].set_title("Performance by price band", loc="left")
    axes[1, 1].legend(frameon=False, ncol=2)
    clean_axis(axes[1, 1])

    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.5, w_pad=2.0)
    save_figure(fig, "model_evaluation_segment_dashboard.png")


def draw_coverage_diagnostics(metrics: pd.DataFrame, calibration: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.2))
    fig.suptitle(
        "Coverage, information density and calibration diagnostics",
        fontsize=19,
        fontweight="bold",
        color=NAVY,
        y=0.99,
    )

    comparable = segment_frame(metrics, "comparable_density")
    x = np.arange(len(comparable))
    axes[0, 0].bar(x, comparable["transactions"], color=BLUE)
    axes[0, 0].set_xticks(x, comparable["segment"])
    axes[0, 0].set_xlabel("Prior sales within 500 m / 365 days")
    axes[0, 0].set_ylabel("2025 transactions")
    axes[0, 0].set_title("Historical-comparable density", loc="left")
    clean_axis(axes[0, 0])

    colors = np.where(comparable["mae_reduction_eur"] >= 0, GREEN, RED)
    axes[0, 1].bar(x, comparable["mae_reduction_eur"] / 1_000, color=colors)
    axes[0, 1].axhline(0, color=INK, linewidth=0.8)
    axes[0, 1].set_xticks(x, comparable["segment"])
    axes[0, 1].set_xlabel("Prior sales within 500 m / 365 days")
    axes[0, 1].set_ylabel("Phase 3 MAE reduction (EUR thousands)")
    axes[0, 1].set_title("Enrichment gain by comparable density", loc="left")
    clean_axis(axes[0, 1])

    dpe = segment_frame(metrics, "dpe_status")
    x = np.arange(len(dpe))
    axes[1, 0].bar(x - 0.18, dpe["phase2_mae_eur"] / 1_000, 0.36, color=MUTED, label="Phase 2")
    axes[1, 0].bar(x + 0.18, dpe["phase3_mae_eur"] / 1_000, 0.36, color=PURPLE, label="Phase 3")
    axes[1, 0].set_xticks(x, dpe["segment"])
    axes[1, 0].set_ylabel("MAE (EUR thousands)")
    axes[1, 0].set_title("Performance by point-in-time DPE availability", loc="left")
    axes[1, 0].legend(frameon=False, ncol=2)
    clean_axis(axes[1, 0])

    x = calibration["mean_actual_eur"] / 1_000
    y = calibration["mean_prediction_eur"] / 1_000
    axes[1, 1].plot(x, y, marker="o", color=GREEN, linewidth=2.2)
    limits = [min(x.min(), y.min()), max(x.max(), y.max())]
    axes[1, 1].plot(limits, limits, linestyle="--", color=RED, label="Perfect calibration")
    for _, row in calibration.iterrows():
        axes[1, 1].annotate(str(row["value_decile"]), (row["mean_actual_eur"] / 1_000, row["mean_prediction_eur"] / 1_000), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[1, 1].set_xlabel("Mean actual value (EUR thousands)")
    axes[1, 1].set_ylabel("Mean predicted value (EUR thousands)")
    axes[1, 1].set_title("Calibration by actual-value decile", loc="left")
    axes[1, 1].legend(frameon=False)
    clean_axis(axes[1, 1], grid_axis="both")

    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5, w_pad=2.0)
    save_figure(fig, "model_evaluation_coverage_diagnostics.png")


def draw_feature_contract() -> None:
    fig, ax = plt.subplots(figsize=(15.5, 7.8))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 8)
    ax.axis("off")
    fig.suptitle(
        "Five-part preprocessing contract",
        fontsize=20,
        fontweight="bold",
        color=NAVY,
        y=0.97,
    )
    ax.text(
        7.75,
        7.22,
        "One mutation per row, explicit provenance, chronological evaluation",
        ha="center",
        fontsize=11,
        color=MUTED,
    )

    stages = [
        ("1", "Property", "Surface, rooms, lots\nlog surface, room density", BLUE),
        ("2", "Time", "Month, day, year\nmonths since 2021", CYAN),
        ("3", "Location", "WGS84, Lambert-93\ndistance, bearing, H3", PURPLE),
        ("4", "Comparables", "7 space-time windows\n90-day publication lag", GOLD),
        ("5", "Enrichment", "BAN, BDNB, dated DPE\ntransport, services, noise", GREEN),
    ]
    positions = np.linspace(0.35, 12.55, len(stages))
    for index, ((number, title, body, color), x) in enumerate(zip(stages, positions)):
        box = FancyBboxPatch(
            (x, 4.1),
            2.6,
            1.65,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            facecolor=LIGHT,
            edgecolor=color,
            linewidth=2.0,
        )
        ax.add_patch(box)
        ax.add_patch(plt.Circle((x + 0.35, 5.38), 0.22, color=color))
        ax.text(x + 0.35, 5.38, number, ha="center", va="center", color="white", fontweight="bold")
        ax.text(x + 0.7, 5.32, title, fontsize=13, fontweight="bold", color=NAVY)
        ax.text(x + 0.2, 4.45, body, fontsize=9.5, color=INK, linespacing=1.35)
        if index < len(stages) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + 2.62, 4.92),
                    (positions[index + 1] - 0.05, 4.92),
                    arrowstyle="-|>",
                    mutation_scale=14,
                    linewidth=1.4,
                    color=MUTED,
                )
            )

    safeguards = [
        ("Quality filters", "Single-apartment sales\nFixed value, surface and price/m2 bounds\nParis coordinates only"),
        ("Leakage controls", "No current-sale price/m2 predictor\nPrior sales lagged at least 90 days\nDPE date no later than sale date"),
        ("Evaluation", "2021-2023 development\n2024 configuration selection\n2025 frozen test: 28,330 sales"),
    ]
    for index, (title, body) in enumerate(safeguards):
        x = 0.8 + index * 4.9
        patch = FancyBboxPatch(
            (x, 1.0),
            4.25,
            1.65,
            boxstyle="round,pad=0.05,rounding_size=0.1",
            facecolor="white",
            edgecolor=GRID,
            linewidth=1.4,
        )
        ax.add_patch(patch)
        ax.text(x + 0.18, 2.3, title, fontsize=11, fontweight="bold", color=NAVY)
        ax.text(x + 0.18, 1.25, body, fontsize=8.8, color=MUTED, linespacing=1.35)

    ax.text(
        7.75,
        0.45,
        "44 Phase 2 model inputs  +  56 building, DPE, context and external-identity inputs  =  100 Phase 3 inputs",
        ha="center",
        fontsize=12,
        color=NAVY,
        fontweight="bold",
    )
    save_figure(fig, "model_evaluation_feature_contract.png")


def draw_model_ladder(comparison: pd.DataFrame) -> None:
    selected_keys = [
        "phase2_comparable_correction",
        "lightgbm_phase2_mae",
        "lightgbm_context_mae",
        "lightgbm_dpe_safe_mae",
        "lightgbm_building_mae",
        "lightgbm_full_no_identity_mae",
        "catboost_full_identity",
    ]
    labels = {
        "phase2_comparable_correction": "Phase 2 correction",
        "lightgbm_phase2_mae": "Phase 2 features / LightGBM",
        "lightgbm_context_mae": "+ neighborhood context",
        "lightgbm_dpe_safe_mae": "+ dated DPE",
        "lightgbm_building_mae": "+ building attributes",
        "lightgbm_full_no_identity_mae": "All enrichment, no identity",
        "catboost_full_identity": "All 100 features / CatBoost",
    }
    frame = comparison.set_index("model_key").loc[selected_keys].reset_index()
    frame["label"] = frame["model_key"].map(labels)
    frame = frame.sort_values("mae_eur", ascending=True)

    fig, ax = plt.subplots(figsize=(12.8, 6.8))
    colors = [GREEN if key == "catboost_full_identity" else BLUE for key in frame["model_key"]]
    colors[frame["model_key"].tolist().index("phase2_comparable_correction")] = MUTED
    ax.barh(frame["label"], frame["mae_eur"] / 1_000, color=colors)
    ax.errorbar(
        frame["mae_eur"] / 1_000,
        frame["label"],
        xerr=np.vstack(
            [
                (frame["mae_eur"] - frame["mae_ci95_lower_eur"]) / 1_000,
                (frame["mae_ci95_upper_eur"] - frame["mae_eur"]) / 1_000,
            ]
        ),
        fmt="none",
        ecolor=NAVY,
        capsize=3,
        linewidth=1.3,
    )
    for position, value in enumerate(frame["mae_eur"]):
        ax.text(value / 1_000 + 0.18, position, f"EUR {value:,.0f}", va="center", fontweight="bold")
    ax.set_xlim(90, 99)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"EUR {value:.0f}k"))
    ax.set_xlabel("Mean absolute error with 95% bootstrap interval")
    ax.set_title("Model and feature-family performance ladder", loc="left", fontsize=18, color=NAVY)
    ax.text(0, 1.02, "Frozen 2025 cohort; lower is better", transform=ax.transAxes, color=MUTED)
    clean_axis(ax, grid_axis="x")
    fig.tight_layout()
    save_figure(fig, "model_evaluation_performance_ladder.png")


def build_feature_importance(model_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract feature and family importance from the selected CatBoost bundle."""
    bundle = joblib.load(model_path)
    importance = pd.DataFrame(
        {
            "feature": bundle["features"],
            "importance": bundle["model"].get_feature_importance(),
        }
    ).sort_values("importance", ascending=False, ignore_index=True)
    importance["feature_label"] = importance["feature"].map(pretty_feature)
    importance["family"] = importance["feature"].map(feature_family)
    importance["importance_percent"] = (
        importance["importance"] / importance["importance"].sum() * 100
    ).round(4)
    family = (
        importance.groupby("family", as_index=False)["importance_percent"]
        .sum()
        .sort_values("importance_percent", ascending=False, ignore_index=True)
    )
    return importance, family


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    parser.add_argument("--gold", type=Path, default=GOLD_PATH)
    parser.add_argument("--comparison", type=Path, default=COMPARISON_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args(argv)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    predictions = pd.read_csv(args.predictions, parse_dates=["date_mutation"])
    gold = pd.read_parquet(
        args.gold,
        columns=["id_mutation", "local_365d_500m_count", "dpe_available_at_sale"],
    )
    comparison = pd.read_csv(args.comparison)
    metrics, calibration = build_segment_metrics(predictions, gold)
    importance, family_importance = build_feature_importance(args.model)
    write_csv_atomic(metrics, REPORT_DIR / "segment_performance.csv")
    write_csv_atomic(calibration, REPORT_DIR / "calibration_by_value_decile.csv")
    write_csv_atomic(
        comparison.sort_values("mae_rank"), REPORT_DIR / "model_performance.csv"
    )
    write_csv_atomic(importance, REPORT_DIR / "feature_importance.csv")
    write_csv_atomic(family_importance, REPORT_DIR / "feature_family_importance.csv")

    draw_feature_contract()
    draw_model_ladder(comparison)
    draw_segment_dashboard(metrics)
    draw_coverage_diagnostics(metrics, calibration)
    print(f"Saved report tables to {REPORT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
