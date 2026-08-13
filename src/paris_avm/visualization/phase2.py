"""Create diagnostics for the leakage-safe Phase 2 valuation benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paris_avm.paths import PROJECT_ROOT

REPORT_DIR = PROJECT_ROOT / "reports/phase2"
OUTPUT_DIR = PROJECT_ROOT / "visuals/phase2"
NAVY = "#15324B"
BLUE = "#3B82C4"
TEAL = "#1F9D8A"
GOLD = "#E5A43B"
CORAL = "#E66A5C"
GRAY = "#667085"


def save(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / name, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    comparison = pd.read_csv(REPORT_DIR / "phase2_model_comparison.csv")
    predictions = pd.read_csv(
        REPORT_DIR / "phase2_test_predictions.csv", parse_dates=["date_mutation"]
    )
    importance = pd.read_csv(REPORT_DIR / "phase2_feature_importance.csv")
    search = pd.read_csv(REPORT_DIR / "comparable_correction_search.csv")
    quality = json.loads((REPORT_DIR / "feature_quality.json").read_text(encoding="utf-8"))

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "font.size": 10,
            "figure.facecolor": "white",
        }
    )

    # 1. Honest 2025 benchmark summary.
    plot_rows = comparison.loc[
        comparison["model_key"].isin(
            [
                "phase2_comparable_correction",
                "phase2_xgboost_total",
                "phase1_lightgbm",
                "phase2_xgboost_ppm2",
                "phase2_blend",
                "phase2_xgboost_residual",
                "phase2_lightgbm_spatial_ablation",
                "phase2_lightgbm_comparables_ablation",
                "phase2_lightgbm_total",
            ]
        )
    ].sort_values("mae_eur")
    colors = [TEAL if k == "phase2_comparable_correction" else BLUE for k in plot_rows["model_key"]]
    colors[plot_rows["model_key"].tolist().index("phase1_lightgbm")] = GOLD

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    fig.suptitle("Phase 2: Leakage-Safe Spatial and Comparable-Sale Benchmark", fontsize=19, color=NAVY)
    y = np.arange(len(plot_rows))
    mae_k = plot_rows["mae_eur"].to_numpy() / 1000
    lower = (plot_rows["mae_eur"] - plot_rows["mae_ci95_lower_eur"]).to_numpy() / 1000
    upper = (plot_rows["mae_ci95_upper_eur"] - plot_rows["mae_eur"]).to_numpy() / 1000
    axes[0].barh(y, mae_k, color=colors)
    axes[0].errorbar(mae_k, y, xerr=np.vstack([lower, upper]), fmt="none", color=NAVY, capsize=3)
    axes[0].set_yticks(y, plot_rows["model"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("2025 mean absolute error (EUR thousands; lower is better)")
    axes[0].set_title("Final chronological holdout")
    axes[0].grid(axis="x", alpha=0.18)
    axes[0].set_xlim(90, max(mae_k) + 8)
    for row, value in zip(y, mae_k):
        axes[0].text(value + 0.45, row, f"EUR {value:.1f}k", va="center", color=NAVY, fontsize=9)

    phase1 = comparison.loc[comparison["model_key"].eq("phase1_lightgbm")].iloc[0]
    winner = comparison.loc[comparison["model_key"].eq("phase2_comparable_correction")].iloc[0]
    metrics = ["MAE\n(EUR k)", "MdAPE\n(%)", "Within 20%\n(%)"]
    p1 = [phase1.mae_eur / 1000, phase1.median_absolute_percentage_error, phase1.within_20_percent * 100]
    p2 = [winner.mae_eur / 1000, winner.median_absolute_percentage_error, winner.within_20_percent * 100]
    x = np.arange(3)
    width = 0.36
    axes[1].bar(x - width / 2, p1, width, label="Phase 1 LightGBM", color=GOLD)
    axes[1].bar(x + width / 2, p2, width, label="Phase 2 correction", color=TEAL)
    axes[1].set_xticks(x, metrics)
    axes[1].set_title("Selected method versus Phase 1")
    axes[1].grid(axis="y", alpha=0.18)
    axes[1].legend(frameon=False)
    for xpos, values in ((x - width / 2, p1), (x + width / 2, p2)):
        for pos, value in zip(xpos, values):
            axes[1].text(pos, value + 1.0, f"{value:.2f}", ha="center", color=NAVY, fontsize=9)
    save(fig, "phase2_accuracy.png")

    # 2. Comparable correction selection, determined only on 2024 validation data.
    label_map = {
        "weighted_comp_ppm2": "Distance/time weighted comps",
        "similar_weighted_comp_ppm2": "Similar-property weighted comps",
        "local_365d_500m_median_ppm2": "500 m / 365 d median",
        "local_365d_1000m_median_ppm2": "1 km / 365 d median",
    }
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    for color, (feature, group) in zip([BLUE, CORAL, TEAL, GOLD], search.groupby("comparable_feature", sort=False)):
        ax.plot(group["phase1_weight"] * 100, group["validation_mae_eur"] / 1000, color=color, linewidth=2.2, label=label_map[feature])
        best = group.loc[group["validation_mae_eur"].idxmin()]
        ax.scatter(best["phase1_weight"] * 100, best["validation_mae_eur"] / 1000, color=color, s=55, zorder=4)
    ax.axvline(75, color=NAVY, linestyle="--", alpha=0.65, label="Selected Phase 1 weight: 75%")
    ax.set(
        xlabel="Weight assigned to Phase 1 model (%)",
        ylabel="2024 validation MAE (EUR thousands)",
        title="Comparable-Sale Correction Was Selected Without Looking at 2025",
    )
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, ncol=2)
    save(fig, "comparable_weight_selection.png")

    # 3. Detailed improvement diagnostics.
    actual = predictions["valeur_fonciere"].to_numpy()
    p1_pred = predictions["prediction_phase1_lightgbm"].to_numpy()
    p2_pred = predictions["prediction_phase2_comparable_correction"].to_numpy()
    p1_error = np.abs(actual - p1_pred)
    p2_error = np.abs(actual - p2_pred)
    p1_ape = p1_error / actual * 100
    p2_ape = p2_error / actual * 100

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    fig.suptitle("What the Historical Comparable Correction Changes", fontsize=19, color=NAVY)
    cap = float(np.quantile(actual, 0.99)) / 1000
    sample = predictions.sample(min(5000, len(predictions)), random_state=42)
    axes[0, 0].scatter(sample["valeur_fonciere"] / 1000, sample["prediction_phase2_comparable_correction"] / 1000, s=9, alpha=0.25, color=TEAL)
    axes[0, 0].plot([0, cap], [0, cap], color=NAVY, linestyle="--")
    axes[0, 0].set(xlim=(0, cap), ylim=(0, cap), xlabel="Actual value (EUR k)", ylabel="Corrected estimate (EUR k)", title="Actual versus corrected prediction")
    axes[0, 0].grid(alpha=0.18)

    for values, color, label in [(p1_ape, GOLD, "Phase 1"), (p2_ape, TEAL, "Phase 2 correction")]:
        clipped = np.sort(values[values <= 100])
        axes[0, 1].plot(clipped, np.arange(1, len(clipped) + 1) / len(clipped) * 100, color=color, linewidth=2.2, label=label)
    axes[0, 1].axvline(20, color=NAVY, linestyle="--", alpha=0.7)
    axes[0, 1].set(xlim=(0, 60), xlabel="Absolute percentage error (%)", ylabel="Cumulative share of sales (%)", title="Error distribution")
    axes[0, 1].grid(alpha=0.18)
    axes[0, 1].legend(frameon=False)

    monthly = pd.DataFrame({"month": predictions["date_mutation"].dt.month, "phase1": p1_error, "phase2": p2_error}).groupby("month").mean() / 1000
    monthly.to_csv(REPORT_DIR / "phase2_monthly_mae.csv")
    axes[1, 0].plot(monthly.index, monthly["phase1"], marker="o", color=GOLD, linewidth=2.2, label="Phase 1")
    axes[1, 0].plot(monthly.index, monthly["phase2"], marker="o", color=TEAL, linewidth=2.2, label="Phase 2 correction")
    axes[1, 0].set(xticks=range(1, 13), xlabel="Month in 2025", ylabel="MAE (EUR k)", title="Month-by-month stability")
    axes[1, 0].grid(alpha=0.18)
    axes[1, 0].legend(frameon=False)

    difference = p1_error - p2_error
    axes[1, 1].hist(np.clip(difference / 1000, -150, 150), bins=70, color=BLUE, alpha=0.85)
    axes[1, 1].axvline(0, color=NAVY, linestyle="--")
    axes[1, 1].set(xlabel="Phase 1 absolute error minus Phase 2 error (EUR k)", ylabel="Number of sales", title="Positive values mean the correction helped")
    axes[1, 1].grid(axis="y", alpha=0.18)
    save(fig, "correction_diagnostics.png")

    # 4. Data-engineering coverage by year.
    coverage = quality["comparable_feature_contract"]["coverage_by_year"]
    years = np.array([int(year) for year in coverage])
    weighted = np.array([coverage[str(year)]["weighted_comparable_coverage"] * 100 for year in years])
    similar = np.array([coverage[str(year)]["similar_comparable_coverage"] * 100 for year in years])
    count500 = np.array([coverage[str(year)]["median_count_365d_500m"] for year in years])
    count1000 = np.array([coverage[str(year)]["median_count_365d_1000m"] for year in years])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    axes[0].plot(years, weighted, marker="o", color=BLUE, linewidth=2.2, label="Any weighted comparable")
    axes[0].plot(years, similar, marker="o", color=TEAL, linewidth=2.2, label="Similar-property comparable")
    axes[0].set(ylim=(70, 101), xticks=years, xlabel="Transaction year", ylabel="Coverage (%)", title="Historical-comparable availability")
    axes[0].grid(alpha=0.18)
    axes[0].legend(frameon=False)
    axes[1].plot(years, count500, marker="o", color=TEAL, linewidth=2.2, label="Within 500 m")
    axes[1].plot(years, count1000, marker="o", color=BLUE, linewidth=2.2, label="Within 1 km")
    axes[1].set(xticks=years, xlabel="Transaction year", ylabel="Median prior sales in trailing 365 days", title="Local evidence density")
    axes[1].grid(alpha=0.18)
    axes[1].legend(frameon=False)
    save(fig, "comparable_coverage.png")

    # 5. Feature importance for the two full total-price boosters.
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)
    for ax, key, title, color in [
        (axes[0], "phase2_lightgbm_total", "LightGBM total-price model", TEAL),
        (axes[1], "phase2_xgboost_total", "XGBoost total-price model", BLUE),
    ]:
        top = importance.loc[importance["model_key"].eq(key)].nlargest(15, "normalized_gain").sort_values("normalized_gain")
        ax.barh(top["feature"].str.replace("_", " "), top["normalized_gain"] * 100, color=color)
        ax.set(xlabel="Normalized gain importance (%)", title=title)
        ax.grid(axis="x", alpha=0.18)
    fig.suptitle("Which Phase 2 Features the Full Models Used", fontsize=19, color=NAVY)
    save(fig, "phase2_feature_importance.png")

    print(f"Saved five Phase 2 figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
