"""Create publication figures and paired tests for the model benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from paris_avm.paths import PROJECT_ROOT


NAVY = "#0B2E59"
BLUE = "#3478B9"
CORAL = "#F05B61"
TEAL = "#35A79C"
GRAY = "#667085"
PALETTE = ["#35A79C", "#3478B9", "#5B5F97", "#8B6F9C", "#E09F3E", "#F05B61"]


def paired_bootstrap(
    errors: pd.DataFrame, repetitions: int = 1_000
) -> pd.DataFrame:
    """Estimate paired MAE differences; negative means the row model is better."""
    rng = np.random.default_rng(20250811)
    keys = list(errors.columns)
    rows: list[dict[str, float | str]] = []
    for row_key in keys:
        for column_key in keys:
            paired_difference = (
                errors[row_key].to_numpy() - errors[column_key].to_numpy()
            )
            samples = np.empty(repetitions)
            for start in range(0, repetitions, 50):
                stop = min(start + 50, repetitions)
                indices = rng.integers(
                    0, len(paired_difference), size=(stop - start, len(paired_difference))
                )
                samples[start:stop] = paired_difference[indices].mean(axis=1)
            rows.append(
                {
                    "row_model_key": row_key,
                    "column_model_key": column_key,
                    "mae_difference_eur": float(paired_difference.mean()),
                    "ci95_lower_eur": float(np.quantile(samples, 0.025)),
                    "ci95_upper_eur": float(np.quantile(samples, 0.975)),
                    "row_model_significantly_better": bool(np.quantile(samples, 0.975) < 0),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        type=Path,
        default=PROJECT_ROOT / "reports/benchmark/model_comparison.csv",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_ROOT / "reports/benchmark/test_predictions_all_models.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "visuals/benchmark"
    )
    parser.add_argument(
        "--analysis-dir", type=Path, default=PROJECT_ROOT / "reports/benchmark"
    )
    args = parser.parse_args()

    comparison = pd.read_csv(args.comparison).sort_values("mae_eur")
    predictions = pd.read_csv(args.predictions)
    predictions["date_mutation"] = pd.to_datetime(predictions["date_mutation"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)

    labels = dict(zip(comparison["model_key"], comparison["model"]))
    ordered_keys = comparison["model_key"].tolist()
    errors = pd.DataFrame(
        {
            key: np.abs(
                predictions["valeur_fonciere"]
                - predictions[f"prediction_{key}"]
            )
            for key in ordered_keys
        }
    )
    paired = paired_bootstrap(errors)
    paired.to_csv(args.analysis_dir / "paired_mae_comparisons.csv", index=False)

    # Figure 1: main accuracy comparison.
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    fig.suptitle(
        "Six-Model Benchmark on the Common 2025 Test Cohort",
        fontsize=21,
        color=NAVY,
        fontweight="bold",
    )
    y = np.arange(len(comparison))
    model_colors = [TEAL] + [BLUE] * 3 + ["#E09F3E", CORAL]

    ax = axes[0, 0]
    values = comparison["mae_eur"].to_numpy() / 1_000
    lower = (
        comparison["mae_eur"] - comparison["mae_ci95_lower_eur"]
    ).to_numpy() / 1_000
    upper = (
        comparison["mae_ci95_upper_eur"] - comparison["mae_eur"]
    ).to_numpy() / 1_000
    ax.barh(y, values, color=model_colors, alpha=0.9)
    ax.errorbar(values, y, xerr=np.vstack([lower, upper]), fmt="none", ecolor=NAVY, capsize=4)
    ax.set_yticks(y, comparison["model"])
    ax.invert_yaxis()
    ax.set(xlabel="Mean absolute error (€000)", title="Primary metric with 95% bootstrap intervals")
    ax.grid(axis="x", alpha=0.18)
    for row, value in zip(y, values):
        ax.text(value + 1.5, row, f"€{value:.1f}k", va="center", color=NAVY)

    ax = axes[0, 1]
    ax.barh(y, comparison["median_absolute_percentage_error"], color=model_colors)
    ax.set_yticks(y, comparison["model"])
    ax.invert_yaxis()
    ax.set(xlabel="Median absolute percentage error (%)", title="Typical relative error")
    ax.grid(axis="x", alpha=0.18)
    for row, value in zip(y, comparison["median_absolute_percentage_error"]):
        ax.text(value + 0.25, row, f"{value:.2f}%", va="center", color=NAVY)

    ax = axes[1, 0]
    within20 = comparison["within_20_percent"].to_numpy() * 100
    ax.barh(y, within20, color=model_colors)
    ax.set_yticks(y, comparison["model"])
    ax.invert_yaxis()
    ax.set(xlim=(45, 75), xlabel="Test sales within ±20% (%)", title="Operational hit rate")
    ax.grid(axis="x", alpha=0.18)
    for row, value in zip(y, within20):
        ax.text(value + 0.35, row, f"{value:.1f}%", va="center", color=NAVY)

    ax = axes[1, 1]
    ax.scatter(
        comparison["validation_mae_eur"] / 1_000,
        comparison["mae_eur"] / 1_000,
        s=100,
        c=model_colors,
    )
    limits = [90, 175]
    ax.plot(limits, limits, color=GRAY, linestyle="--", label="No temporal change")
    for _, row in comparison.iterrows():
        ax.annotate(
            row["model"],
            (row["validation_mae_eur"] / 1_000, row["mae_eur"] / 1_000),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set(
        xlim=limits,
        ylim=limits,
        xlabel="2024 validation MAE (€000)",
        ylabel="2025 test MAE (€000)",
        title="Does validation performance transfer to the next year?",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)
    accuracy_path = args.output_dir / "benchmark_accuracy.png"
    fig.savefig(accuracy_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: accuracy, fit time, inference and artifact size.
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True)
    fig.suptitle("Accuracy–Efficiency Trade-offs", fontsize=20, color=NAVY, fontweight="bold")
    ax = axes[0]
    sizes = np.clip(np.sqrt(comparison["artifact_size_mb"].to_numpy() + 0.05) * 36, 35, 520)
    for index, row in comparison.reset_index(drop=True).iterrows():
        ax.scatter(
            max(row["final_fit_seconds"], 0.01),
            row["mae_eur"] / 1_000,
            s=sizes[index],
            color=model_colors[index],
            alpha=0.85,
            edgecolor="white",
            linewidth=1,
        )
        ax.annotate(
            row["model"],
            (max(row["final_fit_seconds"], 0.01), row["mae_eur"] / 1_000),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xscale("log")
    ax.set(
        xlabel="Final training time (seconds, log scale)",
        ylabel="2025 MAE (€000; lower is better)",
        title="Training cost versus accuracy\nBubble area reflects artifact size",
    )
    ax.grid(alpha=0.18)

    ax = axes[1]
    ax.barh(y, comparison["prediction_ms_per_1000"], color=model_colors)
    ax.set_yticks(y, comparison["model"])
    ax.invert_yaxis()
    ax.set(
        xlabel="Prediction latency (milliseconds per 1,000 properties)",
        title="Batch inference speed",
    )
    ax.grid(axis="x", alpha=0.18)
    efficiency_path = args.output_dir / "benchmark_efficiency.png"
    fig.savefig(efficiency_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: month-by-month MdAPE.
    monthly_rows = []
    for key in ordered_keys:
        frame = predictions[["date_mutation", "valeur_fonciere"]].copy()
        frame["ape"] = (
            np.abs(predictions[f"prediction_{key}"] - predictions["valeur_fonciere"])
            / predictions["valeur_fonciere"]
            * 100
        )
        monthly = frame.groupby(frame["date_mutation"].dt.month)["ape"].median()
        for month, value in monthly.items():
            monthly_rows.append({"model_key": key, "month": month, "mdape": value})
    monthly_frame = pd.DataFrame(monthly_rows)
    monthly_frame.to_csv(args.analysis_dir / "monthly_model_errors.csv", index=False)

    fig, ax = plt.subplots(figsize=(14, 7.5), constrained_layout=True)
    for color, key in zip(PALETTE, ordered_keys):
        model_months = monthly_frame.loc[monthly_frame["model_key"].eq(key)]
        ax.plot(
            model_months["month"],
            model_months["mdape"],
            marker="o",
            linewidth=2.3,
            color=color,
            label=labels[key],
        )
    ax.set(
        xticks=range(1, 13),
        xlabel="Month in 2025",
        ylabel="Median absolute percentage error (%)",
        title="Temporal Stability of the Six Models on 2025 Sales",
    )
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, ncol=2)
    monthly_path = args.output_dir / "benchmark_monthly_stability.png"
    fig.savefig(monthly_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Figure 4: paired mean absolute-error differences.
    difference = paired.pivot(
        index="row_model_key", columns="column_model_key", values="mae_difference_eur"
    ).loc[ordered_keys, ordered_keys]
    fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)
    maximum = float(np.abs(difference.to_numpy()).max()) / 1_000
    image = ax.imshow(
        difference.to_numpy() / 1_000,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum),
    )
    short_labels = [labels[key].replace("Histogram ", "Hist. ") for key in ordered_keys]
    ax.set_xticks(range(len(ordered_keys)), short_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(ordered_keys)), short_labels)
    for row in range(len(ordered_keys)):
        for column in range(len(ordered_keys)):
            value = difference.iloc[row, column] / 1_000
            ax.text(
                column,
                row,
                f"{value:+.1f}",
                ha="center",
                va="center",
                color="white" if abs(value) > maximum * 0.43 else NAVY,
                fontsize=9,
            )
    colorbar = fig.colorbar(image, ax=ax, shrink=0.8)
    colorbar.set_label("Row model MAE minus column model MAE (€000)")
    ax.set_title(
        "Paired Error Differences on Identical 2025 Transactions\nNegative values favor the row model",
        fontsize=16,
        color=NAVY,
    )
    pairwise_path = args.output_dir / "benchmark_pairwise_differences.png"
    fig.savefig(pairwise_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    for path in [accuracy_path, efficiency_path, monthly_path, pairwise_path]:
        print(f"Saved: {path}")
    print(f"Saved: {args.analysis_dir / 'paired_mae_comparisons.csv'}")
    print(f"Saved: {args.analysis_dir / 'monthly_model_errors.csv'}")


if __name__ == "__main__":
    main()
