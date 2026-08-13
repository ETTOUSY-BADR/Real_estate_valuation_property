"""Create a focused dashboard for the untouched 2025 valuation test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paris_avm.paths import PROJECT_ROOT


NAVY = "#0B2E59"
BLUE = "#3478B9"
CORAL = "#F05B61"
TEAL = "#35A79C"
PALE = "#EAF2F8"
GRAY = "#667085"


def add_metric_card(
    ax: plt.Axes, x: float, title: str, value: str, detail: str, color: str = NAVY
) -> None:
    ax.text(
        x,
        0.58,
        value,
        ha="center",
        va="center",
        fontsize=21,
        fontweight="bold",
        color=color,
        bbox=dict(boxstyle="round,pad=0.7", facecolor=PALE, edgecolor="#C7D7E5"),
    )
    ax.text(x, 0.91, title, ha="center", va="center", fontsize=11, color=GRAY)
    ax.text(x, 0.14, detail, ha="center", va="center", fontsize=9, color=GRAY)


def interval_calibration(
    calibration: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    calibration_ratio = (
        calibration["valeur_fonciere"]
        / calibration["predicted_value"].clip(lower=1)
    ).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    test_ratio = (
        test["valeur_fonciere"] / test["predicted_value"].clip(lower=1)
    ).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    nominal = np.array([0.50, 0.80, 0.90, 0.95])
    observed = []
    for level in nominal:
        tail = (1 - level) / 2
        lower, upper = np.quantile(calibration_ratio, [tail, 1 - tail])
        observed.append(np.mean((test_ratio >= lower) & (test_ratio <= upper)))
    return nominal * 100, np.asarray(observed) * 100


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-predictions",
        type=Path,
        default=PROJECT_ROOT / "reports/test_predictions.csv",
    )
    parser.add_argument(
        "--validation-predictions",
        type=Path,
        default=PROJECT_ROOT / "reports/validation_predictions.csv",
    )
    parser.add_argument("--metrics", type=Path, default=PROJECT_ROOT / "reports/metrics.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "visuals/test_results_dashboard.png",
    )
    args = parser.parse_args()

    test = pd.read_csv(args.test_predictions, dtype={"code_postal": "string"})
    calibration = pd.read_csv(
        args.validation_predictions, dtype={"code_postal": "string"}
    )
    report = json.loads(args.metrics.read_text(encoding="utf-8"))
    test["date_mutation"] = pd.to_datetime(test["date_mutation"])
    test["signed_percentage_error"] = (
        (test["predicted_value"] - test["valeur_fonciere"])
        / test["valeur_fonciere"]
        * 100
    )
    test["absolute_percentage_error"] = test["signed_percentage_error"].abs()

    model = report["model_metrics"]
    baseline = report["baseline_metrics"]
    interval = report["prediction_interval"]
    mae_gain = (baseline["mae_eur"] - model["mae_eur"]) / baseline["mae_eur"] * 100
    mdape_gain = (
        baseline["median_absolute_percentage_error"]
        - model["median_absolute_percentage_error"]
    ) / baseline["median_absolute_percentage_error"] * 100
    within20_gain = (model["within_20_percent"] - baseline["within_20_percent"]) * 100

    fig = plt.figure(figsize=(18, 13), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[0.42, 1.35, 1.25])
    fig.suptitle(
        "Paris Valuation Model — Untouched 2025 Test Results",
        fontsize=24,
        color=NAVY,
        fontweight="bold",
    )

    cards = fig.add_subplot(grid[0, :])
    cards.axis("off")
    add_metric_card(cards, 0.09, "UNSEEN SALES", f"{len(test):,}", "Full 2025 test year")
    add_metric_card(cards, 0.295, "R²", f"{model['r2']:.3f}", f"Baseline {baseline['r2']:.3f}")
    add_metric_card(
        cards,
        0.50,
        "MEAN ABSOLUTE ERROR",
        f"€{model['mae_eur'] / 1000:.1f}k",
        f"{mae_gain:.1f}% lower than baseline",
    )
    add_metric_card(
        cards,
        0.705,
        "MEDIAN % ERROR",
        f"{model['median_absolute_percentage_error']:.1f}%",
        f"{mdape_gain:.1f}% lower than baseline",
    )
    add_metric_card(
        cards,
        0.91,
        "90% INTERVAL COVERAGE",
        f"{interval['observed_test_coverage'] * 100:.1f}%",
        "Calibrated on 2024 only",
        TEAL,
    )

    # Actual versus predicted values.
    ax = fig.add_subplot(grid[1, 0])
    display_limit = float(
        np.quantile(
            np.concatenate(
                [test["valeur_fonciere"].to_numpy(), test["predicted_value"].to_numpy()]
            ),
            0.99,
        )
    )
    visible = test.loc[
        test["valeur_fonciere"].le(display_limit)
        & test["predicted_value"].le(display_limit)
    ]
    plot = ax.hexbin(
        visible["valeur_fonciere"] / 1_000,
        visible["predicted_value"] / 1_000,
        gridsize=55,
        bins="log",
        mincnt=1,
        cmap="Blues",
    )
    limit_thousands = display_limit / 1_000
    ax.plot([0, limit_thousands], [0, limit_thousands], color=CORAL, linewidth=2)
    fig.colorbar(plot, ax=ax, label="Log number of sales")
    ax.set(
        xlim=(0, limit_thousands),
        ylim=(0, limit_thousands),
        xlabel="Actual transaction value (€000)",
        ylabel="Predicted transaction value (€000)",
        title="Actual vs predicted — closer to the red line is better",
    )
    ax.grid(alpha=0.15)

    # Empirical cumulative distribution of absolute errors.
    ax = fig.add_subplot(grid[1, 1])
    errors = np.sort(test["absolute_percentage_error"].to_numpy())
    cumulative = np.arange(1, len(errors) + 1) / len(errors) * 100
    ax.plot(errors, cumulative, color=BLUE, linewidth=3)
    ax.axvline(10, color=NAVY, linestyle="--", alpha=0.8)
    ax.axvline(20, color=CORAL, linestyle="--", alpha=0.8)
    ax.scatter(
        [10, 20],
        [model["within_10_percent"] * 100, model["within_20_percent"] * 100],
        color=[NAVY, CORAL],
        s=65,
        zorder=3,
    )
    ax.annotate(
        f"{model['within_10_percent'] * 100:.1f}% within 10%",
        (10, model["within_10_percent"] * 100),
        xytext=(10, -18),
        textcoords="offset points",
        color=NAVY,
    )
    ax.annotate(
        f"{model['within_20_percent'] * 100:.1f}% within 20%\n(+{within20_gain:.1f} points vs baseline)",
        (20, model["within_20_percent"] * 100),
        xytext=(10, -4),
        textcoords="offset points",
        color=CORAL,
    )
    ax.set(
        xlim=(0, 60),
        ylim=(0, 100),
        xlabel="Maximum absolute percentage error",
        ylabel="Share of test sales (%)",
        title="How many test predictions fall within each error threshold?",
    )
    ax.grid(alpha=0.18)

    # Monthly accuracy and signed bias.
    ax = fig.add_subplot(grid[2, 0])
    monthly = test.groupby(test["date_mutation"].dt.month).agg(
        median_error=("absolute_percentage_error", "median"),
        signed_bias=("signed_percentage_error", "median"),
        sales=("id_mutation", "size"),
    )
    ax.bar(monthly.index, monthly["median_error"], color=BLUE, alpha=0.88)
    ax.plot(
        monthly.index,
        monthly["signed_bias"],
        color=CORAL,
        marker="o",
        linewidth=2.5,
        label="Median signed bias",
    )
    ax.axhline(0, color=GRAY, linewidth=1)
    for month, row in monthly.iterrows():
        ax.text(month, 0.35, f"n={int(row['sales'])}", ha="center", fontsize=7, rotation=90, color="white")
    ax.set(
        xticks=range(1, 13),
        xlabel="Month in 2025",
        ylabel="Error (%)",
        title="Monthly stability — bars are median absolute error",
    )
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.18)

    # Calibration learned in 2024 and tested in 2025.
    ax = fig.add_subplot(grid[2, 1])
    nominal, observed = interval_calibration(calibration, test)
    ax.plot([45, 100], [45, 100], color=GRAY, linestyle="--", label="Perfect calibration")
    ax.plot(nominal, observed, color=TEAL, marker="o", linewidth=3, markersize=9)
    for x, y in zip(nominal, observed):
        ax.annotate(f"{y:.1f}%", (x, y), xytext=(7, 6), textcoords="offset points", color=NAVY)
    ax.set(
        xlim=(45, 100),
        ylim=(45, 100),
        xlabel="Promised interval coverage (%)",
        ylabel="Observed 2025 coverage (%)",
        title="Uncertainty calibration — learned on 2024, tested on 2025",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)

    fig.text(
        0.5,
        0.002,
        "DVF ordinary apartment transactions · chronological evaluation · no 2025 sale used to fit the tested model",
        ha="center",
        color=GRAY,
        fontsize=10,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
