"""Build and test an empirical probability distribution around one valuation.

This is not a Bayesian posterior. It uses 2024 out-of-sample target/prediction
ratios to calibrate uncertainty, then tests interval coverage on unseen 2025
sales.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paris_avm.paths import PROJECT_ROOT


NAVY = "#0B2E59"
BLUE = "#2B6CB0"
CORAL = "#EF5B5B"
PALE = "#DCEAF7"
GRAY = "#667085"


def validate_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not 9 <= args.surface <= 300:
        parser.error("--surface must be between 9 and 300 m²")
    if not 0 <= args.rooms <= 15:
        parser.error("--rooms must be between 0 and 15")
    if not re.fullmatch(r"750(?:0[1-9]|1[0-9]|20)", str(args.postal_code)):
        parser.error("--postal-code must be between 75001 and 75020")
    if not 48.80 <= args.latitude <= 48.92 or not 2.20 <= args.longitude <= 2.48:
        parser.error("coordinates must be located within Paris")
    valuation_date = pd.to_datetime(args.date, errors="raise")
    if valuation_date.year != 2025:
        parser.error("--date must be in 2025; this model cannot forecast other years")


def property_features(args: argparse.Namespace) -> pd.DataFrame:
    valuation_date = pd.to_datetime(args.date)
    return pd.DataFrame(
        [
            {
                "surface_reelle_bati": args.surface,
                "nombre_pieces_principales": args.rooms,
                "latitude": args.latitude,
                "longitude": args.longitude,
                "adresse_numero": args.address_number,
                "nombre_lots": args.lots,
                "month": valuation_date.month,
                "day_of_year": valuation_date.dayofyear,
                "year": valuation_date.year,
                "months_since_2021": (valuation_date.year - 2021) * 12
                + valuation_date.month
                - 1,
                "code_postal": str(args.postal_code),
            }
        ]
    )


def select_calibration_errors(
    calibration: pd.DataFrame, postal_code: str, estimate: float
) -> tuple[pd.DataFrame, str]:
    """Prefer errors from the same market and price band when sample size allows."""
    same_postcode = calibration["code_postal"].eq(postal_code)
    similar_price = calibration["predicted_value"].between(estimate * 0.5, estimate * 2.0)
    local_and_similar = calibration.loc[same_postcode & similar_price]
    if len(local_and_similar) >= 120:
        return local_and_similar, "same arrondissement and comparable predicted-price band"

    local = calibration.loc[same_postcode]
    if len(local) >= 120:
        return local, "same arrondissement"

    comparable = calibration.loc[similar_price]
    if len(comparable) >= 250:
        return comparable, "Paris-wide comparable predicted-price band"

    return calibration, "all Paris calibration sales"


def interval_coverage(
    calibration_ratios: np.ndarray, evaluation_ratios: np.ndarray
) -> list[dict[str, float]]:
    results = []
    for nominal in (0.50, 0.80, 0.90, 0.95):
        tail = (1 - nominal) / 2
        lower, upper = np.quantile(calibration_ratios, [tail, 1 - tail])
        observed = np.mean(
            (evaluation_ratios >= lower) & (evaluation_ratios <= upper)
        )
        results.append(
            {
                "nominal": nominal,
                "observed": float(observed),
                "lower_ratio": float(lower),
                "upper_ratio": float(upper),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=PROJECT_ROOT / "models/valuation_model.joblib"
    )
    parser.add_argument(
        "--calibration-predictions",
        type=Path,
        default=PROJECT_ROOT / "reports/validation_predictions.csv",
        help="Out-of-sample predictions used to construct intervals (default: 2024 validation)",
    )
    parser.add_argument(
        "--test-predictions",
        type=Path,
        default=PROJECT_ROOT / "reports/test_predictions.csv",
        help="Later predictions used only to test coverage (default: 2025 test)",
    )
    parser.add_argument("--surface", type=float, required=True)
    parser.add_argument("--rooms", type=float, required=True)
    parser.add_argument("--postal-code", required=True)
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    parser.add_argument("--address-number", type=float, default=None)
    parser.add_argument("--lots", type=float, default=1)
    parser.add_argument("--date", default="2025-12-31")
    parser.add_argument(
        "--asking-price",
        type=float,
        default=None,
        help="Optional asking price used to estimate P(market value >= asking price)",
    )
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "visuals")
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports/property_probability.json",
    )
    args = parser.parse_args()
    validate_arguments(args, parser)

    bundle = joblib.load(args.model)
    features = property_features(args)
    estimate = max(float(bundle["model"].predict(features[bundle["features"]])[0]), 1)

    calibration = pd.read_csv(
        args.calibration_predictions, dtype={"code_postal": "string"}
    )
    evaluation = pd.read_csv(args.test_predictions, dtype={"code_postal": "string"})
    for frame in (calibration, evaluation):
        frame["date_mutation"] = pd.to_datetime(frame["date_mutation"])
        frame.sort_values(["date_mutation", "id_mutation"], inplace=True)
        frame["price_ratio"] = (
            frame["valeur_fonciere"] / frame["predicted_value"].clip(lower=1)
        )

    selected, selection_description = select_calibration_errors(
        calibration, str(args.postal_code), estimate
    )
    ratios = selected["price_ratio"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(ratios) < 100:
        raise RuntimeError("Too few valid calibration errors to build a distribution.")

    rng = np.random.default_rng(42)
    simulated_prices = estimate * rng.choice(ratios, size=args.samples, replace=True)
    quantile_levels = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    quantile_values = np.quantile(simulated_prices, quantile_levels)
    quantiles = {
        f"p{int(level * 100):02d}_eur": round(float(value), 2)
        for level, value in zip(quantile_levels, quantile_values)
    }

    global_calibration_ratios = calibration["price_ratio"].to_numpy()
    evaluation_ratios = evaluation["price_ratio"].to_numpy()
    coverage = interval_coverage(global_calibration_ratios, evaluation_ratios)

    probability_summary = {
        "point_estimate_eur": round(estimate, 2),
        "distribution_method": "Bootstrap of out-of-sample actual/predicted price ratios",
        "calibration_period": "2024",
        "coverage_test_period": "2025",
        "calibration_selection": selection_description,
        "calibration_observations": int(len(ratios)),
        "simulation_samples": int(args.samples),
        "quantiles": quantiles,
        "probability_within_10_percent_of_point_estimate": round(
            float(np.mean(np.abs(simulated_prices / estimate - 1) <= 0.10)), 4
        ),
        "probability_within_20_percent_of_point_estimate": round(
            float(np.mean(np.abs(simulated_prices / estimate - 1) <= 0.20)), 4
        ),
        "interval_calibration_test": coverage,
        "warning": "This empirical uncertainty distribution is not a guaranteed appraisal or a Bayesian posterior.",
    }
    if args.asking_price is not None:
        probability_summary["asking_price_eur"] = round(args.asking_price, 2)
        probability_summary["probability_value_at_least_asking_price"] = round(
            float(np.mean(simulated_prices >= args.asking_price)), 4
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    # Property-specific empirical probability distribution.
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    plot_min, plot_max = np.quantile(simulated_prices, [0.005, 0.995])
    plotted = simulated_prices[
        (simulated_prices >= plot_min) & (simulated_prices <= plot_max)
    ]
    ax.hist(plotted / 1_000, bins=55, density=True, color=BLUE, alpha=0.82)
    p05, p50, p95 = np.quantile(simulated_prices, [0.05, 0.50, 0.95])
    ax.axvspan(p05 / 1_000, p95 / 1_000, color=PALE, alpha=0.55, label="Empirical 90% interval")
    ax.axvline(estimate / 1_000, color=CORAL, linewidth=2.5, label="Model point estimate")
    ax.axvline(p50 / 1_000, color=NAVY, linestyle="--", linewidth=2, label="Distribution median")
    if args.asking_price is not None:
        ax.axvline(
            args.asking_price / 1_000,
            color="#7A5195",
            linestyle=":",
            linewidth=2.5,
            label="Asking price",
        )
    ax.set(
        title="Empirical Probability Distribution of Property Value",
        xlabel="Possible transaction value (€000)",
        ylabel="Probability density",
    )
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False)
    ax.text(
        0.98,
        0.95,
        f"Point estimate: €{estimate:,.0f}\n90% interval: €{p05:,.0f} – €{p95:,.0f}\nCalibration n={len(ratios):,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=NAVY,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor=PALE),
    )
    distribution_path = args.output_dir / "property_probability_distribution.png"
    fig.savefig(distribution_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Reliability check: promised interval coverage versus later observed coverage.
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    nominal = np.array([item["nominal"] for item in coverage]) * 100
    observed = np.array([item["observed"] for item in coverage]) * 100
    ax.plot([45, 100], [45, 100], color=GRAY, linestyle="--", label="Perfect calibration")
    ax.plot(nominal, observed, color=BLUE, marker="o", linewidth=2.5, markersize=8)
    for x, y in zip(nominal, observed):
        ax.annotate(f"{y:.1f}%", (x, y), xytext=(6, 6), textcoords="offset points", color=NAVY)
    ax.set(
        xlim=(45, 100),
        ylim=(45, 100),
        xlabel="Promised interval coverage",
        ylabel="Observed coverage on later unseen sales",
        title="Are the Probability Intervals Calibrated?",
    )
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    calibration_path = args.output_dir / "uncertainty_calibration.png"
    fig.savefig(calibration_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    args.report.write_text(
        json.dumps(probability_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(probability_summary, indent=2, ensure_ascii=False))
    print(f"Saved: {distribution_path}")
    print(f"Saved: {calibration_path}")
    print(f"Saved: {args.report}")


if __name__ == "__main__":
    main()
