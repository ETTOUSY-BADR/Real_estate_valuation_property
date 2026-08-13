"""Estimate a 2025 Paris apartment value with the Phase 2 comparable correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from pyproj import Transformer

from paris_avm.paths import PROJECT_ROOT

DEFAULT_MODEL = PROJECT_ROOT / "models/benchmark/lightgbm.joblib"
DEFAULT_GOLD = PROJECT_ROOT / "data/gold/phase2_sale_features.parquet"
DEFAULT_RESULTS = PROJECT_ROOT / "reports/phase2/phase2_results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", type=float, required=True)
    parser.add_argument("--rooms", type=float, required=True)
    parser.add_argument("--postal-code", required=True)
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    parser.add_argument("--address-number", type=float, default=None)
    parser.add_argument("--lots", type=float, default=1)
    parser.add_argument("--date", default="2025-12-31", help="YYYY-MM-DD in 2025")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gold-table", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    if not 9 <= args.surface <= 300:
        parser.error("--surface must be between 9 and 300 square metres")
    if not re.fullmatch(r"750(?:0[1-9]|1[0-9]|20)", str(args.postal_code)):
        parser.error("--postal-code must be between 75001 and 75020")
    if not 0 <= args.rooms <= 15:
        parser.error("--rooms must be between 0 and 15")
    if not 48.80 <= args.latitude <= 48.92 or not 2.20 <= args.longitude <= 2.48:
        parser.error("coordinates must be located within Paris")
    args.date = pd.to_datetime(args.date, errors="raise")
    if args.date.year != 2025:
        parser.error("this evaluated Phase 2 estimator only supports dates in 2025")
    return args


def selected_correction(results_path: Path) -> tuple[str, float]:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    row = next(
        result
        for result in payload["results"]
        if result["model_key"] == "phase2_comparable_correction"
    )
    config = json.loads(row["selected_configuration"])
    return config["comparable_feature"], float(config["phase1_weight"])


def historical_local_median(
    gold_path: Path, longitude: float, latitude: float, valuation_date: pd.Timestamp
) -> tuple[float | None, int]:
    history = pd.read_parquet(
        gold_path,
        columns=["date_mutation", "x_l93", "y_l93", "price_per_m2"],
    )
    history["date_mutation"] = pd.to_datetime(history["date_mutation"])
    oldest = valuation_date - pd.Timedelta(days=365)
    newest = valuation_date - pd.Timedelta(days=90)
    history = history.loc[history["date_mutation"].between(oldest, newest, inclusive="both")]
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True).transform(
        longitude, latitude
    )
    distance = np.hypot(history["x_l93"].to_numpy() - x, history["y_l93"].to_numpy() - y)
    local = history.loc[distance <= 500, "price_per_m2"].dropna()
    if local.empty:
        return None, 0
    return float(local.median()), int(len(local))


def main() -> None:
    args = parse_args()
    bundle = joblib.load(args.model)
    row = pd.DataFrame(
        [
            {
                "surface_reelle_bati": args.surface,
                "nombre_pieces_principales": args.rooms,
                "latitude": args.latitude,
                "longitude": args.longitude,
                "adresse_numero": args.address_number,
                "nombre_lots": args.lots,
                "month": args.date.month,
                "day_of_year": args.date.dayofyear,
                "year": args.date.year,
                "months_since_2021": (args.date.year - 2021) * 12 + args.date.month - 1,
                "code_postal": str(args.postal_code),
            }
        ]
    )
    phase1_estimate = max(float(bundle["model"].predict(row[bundle["features"]])[0]), 0)
    feature, phase1_weight = selected_correction(args.results)
    if feature != "local_365d_500m_median_ppm2":
        raise RuntimeError(f"Unsupported selected comparable feature: {feature}")
    local_median, comparable_count = historical_local_median(
        args.gold_table, args.longitude, args.latitude, args.date
    )

    print(f"Phase 1 model estimate:       EUR {phase1_estimate:,.0f}")
    if local_median is None:
        corrected = phase1_estimate
        print("Historical local comparable: unavailable; using Phase 1 estimate")
    else:
        comparable_estimate = local_median * args.surface
        corrected = phase1_weight * phase1_estimate + (1 - phase1_weight) * comparable_estimate
        print(f"Local median (500m/365d):    EUR {local_median:,.0f}/m2")
        print(f"Eligible prior transactions: {comparable_count:,}")
        print(f"Comparable-only estimate:    EUR {comparable_estimate:,.0f}")
    print(f"Phase 2 corrected estimate:  EUR {corrected:,.0f}")
    print(f"Corrected value per m2:      EUR {corrected / args.surface:,.0f}/m2")
    print(f"Blend: {phase1_weight:.0%} Phase 1 + {1 - phase1_weight:.0%} local market evidence")
    print("Only sales 90-365 days before the valuation date are eligible comparables.")


if __name__ == "__main__":
    main()
