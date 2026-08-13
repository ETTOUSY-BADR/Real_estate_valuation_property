"""Estimate a Paris apartment's 2025-equivalent transaction value."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import joblib
import pandas as pd

from paris_avm.paths import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=PROJECT_ROOT / "models/valuation_model.joblib"
    )
    parser.add_argument("--surface", type=float, required=True, help="Built area in square metres")
    parser.add_argument("--rooms", type=float, required=True, help="Number of main rooms")
    parser.add_argument("--postal-code", required=True, help="Paris postcode, e.g. 75011")
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    parser.add_argument("--address-number", type=float, default=None)
    parser.add_argument("--lots", type=float, default=1)
    parser.add_argument(
        "--date",
        default="2025-12-31",
        help="Valuation date within 2025 (YYYY-MM-DD; default: 2025-12-31)",
    )
    args = parser.parse_args()

    if not 9 <= args.surface <= 300:
        parser.error("--surface must be between 9 and 300 m² for this model")
    if not re.fullmatch(r"750(?:0[1-9]|1[0-9]|20)", str(args.postal_code)):
        parser.error("--postal-code must be between 75001 and 75020")
    if not 0 <= args.rooms <= 15:
        parser.error("--rooms must be between 0 and 15")
    if not 48.80 <= args.latitude <= 48.92 or not 2.20 <= args.longitude <= 2.48:
        parser.error("coordinates must be located within Paris")
    valuation_date = pd.to_datetime(args.date, errors="raise")
    if valuation_date.year != 2025:
        parser.error("--date must be in 2025; this model cannot forecast other years")

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
    estimate = max(float(bundle["model"].predict(row[bundle["features"]])[0]), 0)
    interval = bundle.get("prediction_interval")
    print(f"Estimated transaction value: €{estimate:,.0f}")
    print(f"Estimated value per m²:      €{estimate / args.surface:,.0f}/m²")
    if interval:
        lower = estimate * interval["lower_price_ratio"]
        upper = estimate * interval["upper_price_ratio"]
        print(f"Empirical 90% price range:   €{lower:,.0f} – €{upper:,.0f}")
    print("Interpretation: 2025-equivalent value for an ordinary Paris apartment.")
    print("The price range reflects historical model errors, not a guaranteed appraisal range.")


if __name__ == "__main__":
    main()
