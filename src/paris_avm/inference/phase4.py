"""Estimate an apartment anywhere covered by the nationwide Phase 4 model."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from paris_avm.features.phase4 import (
    EARTH_RADIUS_METERS,
    LOCAL_WINDOWS,
    MAX_WINDOW_DAYS,
    MAX_RADIUS_METERS,
    add_geography_features,
)
from paris_avm.geography import REGION_NAMES, department_from_commune, region_from_department
from paris_avm.modeling.benchmark_phase4 import safe_inverse_log_prediction
from paris_avm.paths import PROJECT_ROOT


DEFAULT_MODEL = PROJECT_ROOT / "models/phase4/france_catboost.joblib"
DEFAULT_GOLD = PROJECT_ROOT / "data/gold/phase4_france_sale_features.parquet"


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive whole number")
    return number


def valid_date(value: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("must be a valid date in YYYY-MM-DD format") from error


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", required=True, type=positive_float)
    parser.add_argument("--rooms", required=True, type=positive_float)
    parser.add_argument("--postal-code", required=True)
    parser.add_argument("--commune-code", required=True, help="Five-character INSEE commune code")
    parser.add_argument("--address-number", type=positive_int, default=1)
    parser.add_argument("--street-code", default="UNKNOWN")
    parser.add_argument("--address-suffix", default="")
    parser.add_argument("--latitude", required=True, type=float)
    parser.add_argument("--longitude", required=True, type=float)
    parser.add_argument("--lots", type=positive_int, default=1)
    parser.add_argument("--date", required=True, type=valid_date)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gold-table", type=Path, default=DEFAULT_GOLD)
    args = parser.parse_args(arguments)
    args.postal_code = str(args.postal_code).strip().upper()
    args.commune_code = str(args.commune_code).strip().upper()
    if not re.fullmatch(r"[0-9A-Z]{5}", args.commune_code):
        parser.error("--commune-code must be a five-character INSEE code")
    if not re.fullmatch(r"\d{5}", args.postal_code):
        parser.error("--postal-code must contain five digits")
    department = department_from_commune(args.commune_code)
    if department is None or region_from_department(department) is None:
        parser.error("--commune-code does not identify a supported French department")
    if not -90 <= args.latitude <= 90 or not -180 <= args.longitude <= 180:
        parser.error("coordinates must be valid WGS84 latitude/longitude")
    if not 9 <= args.surface <= 300:
        parser.error("--surface must be between 9 and 300 m2")
    return args


def comparable_features(
    history: pd.DataFrame,
    *,
    latitude: float,
    longitude: float,
    surface: float,
    rooms: float,
    valuation_date: pd.Timestamp,
    availability_lag_days: int = 90,
) -> dict[str, float | int]:
    """Calculate the Phase 4 comparable contract for one proposed valuation."""
    result: dict[str, float | int] = {}
    for radius, window in LOCAL_WINDOWS:
        prefix = f"local_{window}d_{radius}m"
        result[f"{prefix}_count"] = 0
        result[f"{prefix}_median_ppm2"] = np.nan
    result.update(
        {
            "local_365d_500m_mad_ppm2": np.nan,
            "weighted_comp_ppm2": np.nan,
            "similar_weighted_comp_ppm2": np.nan,
            "similar_comparable_count": 0,
            "effective_comparable_count": np.nan,
            "nearest_prior_sale_m": np.nan,
            "median_comparable_age_days": np.nan,
        }
    )
    if history.empty:
        result["local_momentum_180v365_500m"] = np.nan
        result["local_momentum_180v365_1000m"] = np.nan
        return result
    coordinates = np.radians(history[["latitude", "longitude"]].to_numpy(float))
    tree = BallTree(coordinates, metric="haversine")
    target = np.radians([[latitude, longitude]])
    indices, angular = tree.query_radius(
        target, r=MAX_RADIUS_METERS / EARTH_RADIUS_METERS, return_distance=True
    )
    indices, distances = indices[0], angular[0] * EARTH_RADIUS_METERS
    ages = (
        valuation_date.to_datetime64().astype("datetime64[D]").astype(np.int64)
        - history["date_mutation"].to_numpy(dtype="datetime64[D]").astype(np.int64)[indices]
    )
    eligible = (ages >= availability_lag_days) & (ages <= MAX_WINDOW_DAYS)
    if not eligible.any():
        result["local_momentum_180v365_500m"] = np.nan
        result["local_momentum_180v365_1000m"] = np.nan
        return result
    indices, distances, ages = indices[eligible], distances[eligible], ages[eligible]
    prices = history["price_per_m2"].to_numpy(float)[indices]
    result["nearest_prior_sale_m"] = float(distances.min())
    result["median_comparable_age_days"] = float(np.median(ages))
    for radius, window in LOCAL_WINDOWS:
        selected = (distances <= radius) & (ages <= window)
        prefix = f"local_{window}d_{radius}m"
        result[f"{prefix}_count"] = int(selected.sum())
        if selected.any():
            result[f"{prefix}_median_ppm2"] = float(np.median(prices[selected]))
    dispersion = (distances <= 500) & (ages <= 365)
    if dispersion.any():
        values = prices[dispersion]
        result["local_365d_500m_mad_ppm2"] = float(np.median(np.abs(values - np.median(values))))
    weights = np.exp(-distances / 350) * np.exp(-(ages - availability_lag_days) / 240)
    result["weighted_comp_ppm2"] = float(np.average(prices, weights=weights))
    result["effective_comparable_count"] = float(weights.sum() ** 2 / np.square(weights).sum())
    surfaces = history["surface_reelle_bati"].to_numpy(float)[indices]
    candidate_rooms = history["nombre_pieces_principales"].to_numpy(float)[indices]
    ratios = surfaces / surface
    similar = (ratios >= 0.7) & (ratios <= 1.3)
    similar &= (~np.isfinite(candidate_rooms)) | (np.abs(candidate_rooms - rooms) <= 1)
    result["similar_comparable_count"] = int(similar.sum())
    if similar.any():
        similar_weights = weights[similar] * np.exp(-np.abs(np.log(ratios[similar])) / 0.25)
        result["similar_weighted_comp_ppm2"] = float(np.average(prices[similar], weights=similar_weights))
    for radius in (500, 1_000):
        short = result[f"local_180d_{radius}m_median_ppm2"]
        long = result[f"local_365d_{radius}m_median_ppm2"]
        result[f"local_momentum_180v365_{radius}m"] = (
            float(short / long - 1) if np.isfinite(short) and np.isfinite(long) and long else np.nan
        )
    return result


def main() -> None:
    args = parse_args()
    bundle = joblib.load(args.model)
    history = pd.read_parquet(args.gold_table)
    history["date_mutation"] = pd.to_datetime(history["date_mutation"])
    department = department_from_commune(args.commune_code)
    assert department is not None
    region = region_from_department(department)
    if department not in set(history["code_departement"].astype(str)):
        raise SystemExit(
            f"Department {department} has no Phase 4 training data. It may be an upstream DVF gap."
        )
    local_history = history.loc[history["code_departement"].astype(str).eq(department)]
    row = pd.DataFrame(
        [
            {
                "id_mutation": "inference",
                "date_mutation": args.date,
                "valeur_fonciere": np.nan,
                "surface_reelle_bati": args.surface,
                "nombre_pieces_principales": args.rooms,
                "adresse_numero": args.address_number,
                "adresse_suffixe": args.address_suffix,
                "adresse_code_voie": args.street_code,
                "nombre_lots": args.lots,
                "surface_terrain": np.nan,
                "latitude": args.latitude,
                "longitude": args.longitude,
                "code_postal": args.postal_code,
                "code_commune": args.commune_code,
                "code_departement": department,
                "code_region": region,
                "id_parcelle": pd.NA,
                "month": args.date.month,
                "day_of_year": args.date.dayofyear,
                "year": args.date.year,
                "months_since_2021": (args.date.year - 2021) * 12 + args.date.month - 1,
            }
        ]
    )
    row = add_geography_features(row)
    for name, value in comparable_features(
        local_history,
        latitude=args.latitude,
        longitude=args.longitude,
        surface=args.surface,
        rooms=args.rooms,
        valuation_date=args.date,
    ).items():
        row[name] = value
    if "department_market_cluster" in bundle["features"]:
        mapping = bundle.get("department_market_cluster_mapping", {})
        row["department_market_cluster"] = mapping.get(department, "market_unknown")
    for column in bundle["categorical_features"]:
        row[column] = row[column].astype("string").fillna("__MISSING__")
    prediction_values, clipped = safe_inverse_log_prediction(
        bundle["model"].predict(row[bundle["features"]])
    )
    prediction = float(prediction_values[0])
    interval = bundle["prediction_interval"]
    regional = interval.get("regional_calibration", {}).get(str(region), interval)
    lower = prediction * float(regional["lower_price_ratio"])
    upper = prediction * float(regional["upper_price_ratio"])
    print(f"Region:                       {REGION_NAMES.get(str(region), region)}")
    print(f"Department:                   {department}")
    print(f"Historical comparables:       {int(row.at[0, 'local_730d_1000m_count']):,}")
    print(f"Phase 4 estimated value:      EUR {prediction:,.0f}")
    print(f"Estimated value per m2:       EUR {prediction / args.surface:,.0f}/m2")
    print(f"Empirical 90% interval:       EUR {lower:,.0f} - {upper:,.0f}")
    if clipped:
        print("Warning: raw model output was clipped to the registered transaction-value bounds.")
    if args.date.year != 2025:
        print("Warning: frozen out-of-time evaluation is for 2025; this date is an extrapolation.")


if __name__ == "__main__":
    main()
