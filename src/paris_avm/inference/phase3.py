"""Estimate a Paris apartment with the selected Phase 3 CatBoost model.

The command resolves an exact normalized DVF/BAN address already represented in
the Phase 3 Gold table. Static building/context fields come from that canonical
building; comparable-sale features and dated DPE fields use only information
available on or before the requested valuation date.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import re

import h3
import joblib
import numpy as np
import pandas as pd
from pyproj import Transformer

from paris_avm.modeling.benchmark_phase3 import catboost_frame
from paris_avm.features.phase2 import LOCAL_WINDOWS
from paris_avm.paths import PROJECT_ROOT


DEFAULT_MODEL = PROJECT_ROOT / "models/phase3/phase3_selected_model.joblib"
DEFAULT_GOLD = PROJECT_ROOT / "data/gold/phase3_sale_features.parquet"
MAX_ADDRESS_COORDINATE_OFFSET_M = 250.0
ADDRESS_SUFFIX_ALIASES = {"bis": "b", "ter": "t", "quater": "q"}


def normalize_address_suffix(value: object) -> str:
    suffix = str(value).strip().lower()
    return ADDRESS_SUFFIX_ALIASES.get(suffix, suffix)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", type=float, required=True)
    parser.add_argument("--rooms", type=float, required=True)
    parser.add_argument("--postal-code", required=True)
    parser.add_argument("--commune-code", required=True, help="Paris code, e.g. 75111")
    parser.add_argument("--street-code", required=True, help="Four-character DVF/FANTOIR road code")
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    parser.add_argument("--address-number", type=float, required=True)
    parser.add_argument(
        "--address-suffix",
        default="",
        help="Repetition index, e.g. B/bis, T/ter or Q/quater",
    )
    parser.add_argument("--lots", type=float, default=1)
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gold-table", type=Path, default=DEFAULT_GOLD)
    args = parser.parse_args(argv)
    if not 9 <= args.surface <= 300:
        parser.error("--surface must be between 9 and 300 square metres")
    if not re.fullmatch(r"750(?:0[1-9]|1[0-9]|20)", str(args.postal_code)):
        parser.error("--postal-code must be between 75001 and 75020")
    if not re.fullmatch(r"751(?:0[1-9]|1[0-9]|20)", str(args.commune_code)):
        parser.error("--commune-code must be between 75101 and 75120")
    if str(args.postal_code)[-2:] != str(args.commune_code)[-2:]:
        parser.error(
            "--postal-code and --commune-code must identify the same arrondissement"
        )
    if not re.fullmatch(r"[0-9A-Z]{4}", str(args.street_code).upper()):
        parser.error("--street-code must contain four letters/digits")
    if (
        not np.isfinite(args.address_number)
        or args.address_number <= 0
        or not float(args.address_number).is_integer()
    ):
        parser.error("--address-number must be a positive whole number")
    args.address_number = int(args.address_number)
    if not 0 <= args.rooms <= 15:
        parser.error("--rooms must be between 0 and 15")
    if not 48.80 <= args.latitude <= 48.92 or not 2.20 <= args.longitude <= 2.48:
        parser.error("coordinates must be located within Paris")
    try:
        args.date = pd.to_datetime(args.date, format="%Y-%m-%d", errors="raise")
    except (TypeError, ValueError):
        parser.error("--date must be a valid date in YYYY-MM-DD format")
    if args.date.year != 2025:
        parser.error("--date must be in 2025; this model is only evaluated for 2025")
    args.address_suffix = normalize_address_suffix(args.address_suffix)
    return args


def normalized_address_id(args: argparse.Namespace) -> str:
    suffix = normalize_address_suffix(args.address_suffix)
    return f"{args.commune_code}_{args.street_code.upper()}_{args.address_number}_{suffix}"


def validate_coordinate_consistency(
    args: argparse.Namespace, template: pd.Series
) -> float:
    canonical_x = pd.to_numeric(
        pd.Series([template.get("ban_x")]), errors="coerce"
    ).iloc[0]
    canonical_y = pd.to_numeric(
        pd.Series([template.get("ban_y")]), errors="coerce"
    ).iloc[0]
    if pd.isna(canonical_x) or pd.isna(canonical_y):
        canonical_x = pd.to_numeric(
            pd.Series([template.get("x_l93")]), errors="coerce"
        ).iloc[0]
        canonical_y = pd.to_numeric(
            pd.Series([template.get("y_l93")]), errors="coerce"
        ).iloc[0]
    if pd.isna(canonical_x) or pd.isna(canonical_y):
        raise SystemExit(
            "Canonical address coordinates are unavailable in the Gold table."
        )

    supplied_x, supplied_y = Transformer.from_crs(
        4326, 2154, always_xy=True
    ).transform(args.longitude, args.latitude)
    offset = float(np.hypot(supplied_x - canonical_x, supplied_y - canonical_y))
    if offset > MAX_ADDRESS_COORDINATE_OFFSET_M:
        raise SystemExit(
            f"Provided coordinates are {offset:,.0f} m from the resolved address; "
            "check --latitude/--longitude or the address identity fields."
        )
    return offset


def add_base_and_comparable_features(
    row: pd.DataFrame, history: pd.DataFrame, valuation_date: pd.Timestamp
) -> pd.DataFrame:
    transformer = Transformer.from_crs(4326, 2154, always_xy=True)
    x, y = transformer.transform(float(row.at[0, "longitude"]), float(row.at[0, "latitude"]))
    row["x_l93"] = x
    row["y_l93"] = y
    center_x, center_y = transformer.transform(2.3499, 48.8530)
    dx, dy = x - center_x, y - center_y
    row["distance_paris_center_m"] = np.hypot(dx, dy)
    row["bearing_sin"] = np.sin(np.arctan2(dy, dx))
    row["bearing_cos"] = np.cos(np.arctan2(dy, dx))
    surface = float(row.at[0, "surface_reelle_bati"])
    rooms = float(row.at[0, "nombre_pieces_principales"])
    row["log_surface"] = np.log1p(surface)
    row["rooms_per_10m2"] = rooms / surface * 10
    for resolution in (8, 9, 10):
        row[f"h3_r{resolution}"] = h3.latlng_to_cell(
            float(row.at[0, "latitude"]), float(row.at[0, "longitude"]), resolution
        )

    dates = pd.to_datetime(history["date_mutation"])
    ages = (valuation_date - dates).dt.days.to_numpy()
    xy = history[["x_l93", "y_l93"]].to_numpy(dtype=float)
    distances = np.hypot(xy[:, 0] - x, xy[:, 1] - y)
    ppm2 = history["price_per_m2"].to_numpy(dtype=float)
    eligible = (ages >= 90) & (ages <= 730) & (distances <= 1_000) & np.isfinite(ppm2)
    selected_distance = distances[eligible]
    selected_age = ages[eligible]
    selected_ppm2 = ppm2[eligible]
    selected_surface = history.loc[eligible, "surface_reelle_bati"].to_numpy(dtype=float)
    selected_rooms = history.loc[eligible, "nombre_pieces_principales"].to_numpy(dtype=float)

    for radius, window in LOCAL_WINDOWS:
        chosen = (selected_distance <= radius) & (selected_age <= window)
        prefix = f"local_{window}d_{radius}m"
        row[f"{prefix}_count"] = int(chosen.sum())
        row[f"{prefix}_median_ppm2"] = (
            float(np.median(selected_ppm2[chosen])) if chosen.any() else np.nan
        )
    dispersion = (selected_distance <= 500) & (selected_age <= 365)
    if dispersion.any():
        values = selected_ppm2[dispersion]
        row["local_365d_500m_mad_ppm2"] = float(
            np.median(np.abs(values - np.median(values)))
        )
    else:
        row["local_365d_500m_mad_ppm2"] = np.nan
    if eligible.any():
        weights = np.exp(-selected_distance / 350) * np.exp(-(selected_age - 90) / 240)
        row["weighted_comp_ppm2"] = float(np.average(selected_ppm2, weights=weights))
        row["effective_comparable_count"] = float(weights.sum() ** 2 / np.square(weights).sum())
        row["nearest_prior_sale_m"] = float(selected_distance.min())
        row["median_comparable_age_days"] = float(np.median(selected_age))
        ratio = selected_surface / float(row.at[0, "surface_reelle_bati"])
        similar = (ratio >= 0.7) & (ratio <= 1.3)
        similar &= (~np.isfinite(selected_rooms)) | (
            np.abs(selected_rooms - float(row.at[0, "nombre_pieces_principales"])) <= 1
        )
        row["similar_comparable_count"] = int(similar.sum())
        if similar.any():
            similar_weights = weights[similar] * np.exp(-np.abs(np.log(ratio[similar])) / 0.25)
            row["similar_weighted_comp_ppm2"] = float(
                np.average(selected_ppm2[similar], weights=similar_weights)
            )
        else:
            row["similar_weighted_comp_ppm2"] = np.nan
    else:
        for column in (
            "weighted_comp_ppm2",
            "effective_comparable_count",
            "nearest_prior_sale_m",
            "median_comparable_age_days",
            "similar_comparable_count",
            "similar_weighted_comp_ppm2",
        ):
            row[column] = np.nan
    row["local_momentum_180v365_500m"] = (
        row["local_180d_500m_median_ppm2"] / row["local_365d_500m_median_ppm2"] - 1
    )
    row["local_momentum_180v365_1000m"] = (
        row["local_180d_1000m_median_ppm2"] / row["local_365d_1000m_median_ppm2"] - 1
    )
    return row


def main() -> None:
    args = parse_args()
    bundle = joblib.load(args.model)
    gold = pd.read_parquet(args.gold_table)
    address_id = normalized_address_id(args)
    same_address = gold.loc[gold["address_id"].astype("string").eq(address_id)].copy()
    if same_address.empty:
        raise SystemExit(
            f"Exact normalized address {address_id!r} is absent from the Gold table. "
            "Check --commune-code/--street-code/--address-number or rebuild entity resolution."
        )
    # Static building/context comes from the most recent observed row at this
    # canonical address. Dated DPE fields are then rolled back below.
    template = same_address.sort_values("date_mutation").iloc[-1].copy()
    validate_coordinate_consistency(args, template)
    row = template.to_frame().T.reset_index(drop=True)
    row.loc[0, "surface_reelle_bati"] = args.surface
    row.loc[0, "nombre_pieces_principales"] = args.rooms
    row.loc[0, "code_postal"] = str(args.postal_code)
    row.loc[0, "code_commune"] = str(args.commune_code)
    row.loc[0, "adresse_code_voie"] = str(args.street_code).upper()
    row.loc[0, "adresse_numero"] = args.address_number
    row.loc[0, "adresse_suffixe"] = args.address_suffix
    row.loc[0, "latitude"] = args.latitude
    row.loc[0, "longitude"] = args.longitude
    row.loc[0, "nombre_lots"] = args.lots
    row.loc[0, "date_mutation"] = args.date
    row.loc[0, "month"] = args.date.month
    row.loc[0, "day_of_year"] = args.date.dayofyear
    row.loc[0, "year"] = args.date.year
    row.loc[0, "months_since_2021"] = (args.date.year - 2021) * 12 + args.date.month - 1

    dpe_columns = [
        "identifiant_dpe",
        "date_etablissement_dpe",
        "annee_construction_dpe",
        "nombre_niveau_logement",
        "nombre_niveau_immeuble",
        "surface_habitable_immeuble",
        "surface_habitable_logement",
        "conso_5_usages_ep_m2",
        "emission_ges_5_usages_m2",
        "classe_bilan_dpe",
        "classe_emission_ges",
        "type_installation_chauffage",
        "type_energie_chauffage",
    ]
    building_id = template["bdnb_building_id"]
    prior_building = gold.loc[
        gold["bdnb_building_id"].eq(building_id)
        & pd.to_datetime(gold["date_etablissement_dpe"], errors="coerce").le(args.date)
    ].sort_values("date_etablissement_dpe")
    if prior_building.empty:
        for column in dpe_columns:
            row.loc[0, column] = pd.NaT if column == "date_etablissement_dpe" else np.nan
        row.loc[0, "dpe_available_at_sale"] = 0
        row.loc[0, "dpe_age_days"] = np.nan
        row.loc[0, "dpe_match_confidence"] = 0.0
        row.loc[0, "dpe_surface_difference_ratio"] = np.nan
    else:
        dpe_source = prior_building.iloc[-1]
        for column in dpe_columns:
            row.loc[0, column] = dpe_source[column]
        row.loc[0, "dpe_available_at_sale"] = 1
        row.loc[0, "dpe_age_days"] = (
            args.date - pd.to_datetime(dpe_source["date_etablissement_dpe"])
        ).days
        dpe_surface = pd.to_numeric(
            pd.Series([dpe_source["surface_habitable_logement"]]), errors="coerce"
        ).iloc[0]
        difference = (
            abs(float(dpe_surface) - args.surface) / args.surface
            if pd.notna(dpe_surface)
            else np.nan
        )
        row.loc[0, "dpe_surface_difference_ratio"] = difference
        row.loc[0, "dpe_match_confidence"] = (
            1.0 if pd.notna(difference) and difference <= 0.15 else 0.6
        )

    row = add_base_and_comparable_features(row, gold, args.date)
    features = bundle["features"]
    categorical = bundle["categorical_features"]
    prediction_frame = catboost_frame(row, features, categorical)
    estimate = max(float(np.expm1(bundle["model"].predict(prediction_frame)[0])), 0)
    print(f"Resolved BAN address:        {row.at[0, 'ban_address_id']}")
    print(f"Resolved BDNB building:      {row.at[0, 'bdnb_building_id']}")
    print(f"Building match confidence:   {float(row.at[0, 'bdnb_match_confidence']):.0%}")
    print(f"DPE available by date:       {'yes' if row.at[0, 'dpe_available_at_sale'] else 'no'}")
    print(f"Historical comparables:      {int(row.at[0, 'local_365d_500m_count']):,}")
    print(f"Phase 3 estimated value:     EUR {estimate:,.0f}")
    print(f"Estimated value per m2:      EUR {estimate / args.surface:,.0f}/m2")
    print("Warning: building and neighborhood fields use the 2026 static source snapshot.")


if __name__ == "__main__":
    main()
