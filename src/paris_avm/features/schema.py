"""Feature contracts shared by preprocessing, training and inference.

The lists in this module distinguish model inputs from audit columns and from
values derived from the current transaction target.  In particular,
``price_per_m2`` is useful for quality filtering and for constructing strictly
historical comparable-sale features, but it must never be fitted as a feature
for the transaction from which it was calculated.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


TARGET_COLUMN = "valeur_fonciere"

PROPERTY_NUMERIC_FEATURES = (
    "surface_reelle_bati",
    "nombre_pieces_principales",
    "adresse_numero",
    "nombre_lots",
)
PROPERTY_ENGINEERED_NUMERIC_FEATURES = (
    "log_surface",
    "rooms_per_10m2",
    "surface_terrain",
)
TEMPORAL_NUMERIC_FEATURES = (
    "month",
    "day_of_year",
    "year",
    "months_since_2021",
)
COORDINATE_NUMERIC_FEATURES = ("latitude", "longitude")
SPATIAL_NUMERIC_FEATURES = (
    "x_l93",
    "y_l93",
    "distance_paris_center_m",
    "bearing_sin",
    "bearing_cos",
)
BASE_CATEGORICAL_FEATURES = ("code_postal",)
SPATIAL_CATEGORICAL_FEATURES = ("h3_r8", "h3_r9")

COMPARABLE_NUMERIC_FEATURES = (
    "local_365d_250m_count",
    "local_365d_250m_median_ppm2",
    "local_180d_500m_count",
    "local_180d_500m_median_ppm2",
    "local_365d_500m_count",
    "local_365d_500m_median_ppm2",
    "local_730d_500m_count",
    "local_730d_500m_median_ppm2",
    "local_180d_1000m_count",
    "local_180d_1000m_median_ppm2",
    "local_365d_1000m_count",
    "local_365d_1000m_median_ppm2",
    "local_730d_1000m_count",
    "local_730d_1000m_median_ppm2",
    "local_365d_500m_mad_ppm2",
    "weighted_comp_ppm2",
    "similar_weighted_comp_ppm2",
    "similar_comparable_count",
    "effective_comparable_count",
    "nearest_prior_sale_m",
    "median_comparable_age_days",
    "local_momentum_180v365_500m",
    "local_momentum_180v365_1000m",
)

BUILDING_NUMERIC_FEATURES = (
    "ban_match_distance_m",
    "ban_match_confidence",
    "bdnb_match_confidence",
    "bdnb_address_candidate_count",
    "bdnb_parcel_candidate_count",
    "building_level_count",
    "building_year",
    "building_dwelling_count",
    "building_height_m",
    "building_footprint_m2",
    "building_parking_lot_count",
    "building_total_lot_count",
    "building_dwelling_count_rnc",
    "building_age_at_sale",
    "building_year_missing",
)
BUILDING_CATEGORICAL_FEATURES = (
    "usage_niveau_1_txt",
    "mat_mur_txt",
    "mat_toit_txt",
    "building_state",
    "building_period_rnc",
)
DPE_NUMERIC_FEATURES = (
    "dpe_available_at_sale",
    "dpe_age_days",
    "dpe_surface_difference_ratio",
    "dpe_match_confidence",
    "annee_construction_dpe",
    "nombre_niveau_logement",
    "nombre_niveau_immeuble",
    "surface_habitable_immeuble",
    "surface_habitable_logement",
    "conso_5_usages_ep_m2",
    "emission_ges_5_usages_m2",
)
DPE_CATEGORICAL_FEATURES = (
    "classe_bilan_dpe",
    "classe_emission_ges",
    "type_installation_chauffage",
    "type_energie_chauffage",
)
CONTEXT_NUMERIC_FEATURES = (
    "distance_metro_m",
    "distance_rer_m",
    "distance_train_m",
    "distance_tram_m",
    "rail_station_mode_count_1000m",
    "rail_line_count_1000m",
    "distance_school_m",
    "school_count_1000m",
    "distance_green_space_centroid_m",
    "green_space_count_500m",
    "green_area_500m2",
    "distance_shop_service_m",
    "shop_service_count_500m",
    "shop_service_count_1000m",
    "noise_road_lden_db",
    "noise_rail_ratp_lden_db",
    "noise_rail_sncf_lden_db",
    "noise_transport_lden_max_db",
)
EXTERNAL_IDENTITY_CATEGORICAL_FEATURES = (
    "ban_address_id",
    "bdnb_building_id",
    "bdnb_match_method",
)

PHASE2_FEATURE_GROUPS: Mapping[str, tuple[str, ...]] = {
    "property": PROPERTY_NUMERIC_FEATURES + PROPERTY_ENGINEERED_NUMERIC_FEATURES,
    "time": TEMPORAL_NUMERIC_FEATURES,
    "coordinates": COORDINATE_NUMERIC_FEATURES,
    "spatial": SPATIAL_NUMERIC_FEATURES + SPATIAL_CATEGORICAL_FEATURES,
    "historical_comparables": COMPARABLE_NUMERIC_FEATURES,
    "categorical": BASE_CATEGORICAL_FEATURES,
}
PHASE3_FEATURE_GROUPS: Mapping[str, tuple[str, ...]] = {
    "building": BUILDING_NUMERIC_FEATURES + BUILDING_CATEGORICAL_FEATURES,
    "dpe_point_in_time": DPE_NUMERIC_FEATURES + DPE_CATEGORICAL_FEATURES,
    "neighborhood_context": CONTEXT_NUMERIC_FEATURES,
    "external_identity": EXTERNAL_IDENTITY_CATEGORICAL_FEATURES,
}

PHASE2_NUMERIC_FEATURES = (
    "surface_reelle_bati",
    "nombre_pieces_principales",
    "latitude",
    "longitude",
    "adresse_numero",
    "nombre_lots",
) + (
    TEMPORAL_NUMERIC_FEATURES
    + PROPERTY_ENGINEERED_NUMERIC_FEATURES
    + SPATIAL_NUMERIC_FEATURES
    + COMPARABLE_NUMERIC_FEATURES
)
PHASE2_CATEGORICAL_FEATURES = (
    BASE_CATEGORICAL_FEATURES + SPATIAL_CATEGORICAL_FEATURES
)
PHASE2_MODEL_FEATURES = PHASE2_NUMERIC_FEATURES + PHASE2_CATEGORICAL_FEATURES
PHASE3_NUMERIC_FEATURES = (
    PHASE2_NUMERIC_FEATURES
    + DPE_NUMERIC_FEATURES
    + BUILDING_NUMERIC_FEATURES
    + CONTEXT_NUMERIC_FEATURES
)
PHASE3_CATEGORICAL_FEATURES = (
    PHASE2_CATEGORICAL_FEATURES
    + DPE_CATEGORICAL_FEATURES
    + BUILDING_CATEGORICAL_FEATURES
    + EXTERNAL_IDENTITY_CATEGORICAL_FEATURES
)
PHASE3_MODEL_FEATURES = PHASE3_NUMERIC_FEATURES + PHASE3_CATEGORICAL_FEATURES

AUDIT_COLUMNS = (
    "id_mutation",
    "date_mutation",
    "id_parcelle",
    "street_id",
    "address_id",
    "parcel_id",
    "h3_r10",
    "source_file",
)
CURRENT_TRANSACTION_DERIVED_COLUMNS = ("price_per_m2",)


class FeatureContractError(ValueError):
    """Raised when a preprocessed table violates its published contract."""


def _flatten(groups: Mapping[str, Iterable[str]]) -> tuple[str, ...]:
    return tuple(feature for features in groups.values() for feature in features)


def contract_summary() -> dict[str, object]:
    """Return a JSON-serializable description of all preprocessing features."""
    return {
        "target": TARGET_COLUMN,
        "phase2_feature_count": len(PHASE2_MODEL_FEATURES),
        "phase3_feature_count": len(PHASE3_MODEL_FEATURES),
        "phase2_groups": {
            name: list(features) for name, features in PHASE2_FEATURE_GROUPS.items()
        },
        "phase3_groups": {
            name: list(features) for name, features in PHASE3_FEATURE_GROUPS.items()
        },
        "audit_columns": list(AUDIT_COLUMNS),
        "excluded_target_derived_columns": list(CURRENT_TRANSACTION_DERIVED_COLUMNS),
    }


def _require_columns(
    data: pd.DataFrame, groups: Mapping[str, Iterable[str]], stage: str
) -> None:
    missing = sorted(set(_flatten(groups)).difference(data.columns))
    if missing:
        raise FeatureContractError(
            f"{stage} is missing required feature columns: {', '.join(missing)}"
        )


def _validate_identity_and_order(data: pd.DataFrame, stage: str) -> None:
    required = {"id_mutation", "date_mutation", TARGET_COLUMN}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise FeatureContractError(
            f"{stage} is missing required identity columns: {', '.join(missing)}"
        )
    if data["id_mutation"].isna().any() or data["id_mutation"].duplicated().any():
        raise FeatureContractError(f"{stage} mutation identifiers must be unique and non-null")
    if data["date_mutation"].isna().any():
        raise FeatureContractError(f"{stage} mutation dates must be non-null")
    if not data["date_mutation"].is_monotonic_increasing:
        raise FeatureContractError(f"{stage} rows must be chronologically ordered")


def validate_phase2_table(data: pd.DataFrame) -> None:
    """Validate core, temporal, spatial and historical-comparable features."""
    _validate_identity_and_order(data, "Phase 2")
    _require_columns(data, PHASE2_FEATURE_GROUPS, "Phase 2")

    non_nullable = (
        ("surface_reelle_bati", "log_surface")
        + TEMPORAL_NUMERIC_FEATURES
        + COORDINATE_NUMERIC_FEATURES
        + SPATIAL_NUMERIC_FEATURES
        + BASE_CATEGORICAL_FEATURES
        + SPATIAL_CATEGORICAL_FEATURES
    )
    missing_values = [column for column in non_nullable if data[column].isna().any()]
    if missing_values:
        raise FeatureContractError(
            "Phase 2 has null values in required features: " + ", ".join(missing_values)
        )

    if not data["surface_reelle_bati"].between(9, 300).all():
        raise FeatureContractError("Phase 2 contains property surfaces outside [9, 300] m2")
    if not data[TARGET_COLUMN].between(50_000, 10_000_000).all():
        raise FeatureContractError("Phase 2 contains sale values outside the accepted range")
    if not data["latitude"].between(48.80, 48.92).all() or not data[
        "longitude"
    ].between(2.20, 2.48).all():
        raise FeatureContractError("Phase 2 contains coordinates outside Paris bounds")

    count_columns = [
        column for column in COMPARABLE_NUMERIC_FEATURES if column.endswith("_count")
    ]
    if any((data[column].dropna() < 0).any() for column in count_columns):
        raise FeatureContractError("Phase 2 comparable counts cannot be negative")
    finite_columns = (
        ("surface_reelle_bati", "log_surface")
        + TEMPORAL_NUMERIC_FEATURES
        + COORDINATE_NUMERIC_FEATURES
        + SPATIAL_NUMERIC_FEATURES
    )
    if not np.isfinite(data[list(finite_columns)].to_numpy(dtype=float)).all():
        raise FeatureContractError("Phase 2 contains non-finite core numeric features")


def validate_phase3_table(data: pd.DataFrame) -> None:
    """Validate enrichment completeness and point-in-time DPE safety."""
    validate_phase2_table(data)
    _require_columns(data, PHASE3_FEATURE_GROUPS, "Phase 3")
    if "date_etablissement_dpe" not in data.columns:
        raise FeatureContractError("Phase 3 is missing date_etablissement_dpe")
    leakage = data["date_etablissement_dpe"].notna() & (
        data["date_etablissement_dpe"] > data["date_mutation"]
    )
    if leakage.any():
        raise FeatureContractError("Phase 3 contains DPE observations dated after the sale")


def validate_contract_definition() -> None:
    """Fail fast if a feature is duplicated or target-derived leakage is fitted."""
    for stage, features in (
        ("Phase 2", PHASE2_MODEL_FEATURES),
        ("Phase 3", PHASE3_MODEL_FEATURES),
    ):
        duplicates = sorted(
            feature for feature in set(features) if features.count(feature) > 1
        )
        if duplicates:
            raise FeatureContractError(
                f"{stage} contract contains duplicate features: {', '.join(duplicates)}"
            )
        forbidden = {TARGET_COLUMN, *CURRENT_TRANSACTION_DERIVED_COLUMNS}.intersection(
            features
        )
        if forbidden:
            raise FeatureContractError(
                f"{stage} contract contains target-derived features: {', '.join(sorted(forbidden))}"
            )


validate_contract_definition()
