"""Build leakage-safe nationwide France features for the Phase 4 AVM.

The builder reads the annual full Geo-DVF archives in bounded chunks, retains
ordinary single-apartment sales, adds geography features that also work in the
overseas departments, and computes comparable sales independently within each
department.  Only transactions at least 90 days old may inform another sale.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import h3
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from paris_avm.artifacts import write_json_atomic, write_parquet_atomic
from paris_avm.geography import (
    KNOWN_DVF_GAPS,
    REGION_NAMES,
    department_from_commune,
    normalize_department,
    region_from_department,
)
from paris_avm.paths import PROJECT_ROOT


EARTH_RADIUS_METERS = 6_371_008.8
MIN_TRANSACTION_VALUE_EUR = 10_000.0
MAX_TRANSACTION_VALUE_EUR = 10_000_000.0
LOCAL_WINDOWS = (
    (250, 365),
    (500, 180),
    (500, 365),
    (500, 730),
    (1_000, 180),
    (1_000, 365),
    (1_000, 730),
)
MAX_RADIUS_METERS = max(radius for radius, _ in LOCAL_WINDOWS)
MAX_WINDOW_DAYS = max(window for _, window in LOCAL_WINDOWS)

RAW_COLUMNS = (
    "id_mutation",
    "date_mutation",
    "nature_mutation",
    "valeur_fonciere",
    "adresse_numero",
    "adresse_suffixe",
    "adresse_nom_voie",
    "adresse_code_voie",
    "code_postal",
    "code_commune",
    "code_departement",
    "id_parcelle",
    "code_type_local",
    "surface_reelle_bati",
    "nombre_pieces_principales",
    "nombre_lots",
    "surface_terrain",
    "longitude",
    "latitude",
)

COMPARABLE_FEATURES = tuple(
    feature
    for radius, window in LOCAL_WINDOWS
    for feature in (
        f"local_{window}d_{radius}m_count",
        f"local_{window}d_{radius}m_median_ppm2",
    )
) + (
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

PHASE4_NUMERIC_FEATURES = (
    "surface_reelle_bati",
    "nombre_pieces_principales",
    "adresse_numero",
    "nombre_lots",
    "log_surface",
    "rooms_per_10m2",
    "month",
    "day_of_year",
    "year",
    "months_since_2021",
    "latitude",
    "longitude",
    "geo_x",
    "geo_y",
    "geo_z",
) + COMPARABLE_FEATURES

PHASE4_CATEGORICAL_FEATURES = (
    "code_region",
    "code_departement",
    "code_commune",
    "code_postal",
    "h3_r7",
    "h3_r8",
    "h3_r9",
)
PHASE4_MODEL_FEATURES = PHASE4_NUMERIC_FEATURES + PHASE4_CATEGORICAL_FEATURES


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Collapse a raw frame containing complete mutations to eligible sales."""
    counts = {"raw_rows": int(len(raw)), "raw_mutations": int(raw["id_mutation"].nunique())}
    raw = raw.copy()
    raw["date_mutation"] = pd.to_datetime(raw["date_mutation"], errors="coerce")
    for column in (
        "valeur_fonciere",
        "adresse_numero",
        "surface_reelle_bati",
        "nombre_pieces_principales",
        "nombre_lots",
        "surface_terrain",
        "longitude",
        "latitude",
    ):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    sale_rows = raw.loc[raw["nature_mutation"].eq("Vente")]
    summary = sale_rows.groupby("id_mutation", sort=False)["code_type_local"].agg(
        apartment_count=lambda values: int(values.eq("2").sum()),
        incompatible_count=lambda values: int(values.isin(("1", "4")).sum()),
    )
    eligible = summary.index[
        summary["apartment_count"].eq(1) & summary["incompatible_count"].eq(0)
    ]
    apartments = sale_rows.loc[
        sale_rows["id_mutation"].isin(eligible) & sale_rows["code_type_local"].eq("2")
    ].copy()
    counts["single_apartment_sale_rows"] = int(len(apartments))

    apartments = apartments.dropna(
        subset=(
            "date_mutation",
            "valeur_fonciere",
            "surface_reelle_bati",
            "code_commune",
            "latitude",
            "longitude",
        )
    )
    for column in ("code_postal", "code_commune", "code_departement"):
        apartments[column] = apartments[column].astype("string").str.strip().str.upper()
    inferred = apartments["code_commune"].map(department_from_commune)
    apartments["code_departement"] = apartments["code_departement"].map(normalize_department)
    apartments["code_departement"] = apartments["code_departement"].fillna(inferred)
    apartments["code_region"] = apartments["code_departement"].map(region_from_department)
    apartments["code_postal"] = (
        apartments["code_postal"]
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
        .fillna("00000")
    )
    apartments["price_per_m2"] = apartments["valeur_fonciere"] / apartments["surface_reelle_bati"]

    apartments = apartments.loc[
        apartments["surface_reelle_bati"].between(9, 300)
        & apartments["valeur_fonciere"].between(
            MIN_TRANSACTION_VALUE_EUR, MAX_TRANSACTION_VALUE_EUR
        )
        & apartments["price_per_m2"].between(100, 30_000)
        & apartments["latitude"].between(-90, 90)
        & apartments["longitude"].between(-180, 180)
        & apartments["code_region"].notna()
    ].copy()
    apartments["month"] = apartments["date_mutation"].dt.month
    apartments["day_of_year"] = apartments["date_mutation"].dt.dayofyear
    apartments["year"] = apartments["date_mutation"].dt.year
    apartments["months_since_2021"] = (
        (apartments["year"] - 2021) * 12 + apartments["month"] - 1
    )
    counts["clean_model_rows"] = int(len(apartments))
    counts["removed_during_quality_filters"] = (
        counts["single_apartment_sale_rows"] - counts["clean_model_rows"]
    )
    return apartments, counts


def load_and_clean_file(path: Path, *, read_chunk_rows: int = 500_000) -> tuple[pd.DataFrame, dict[str, int]]:
    """Read one sorted Geo-DVF archive without splitting a mutation across chunks."""
    if read_chunk_rows < 1:
        raise ValueError("read_chunk_rows must be positive")
    dtype = {
        column: "string"
        for column in (
            "id_mutation",
            "adresse_suffixe",
            "adresse_nom_voie",
            "adresse_code_voie",
            "code_postal",
            "code_commune",
            "code_departement",
            "id_parcelle",
            "code_type_local",
        )
    }
    frames: list[pd.DataFrame] = []
    totals = {
        "raw_rows": 0,
        "raw_mutations": 0,
        "single_apartment_sale_rows": 0,
        "clean_model_rows": 0,
        "removed_during_quality_filters": 0,
    }
    carry: pd.DataFrame | None = None
    for chunk in pd.read_csv(
        path,
        usecols=list(RAW_COLUMNS),
        dtype=dtype,
        low_memory=False,
        chunksize=read_chunk_rows,
        compression="infer",
    ):
        if carry is not None:
            chunk = pd.concat((carry, chunk), ignore_index=True)
        last_id = chunk["id_mutation"].iloc[-1]
        boundary = chunk["id_mutation"].eq(last_id)
        carry = chunk.loc[boundary].copy()
        complete = chunk.loc[~boundary]
        if complete.empty:
            continue
        cleaned, counts = _clean_transactions(complete)
        frames.append(cleaned)
        for key in totals:
            totals[key] += counts[key]
    if carry is not None and not carry.empty:
        cleaned, counts = _clean_transactions(carry)
        frames.append(cleaned)
        for key in totals:
            totals[key] += counts[key]
    if not frames:
        raise RuntimeError(f"No eligible apartment sales found in {path}")
    output = pd.concat(frames, ignore_index=True)
    output["source_file"] = path.name
    return output, totals


def load_and_clean_many(
    paths: list[Path], *, read_chunk_rows: int = 500_000
) -> tuple[pd.DataFrame, dict[str, object]]:
    frames: list[pd.DataFrame] = []
    per_file: dict[str, dict[str, int]] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing national DVF file: {path}")
        frame, counts = load_and_clean_file(path, read_chunk_rows=read_chunk_rows)
        frames.append(frame)
        per_file[path.name] = counts
        print(
            f"Cleaned {path.name}: {counts['clean_model_rows']:,} eligible rows "
            f"from {counts['raw_rows']:,} source rows",
            flush=True,
        )
    data = pd.concat(frames, ignore_index=True).sort_values(
        ["date_mutation", "id_mutation"]
    ).reset_index(drop=True)
    if data["id_mutation"].duplicated().any():
        examples = data.loc[data["id_mutation"].duplicated(False), "id_mutation"].head().tolist()
        raise RuntimeError(f"Mutation IDs overlap across annual files: {examples}")
    aggregate: dict[str, object] = {
        key: int(sum(item[key] for item in per_file.values()))
        for key in next(iter(per_file.values()))
    }
    aggregate["per_file"] = per_file
    return data, aggregate


def add_geography_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add global location, property, identity, and H3 features."""
    output = data.copy()
    latitude_radians = np.radians(output["latitude"].to_numpy(float))
    longitude_radians = np.radians(output["longitude"].to_numpy(float))
    cosine_latitude = np.cos(latitude_radians)
    output["geo_x"] = cosine_latitude * np.cos(longitude_radians)
    output["geo_y"] = cosine_latitude * np.sin(longitude_radians)
    output["geo_z"] = np.sin(latitude_radians)
    output["log_surface"] = np.log1p(output["surface_reelle_bati"])
    output["rooms_per_10m2"] = (
        output["nombre_pieces_principales"] / output["surface_reelle_bati"].clip(lower=1) * 10
    )
    output["street_id"] = output["code_commune"].fillna("unknown") + "_" + output[
        "adresse_code_voie"
    ].fillna("unknown")
    number = output["adresse_numero"].fillna(-1).round().astype("Int64").astype("string")
    suffix = output["adresse_suffixe"].fillna("").str.lower().str.strip()
    output["address_id"] = output["street_id"] + "_" + number + "_" + suffix
    output["parcel_id"] = output["id_parcelle"].fillna("unknown")
    coordinates = list(zip(output["latitude"], output["longitude"]))
    for resolution in (7, 8, 9):
        output[f"h3_r{resolution}"] = [
            h3.latlng_to_cell(float(latitude), float(longitude), resolution)
            for latitude, longitude in coordinates
        ]
    return output


def _department_comparables(
    group: pd.DataFrame, availability_lag_days: int, query_chunk_rows: int
) -> dict[str, np.ndarray]:
    coordinates = np.radians(group[["latitude", "longitude"]].to_numpy(float))
    dates = group["date_mutation"].to_numpy(dtype="datetime64[D]").astype(np.int64)
    prices = group["price_per_m2"].to_numpy(float)
    surfaces = group["surface_reelle_bati"].to_numpy(float)
    rooms = group["nombre_pieces_principales"].to_numpy(float)
    tree = BallTree(coordinates, metric="haversine")
    arrays: dict[str, np.ndarray] = {}
    for radius, window in LOCAL_WINDOWS:
        prefix = f"local_{window}d_{radius}m"
        arrays[f"{prefix}_count"] = np.zeros(len(group), dtype=np.int32)
        arrays[f"{prefix}_median_ppm2"] = np.full(len(group), np.nan)
    arrays["similar_comparable_count"] = np.zeros(len(group), dtype=np.int32)
    for name in (
        "local_365d_500m_mad_ppm2",
        "weighted_comp_ppm2",
        "similar_weighted_comp_ppm2",
        "effective_comparable_count",
        "nearest_prior_sale_m",
        "median_comparable_age_days",
    ):
        arrays[name] = np.full(len(group), np.nan)

    radius_radians = MAX_RADIUS_METERS / EARTH_RADIUS_METERS
    for start in range(0, len(group), query_chunk_rows):
        stop = min(start + query_chunk_rows, len(group))
        neighbors, angular_distances = tree.query_radius(
            coordinates[start:stop], r=radius_radians, return_distance=True
        )
        for offset, (indices, angular) in enumerate(zip(neighbors, angular_distances)):
            row = start + offset
            distances = angular * EARTH_RADIUS_METERS
            ages = dates[row] - dates[indices]
            eligible = (ages >= availability_lag_days) & (ages <= MAX_WINDOW_DAYS)
            if not eligible.any():
                continue
            indices, distances, ages = indices[eligible], distances[eligible], ages[eligible]
            candidate_prices = prices[indices]
            arrays["nearest_prior_sale_m"][row] = distances.min()
            arrays["median_comparable_age_days"][row] = np.median(ages)
            for radius, window in LOCAL_WINDOWS:
                selected = (distances <= radius) & (ages <= window)
                prefix = f"local_{window}d_{radius}m"
                arrays[f"{prefix}_count"][row] = selected.sum()
                if selected.any():
                    arrays[f"{prefix}_median_ppm2"][row] = np.median(candidate_prices[selected])
            dispersion = (distances <= 500) & (ages <= 365)
            if dispersion.any():
                values = candidate_prices[dispersion]
                arrays["local_365d_500m_mad_ppm2"][row] = np.median(
                    np.abs(values - np.median(values))
                )
            comparable = (distances <= 1_000) & (ages <= 730)
            if comparable.any():
                comp_indices = indices[comparable]
                comp_distances = distances[comparable]
                comp_ages = ages[comparable]
                comp_prices = candidate_prices[comparable]
                weights = np.exp(-comp_distances / 350) * np.exp(
                    -(comp_ages - availability_lag_days) / 240
                )
                arrays["weighted_comp_ppm2"][row] = np.average(comp_prices, weights=weights)
                arrays["effective_comparable_count"][row] = weights.sum() ** 2 / np.square(weights).sum()
                ratios = surfaces[comp_indices] / surfaces[row]
                similar = (ratios >= 0.7) & (ratios <= 1.3)
                if np.isfinite(rooms[row]):
                    candidate_rooms = rooms[comp_indices]
                    similar &= (~np.isfinite(candidate_rooms)) | (np.abs(candidate_rooms - rooms[row]) <= 1)
                if similar.any():
                    similar_weights = weights[similar] * np.exp(-np.abs(np.log(ratios[similar])) / 0.25)
                    arrays["similar_comparable_count"][row] = similar.sum()
                    arrays["similar_weighted_comp_ppm2"][row] = np.average(
                        comp_prices[similar], weights=similar_weights
                    )
    arrays["local_momentum_180v365_500m"] = (
        arrays["local_180d_500m_median_ppm2"] / arrays["local_365d_500m_median_ppm2"] - 1
    )
    arrays["local_momentum_180v365_1000m"] = (
        arrays["local_180d_1000m_median_ppm2"] / arrays["local_365d_1000m_median_ppm2"] - 1
    )
    return arrays


def add_comparable_features(
    data: pd.DataFrame, *, availability_lag_days: int = 90, query_chunk_rows: int = 2_048
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compute geodesic, point-in-time comparable features by department."""
    if availability_lag_days < 1 or query_chunk_rows < 1:
        raise ValueError("availability_lag_days and query_chunk_rows must be positive")
    output = data.reset_index(drop=True).copy()
    combined: dict[str, np.ndarray] = {}
    started = perf_counter()
    for position, (department, indices) in enumerate(output.groupby("code_departement", sort=True).indices.items(), 1):
        group = output.iloc[indices].reset_index(drop=True)
        arrays = _department_comparables(group, availability_lag_days, query_chunk_rows)
        for name, values in arrays.items():
            if name not in combined:
                combined[name] = np.full(len(output), np.nan)
            combined[name][indices] = values
        print(
            f"Comparable features: department {department} ({position}/"
            f"{output['code_departement'].nunique()}) in {perf_counter() - started:.1f}s",
            flush=True,
        )
    integer_count_features = {
        name
        for name in combined
        if name.startswith("local_") and name.endswith("_count")
    } | {"similar_comparable_count"}
    for name, values in combined.items():
        if name in integer_count_features:
            output[name] = values.astype(np.int32)
        else:
            output[name] = values
    quality = {
        "availability_lag_days": availability_lag_days,
        "maximum_radius_meters": MAX_RADIUS_METERS,
        "maximum_history_days": MAX_WINDOW_DAYS,
        "distance_method": "BallTree haversine, processed independently by department",
        "coverage_by_region": {
            str(region): {
                "name": REGION_NAMES.get(str(region), "Unknown"),
                "rows": int(len(group)),
                "weighted_comparable_coverage": round(float(group["weighted_comp_ppm2"].notna().mean()), 4),
            }
            for region, group in output.groupby("code_region")
        },
    }
    return output, quality


def validate_phase4_table(data: pd.DataFrame) -> None:
    missing = sorted(
        {"id_mutation", "date_mutation", "valeur_fonciere", *PHASE4_MODEL_FEATURES}.difference(data.columns)
    )
    if missing:
        raise ValueError("Phase 4 is missing required columns: " + ", ".join(missing))
    if data["id_mutation"].isna().any() or data["id_mutation"].duplicated().any():
        raise ValueError("Phase 4 mutation identifiers must be unique and non-null")
    if not data["date_mutation"].is_monotonic_increasing:
        raise ValueError("Phase 4 rows must be chronologically ordered")
    if not data["surface_reelle_bati"].between(9, 300).all():
        raise ValueError("Phase 4 contains property surfaces outside [9, 300] m2")
    if not data["latitude"].between(-90, 90).all() or not data["longitude"].between(-180, 180).all():
        raise ValueError("Phase 4 contains invalid WGS84 coordinates")
    if data[list(PHASE4_CATEGORICAL_FEATURES)].isna().any(axis=None):
        raise ValueError("Phase 4 has null territorial categorical features")
    forbidden = {"valeur_fonciere", "price_per_m2"}.intersection(PHASE4_MODEL_FEATURES)
    if forbidden:
        raise ValueError(f"Phase 4 model contract contains target leakage: {sorted(forbidden)}")


def build_phase4_features(
    data_files: list[Path],
    output: Path,
    quality_report: Path,
    *,
    availability_lag_days: int = 90,
    read_chunk_rows: int = 500_000,
    query_chunk_rows: int = 2_048,
) -> tuple[pd.DataFrame, dict[str, object]]:
    started = perf_counter()
    data, counts = load_and_clean_many(data_files, read_chunk_rows=read_chunk_rows)
    data = add_geography_features(data)
    data, comparable_quality = add_comparable_features(
        data,
        availability_lag_days=availability_lag_days,
        query_chunk_rows=query_chunk_rows,
    )
    validate_phase4_table(data)
    write_parquet_atomic(data, output)
    observed_departments = sorted(data["code_departement"].unique().tolist())
    report: dict[str, object] = {
        "schema_version": 1,
        "phase": 4,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "All ordinary apartment sales available in nationwide Geo-DVF",
        "known_upstream_gaps": list(KNOWN_DVF_GAPS),
        "observed_departments": observed_departments,
        "observed_regions": sorted(data["code_region"].unique().tolist()),
        "gold_table": str(output.resolve()),
        "output_sha256": file_sha256(output),
        "row_count": int(len(data)),
        "date_min": data["date_mutation"].min().date().isoformat(),
        "date_max": data["date_mutation"].max().date().isoformat(),
        "source_files": [
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in data_files
        ],
        "source_cleaning_counts": counts,
        "comparable_feature_contract": comparable_quality,
        "numeric_features": list(PHASE4_NUMERIC_FEATURES),
        "categorical_features": list(PHASE4_CATEGORICAL_FEATURES),
        "excluded_target_derived_columns": ["price_per_m2"],
        "runtime_seconds": round(perf_counter() - started, 3),
    }
    write_json_atomic(quality_report, report)
    return data, report


def default_data_files() -> list[Path]:
    return [PROJECT_ROOT / f"data/raw/dvf_france/france_{year}.csv.gz" for year in range(2021, 2026)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", type=Path, default=default_data_files())
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/gold/phase4_france_sale_features.parquet")
    parser.add_argument("--quality-report", type=Path, default=PROJECT_ROOT / "reports/phase4/feature_quality.json")
    parser.add_argument("--availability-lag-days", type=int, default=90)
    parser.add_argument("--read-chunk-rows", type=int, default=500_000)
    parser.add_argument("--query-chunk-rows", type=int, default=2_048)
    args = parser.parse_args()
    if min(args.availability_lag_days, args.read_chunk_rows, args.query_chunk_rows) < 1:
        parser.error("lag and chunk sizes must be positive")
    _, report = build_phase4_features(
        args.data,
        args.output,
        args.quality_report,
        availability_lag_days=args.availability_lag_days,
        read_chunk_rows=args.read_chunk_rows,
        query_chunk_rows=args.query_chunk_rows,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
