"""Build a leakage-safe Phase 2 Gold feature table from annual Paris DVF files.

Comparable-sale features are calculated point in time. A transaction may use
only sales at least ``availability_lag_days`` older than its mutation date. The
default 90-day lag is a conservative proxy for DVF publication delay.
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
from pyproj import Transformer
from sklearn.neighbors import BallTree

from paris_avm.artifacts import write_json_atomic, write_parquet_atomic
from paris_avm.modeling.train_phase1 import DEFAULT_DATA_FILES, load_and_clean_many
from paris_avm.paths import PROJECT_ROOT


LOCAL_WINDOWS = [
    (250, 365),
    (500, 180),
    (500, 365),
    (500, 730),
    (1_000, 180),
    (1_000, 365),
    (1_000, 730),
]
MAX_RADIUS_METERS = max(radius for radius, _ in LOCAL_WINDOWS)
MAX_WINDOW_DAYS = max(window for _, window in LOCAL_WINDOWS)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_identity_and_spatial_features(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    x_l93, y_l93 = transformer.transform(
        output["longitude"].to_numpy(), output["latitude"].to_numpy()
    )
    output["x_l93"] = x_l93
    output["y_l93"] = y_l93

    paris_center_x, paris_center_y = transformer.transform(2.3499, 48.8530)
    delta_x = output["x_l93"] - paris_center_x
    delta_y = output["y_l93"] - paris_center_y
    output["distance_paris_center_m"] = np.hypot(delta_x, delta_y)
    output["bearing_sin"] = np.sin(np.arctan2(delta_y, delta_x))
    output["bearing_cos"] = np.cos(np.arctan2(delta_y, delta_x))
    output["log_surface"] = np.log1p(output["surface_reelle_bati"])
    output["rooms_per_10m2"] = (
        output["nombre_pieces_principales"]
        / output["surface_reelle_bati"].clip(lower=1)
        * 10
    )

    output["street_id"] = (
        output["code_commune"].fillna("unknown")
        + "_"
        + output["adresse_code_voie"].fillna("unknown")
    )
    normalized_number = output["adresse_numero"].fillna(-1).round().astype("Int64").astype("string")
    normalized_suffix = output["adresse_suffixe"].fillna("").str.lower().str.strip()
    output["address_id"] = (
        output["street_id"] + "_" + normalized_number + "_" + normalized_suffix
    )
    output["parcel_id"] = output["id_parcelle"].fillna("unknown")

    coordinates = zip(output["latitude"].to_numpy(), output["longitude"].to_numpy())
    coordinate_list = list(coordinates)
    for resolution in (8, 9, 10):
        output[f"h3_r{resolution}"] = [
            h3.latlng_to_cell(latitude, longitude, resolution)
            for latitude, longitude in coordinate_list
        ]
    return output


def add_comparable_features(
    data: pd.DataFrame,
    availability_lag_days: int,
    chunk_size: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    output = data.copy()
    coordinates = output[["x_l93", "y_l93"]].to_numpy(dtype=float)
    dates = output["date_mutation"].to_numpy(dtype="datetime64[D]").astype(np.int64)
    prices_m2 = output["price_per_m2"].to_numpy(dtype=float)
    surfaces = output["surface_reelle_bati"].to_numpy(dtype=float)
    rooms = output["nombre_pieces_principales"].to_numpy(dtype=float)
    tree = BallTree(coordinates, metric="euclidean")

    feature_arrays: dict[str, np.ndarray] = {}
    for radius, window in LOCAL_WINDOWS:
        prefix = f"local_{window}d_{radius}m"
        feature_arrays[f"{prefix}_count"] = np.zeros(len(output), dtype=np.int32)
        feature_arrays[f"{prefix}_median_ppm2"] = np.full(len(output), np.nan)
    for name in [
        "local_365d_500m_mad_ppm2",
        "weighted_comp_ppm2",
        "similar_weighted_comp_ppm2",
        "similar_comparable_count",
        "effective_comparable_count",
        "nearest_prior_sale_m",
        "median_comparable_age_days",
    ]:
        feature_arrays[name] = np.full(len(output), np.nan)

    start_time = perf_counter()
    for chunk_start in range(0, len(output), chunk_size):
        chunk_stop = min(chunk_start + chunk_size, len(output))
        candidate_indices, candidate_distances = tree.query_radius(
            coordinates[chunk_start:chunk_stop],
            r=MAX_RADIUS_METERS,
            return_distance=True,
            sort_results=False,
        )
        for offset, (indices, distances) in enumerate(
            zip(candidate_indices, candidate_distances)
        ):
            row_index = chunk_start + offset
            age_days = dates[row_index] - dates[indices]
            eligible = (
                (age_days >= availability_lag_days)
                & (age_days <= MAX_WINDOW_DAYS)
                & np.isfinite(prices_m2[indices])
            )
            if not eligible.any():
                continue
            indices = indices[eligible]
            distances = distances[eligible]
            age_days = age_days[eligible]
            candidate_ppm2 = prices_m2[indices]

            feature_arrays["nearest_prior_sale_m"][row_index] = float(distances.min())
            feature_arrays["median_comparable_age_days"][row_index] = float(
                np.median(age_days)
            )

            for radius, window in LOCAL_WINDOWS:
                selected = (distances <= radius) & (age_days <= window)
                prefix = f"local_{window}d_{radius}m"
                count = int(selected.sum())
                feature_arrays[f"{prefix}_count"][row_index] = count
                if count:
                    feature_arrays[f"{prefix}_median_ppm2"][row_index] = float(
                        np.median(candidate_ppm2[selected])
                    )

            dispersion_selected = (distances <= 500) & (age_days <= 365)
            if dispersion_selected.any():
                values = candidate_ppm2[dispersion_selected]
                median = np.median(values)
                feature_arrays["local_365d_500m_mad_ppm2"][row_index] = float(
                    np.median(np.abs(values - median))
                )

            comparable = (distances <= 1_000) & (age_days <= 730)
            if comparable.any():
                comparable_indices = indices[comparable]
                comparable_distances = distances[comparable]
                comparable_ages = age_days[comparable]
                comparable_ppm2 = candidate_ppm2[comparable]
                weights = np.exp(-comparable_distances / 350.0) * np.exp(
                    -(comparable_ages - availability_lag_days) / 240.0
                )
                feature_arrays["weighted_comp_ppm2"][row_index] = float(
                    np.average(comparable_ppm2, weights=weights)
                )
                feature_arrays["effective_comparable_count"][row_index] = float(
                    weights.sum() ** 2 / np.square(weights).sum()
                )

                surface_ratio = surfaces[comparable_indices] / surfaces[row_index]
                similar = (surface_ratio >= 0.70) & (surface_ratio <= 1.30)
                if np.isfinite(rooms[row_index]):
                    candidate_rooms = rooms[comparable_indices]
                    similar &= (~np.isfinite(candidate_rooms)) | (
                        np.abs(candidate_rooms - rooms[row_index]) <= 1
                    )
                if similar.any():
                    surface_similarity = np.exp(
                        -np.abs(np.log(surface_ratio[similar])) / 0.25
                    )
                    similar_weights = weights[similar] * surface_similarity
                    feature_arrays["similar_comparable_count"][row_index] = int(
                        similar.sum()
                    )
                    feature_arrays["similar_weighted_comp_ppm2"][row_index] = float(
                        np.average(comparable_ppm2[similar], weights=similar_weights)
                    )

        completed = chunk_stop
        if completed % (chunk_size * 20) == 0 or completed == len(output):
            print(
                f"Comparable features: {completed:,}/{len(output):,} rows "
                f"({perf_counter() - start_time:.1f}s)",
                flush=True,
            )

    for name, values in feature_arrays.items():
        output[name] = values
    output["local_momentum_180v365_500m"] = (
        output["local_180d_500m_median_ppm2"]
        / output["local_365d_500m_median_ppm2"]
        - 1
    )
    output["local_momentum_180v365_1000m"] = (
        output["local_180d_1000m_median_ppm2"]
        / output["local_365d_1000m_median_ppm2"]
        - 1
    )

    count_columns = [column for column in feature_arrays if column.endswith("_count")]
    quality = {
        "availability_lag_days": availability_lag_days,
        "maximum_radius_meters": MAX_RADIUS_METERS,
        "maximum_history_days": MAX_WINDOW_DAYS,
        "minimum_observed_comparable_age_days": int(availability_lag_days),
        "coverage_by_year": {},
    }
    for year, group in output.groupby("year"):
        quality["coverage_by_year"][str(year)] = {
            "rows": int(len(group)),
            "weighted_comparable_coverage": round(
                float(group["weighted_comp_ppm2"].notna().mean()), 4
            ),
            "similar_comparable_coverage": round(
                float(group["similar_weighted_comp_ppm2"].notna().mean()), 4
            ),
            "median_count_365d_500m": round(
                float(group["local_365d_500m_count"].median()), 2
            ),
            "median_count_365d_1000m": round(
                float(group["local_365d_1000m_count"].median()), 2
            ),
        }
    quality["count_feature_ranges"] = {
        column: {
            "minimum": int(output[column].min()),
            "maximum": int(output[column].max()),
        }
        for column in count_columns
    }
    return output, quality


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", type=Path, default=DEFAULT_DATA_FILES)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/gold/phase2_sale_features.parquet",
    )
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=PROJECT_ROOT / "reports/phase2/feature_quality.json",
    )
    parser.add_argument("--availability-lag-days", type=int, default=90)
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()
    if args.availability_lag_days < 1:
        parser.error("--availability-lag-days must be positive")

    started = perf_counter()
    data, counts = load_and_clean_many(args.data)
    data = add_identity_and_spatial_features(data)
    data, comparable_quality = add_comparable_features(
        data,
        availability_lag_days=args.availability_lag_days,
        chunk_size=args.chunk_size,
    )
    if data["id_mutation"].duplicated().any():
        raise RuntimeError("Gold feature table contains duplicate mutation identifiers.")
    if not data["date_mutation"].is_monotonic_increasing:
        raise RuntimeError("Gold feature table is not chronologically ordered.")

    write_parquet_atomic(data, args.output)
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gold_table": str(args.output.resolve()),
        "output_sha256": file_sha256(args.output),
        "row_count": int(len(data)),
        "date_min": data["date_mutation"].min().date().isoformat(),
        "date_max": data["date_mutation"].max().date().isoformat(),
        "crs_storage": "EPSG:4326",
        "crs_metric_features": "EPSG:2154",
        "h3_resolutions": [8, 9, 10],
        "source_files": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in args.data
        ],
        "source_cleaning_counts": counts,
        "comparable_feature_contract": comparable_quality,
        "feature_columns": data.columns.tolist(),
        "runtime_seconds": round(perf_counter() - started, 3),
    }
    write_json_atomic(args.quality_report, report)
    print(json.dumps(comparable_quality, indent=2, ensure_ascii=False))
    print(f"Saved Gold feature table: {args.output}")
    print(f"Saved quality report: {args.quality_report}")


if __name__ == "__main__":
    main()
