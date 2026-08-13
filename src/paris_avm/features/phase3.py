"""Build Phase 3 Paris sale features from BAN, BDNB, DPE and local context.

The input sale cohort is the leakage-safe Phase 2 Gold table. DPE attributes
are joined only when the diagnostic date is on or before the transaction date.
BAN/BDNB, transport, school, park and noise releases are retained as explicitly
flagged retrospective-static context because historical snapshots are not
available consistently for 2021--2025.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree

from paris_avm.data.acquire_phase3 import BAN_PATH, BDNB_ARCHIVE
from paris_avm.paths import PROJECT_ROOT


CONTEXT_DIR = PROJECT_ROOT / "data/bronze/phase3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_bdnb(
    archive: zipfile.ZipFile, table: str, usecols: list[str] | None = None
) -> pd.DataFrame:
    with archive.open(f"csv/{table}.csv") as handle:
        return pd.read_csv(
            handle,
            sep=";",
            usecols=usecols,
            low_memory=False,
            encoding="utf-8",
        )


def address_key(
    commune: pd.Series,
    road_code: pd.Series,
    number: pd.Series,
    suffix: pd.Series,
) -> pd.Series:
    numeric = pd.to_numeric(number, errors="coerce").astype("Int64")
    suffix_clean = suffix.astype("string").fillna("").str.lower().str.strip()
    return (
        commune.astype("string").str.strip()
        + "_"
        + road_code.astype("string").str.zfill(4)
        + "_"
        + numeric.astype("string").str.zfill(5)
        + suffix_clean.map(lambda value: f"_{value}" if value else "")
    )


def parse_point(value: object) -> tuple[float, float]:
    if pd.isna(value):
        return np.nan, np.nan
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", str(value))
    if len(numbers) < 2:
        return np.nan, np.nan
    # OpenDataSoft geo_point_2d is latitude, longitude.
    return float(numbers[1]), float(numbers[0])


def wkt_centroid(value: object) -> tuple[float, float]:
    if pd.isna(value):
        return np.nan, np.nan
    numbers = np.asarray(
        [float(item) for item in re.findall(r"[-+]?\d+(?:\.\d+)?", str(value))],
        dtype=float,
    )
    if len(numbers) < 2:
        return np.nan, np.nan
    pairs = numbers[: len(numbers) // 2 * 2].reshape(-1, 2)
    return float(np.mean(pairs[:, 0])), float(np.mean(pairs[:, 1]))


def add_xy_from_geopoint(data: pd.DataFrame, column: str) -> pd.DataFrame:
    points = np.asarray([parse_point(value) for value in data[column]], dtype=float)
    output = data.copy()
    output["point_lon"] = points[:, 0]
    output["point_lat"] = points[:, 1]
    # Fast vectorized WGS84 -> Lambert-93 transformation.
    from pyproj import Transformer

    transformer = Transformer.from_crs(4326, 2154, always_xy=True)
    output["point_x"], output["point_y"] = transformer.transform(
        output["point_lon"].to_numpy(), output["point_lat"].to_numpy()
    )
    return output


def nearest_distance(
    target_xy: np.ndarray, reference_xy: np.ndarray
) -> np.ndarray:
    valid = np.isfinite(reference_xy).all(axis=1)
    if not valid.any():
        return np.full(len(target_xy), np.nan)
    distances, _ = cKDTree(reference_xy[valid]).query(target_xy, k=1)
    return distances.astype(float)


def radius_count(
    target_xy: np.ndarray,
    reference_xy: np.ndarray,
    radius: float,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    valid = np.isfinite(reference_xy).all(axis=1)
    tree = cKDTree(reference_xy[valid])
    neighborhoods = tree.query_ball_point(target_xy, radius)
    if weights is None:
        return np.asarray([len(items) for items in neighborhoods], dtype=np.int32)
    valid_weights = np.asarray(weights)[valid]
    return np.asarray(
        [float(valid_weights[items].sum()) for items in neighborhoods], dtype=float
    )


def load_context_points(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, sep=";", low_memory=False)
    return add_xy_from_geopoint(data, "geo_point_2d")


def add_transport_features(data: pd.DataFrame) -> pd.DataFrame:
    stations = load_context_points(CONTEXT_DIR / "idfm_stations.csv")
    # Multiple rows represent lines serving one station; use unique mode/point
    # rows for distance and retain line multiplicity for accessibility counts.
    target_xy = data[["x_l93", "y_l93"]].to_numpy(dtype=float)
    station_xy = stations[["point_x", "point_y"]].to_numpy(dtype=float)
    output = data.copy()
    mode_normalized = stations["mode"].astype("string").str.upper()
    for label, modes in {
        "metro": {"METRO"},
        "rer": {"RER"},
        "train": {"TRAIN"},
        "tram": {"TRAMWAY"},
    }.items():
        selected = mode_normalized.isin(modes).to_numpy()
        output[f"distance_{label}_m"] = nearest_distance(target_xy, station_xy[selected])
    station_unique = stations.drop_duplicates(["id_ref_zdc", "mode"])
    unique_xy = station_unique[["point_x", "point_y"]].to_numpy(dtype=float)
    output["rail_station_mode_count_1000m"] = radius_count(
        target_xy, unique_xy, 1_000
    )
    output["rail_line_count_1000m"] = radius_count(target_xy, station_xy, 1_000)
    return output


def add_school_features(data: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for name in ("schools_elementary", "schools_maternelle", "schools_colleges"):
        frame = load_context_points(CONTEXT_DIR / f"{name}.csv")
        frame["school_family"] = name
        frames.append(frame)
    schools = pd.concat(frames, ignore_index=True)
    schools = schools.sort_values("annee_scol").drop_duplicates(
        ["school_family", "adresse"], keep="last"
    )
    target_xy = data[["x_l93", "y_l93"]].to_numpy(dtype=float)
    school_xy = schools[["point_x", "point_y"]].to_numpy(dtype=float)
    output = data.copy()
    output["distance_school_m"] = nearest_distance(target_xy, school_xy)
    output["school_count_1000m"] = radius_count(target_xy, school_xy, 1_000)
    return output


def add_green_features(data: pd.DataFrame) -> pd.DataFrame:
    parks = pd.read_csv(CONTEXT_DIR / "green_spaces.csv", sep=";", low_memory=False)
    parks = add_xy_from_geopoint(parks, "geom_x_y")
    parks = parks.drop_duplicates("nsq_espace_vert")
    park_xy = parks[["point_x", "point_y"]].to_numpy(dtype=float)
    target_xy = data[["x_l93", "y_l93"]].to_numpy(dtype=float)
    area = pd.to_numeric(parks["surface_totale_reelle"], errors="coerce").fillna(0)
    output = data.copy()
    output["distance_green_space_centroid_m"] = nearest_distance(target_xy, park_xy)
    output["green_space_count_500m"] = radius_count(target_xy, park_xy, 500)
    output["green_area_500m2"] = radius_count(
        target_xy, park_xy, 500, area.to_numpy(dtype=float)
    )
    return output


def add_noise_features(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    points = output[["x_l93", "y_l93"]].to_numpy(dtype=float)
    for family in ("road", "rail_ratp", "rail_sncf"):
        path = CONTEXT_DIR / f"noise_{family}_lden.gml"
        feature = np.full(len(output), np.nan)
        if not path.exists():
            output[f"noise_{family}_lden_db"] = feature
            continue
        import xml.etree.ElementTree as ET

        root = ET.parse(path).getroot()
        ns = {
            "gml": "http://www.opengis.net/gml/3.2",
            "wfs": "http://www.opengis.net/wfs/2.0",
            "ms": "http://mapserver.gis.umn.edu/mapserver",
        }
        for member in root.findall(".//wfs:member", ns):
            legend = member.find(".//ms:legende", ns)
            if legend is None or not legend.text:
                continue
            level = float(re.findall(r"\d+", legend.text)[0])
            for pos in member.findall(".//gml:exterior//gml:posList", ns):
                coords = np.fromstring(pos.text or "", sep=" ")
                if len(coords) < 6:
                    continue
                polygon = coords.reshape(-1, 2)
                xmin, ymin = polygon.min(axis=0)
                xmax, ymax = polygon.max(axis=0)
                candidate = (
                    (points[:, 0] >= xmin)
                    & (points[:, 0] <= xmax)
                    & (points[:, 1] >= ymin)
                    & (points[:, 1] <= ymax)
                )
                if candidate.any():
                    inside = MplPath(polygon).contains_points(points[candidate])
                    index = np.flatnonzero(candidate)[inside]
                    current = feature[index]
                    feature[index] = np.where(
                        np.isnan(current), level, np.maximum(current, level)
                    )
        output[f"noise_{family}_lden_db"] = feature
    output["noise_road_lden_missing"] = output["noise_road_lden_db"].isna().astype("int8")
    # Type-A maps begin at 55 dB Lden; outside mapped polygons means <55 dB.
    for column in (
        "noise_road_lden_db",
        "noise_rail_ratp_lden_db",
        "noise_rail_sncf_lden_db",
    ):
        output[column] = output[column].fillna(50.0)
    output["noise_transport_lden_max_db"] = output[
        ["noise_road_lden_db", "noise_rail_ratp_lden_db", "noise_rail_sncf_lden_db"]
    ].max(axis=1)
    return output


def link_addresses_and_buildings(
    sales: pd.DataFrame, archive: zipfile.ZipFile
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    ban = pd.read_csv(BAN_PATH, sep=";", dtype="string", low_memory=False)
    ban["canonical_address_key"] = address_key(
        ban["code_insee"], ban["id_fantoir"].str[-4:], ban["numero"], ban["rep"]
    )
    ban["ban_x"] = pd.to_numeric(ban["x"], errors="coerce")
    ban["ban_y"] = pd.to_numeric(ban["y"], errors="coerce")
    ban = ban.sort_values(
        ["canonical_address_key", "certification_commune"], ascending=[True, False]
    ).drop_duplicates("canonical_address_key")

    output = sales.copy()
    output["canonical_address_key"] = address_key(
        output["code_commune"],
        output["adresse_code_voie"],
        output["adresse_numero"],
        output["adresse_suffixe"],
    )
    output = output.merge(
        ban[
            [
                "canonical_address_key",
                "id",
                "id_fantoir",
                "type_position",
                "certification_commune",
                "ban_x",
                "ban_y",
                "cad_parcelles",
            ]
        ].rename(columns={"id": "ban_address_id"}),
        how="left",
        on="canonical_address_key",
        validate="many_to_one",
    )
    output["ban_match_method"] = np.where(
        output["ban_address_id"].notna(), "exact_fantoir_number", "unmatched"
    )
    output["ban_match_distance_m"] = np.hypot(
        output["x_l93"] - output["ban_x"], output["y_l93"] - output["ban_y"]
    )
    output["ban_match_confidence"] = np.where(
        output["ban_address_id"].notna(),
        np.where(output["ban_match_distance_m"].le(100).fillna(False), 1.0, 0.85),
        0.0,
    )

    rel_address = read_bdnb(
        archive,
        "rel_batiment_groupe_adresse",
        ["batiment_groupe_id", "cle_interop_adr", "origine", "fiabilite"],
    )
    rel_address["fiabilite"] = pd.to_numeric(rel_address["fiabilite"], errors="coerce")
    rel_address = rel_address.sort_values(
        ["cle_interop_adr", "fiabilite"], ascending=[True, False]
    )
    address_candidates = rel_address.groupby("cle_interop_adr").size()
    best_address = rel_address.drop_duplicates("cle_interop_adr").rename(
        columns={
            "cle_interop_adr": "ban_address_id",
            "batiment_groupe_id": "bdnb_building_id_address",
            "origine": "bdnb_address_origin",
            "fiabilite": "bdnb_address_reliability",
        }
    )
    output = output.merge(best_address, how="left", on="ban_address_id", validate="many_to_one")
    output["bdnb_address_candidate_count"] = (
        output["ban_address_id"].map(address_candidates).fillna(0).astype("int16")
    )

    rel_parcel = read_bdnb(
        archive,
        "rel_batiment_groupe_parcelle",
        ["batiment_groupe_id", "parcelle_id", "parcelle_principale"],
    )
    rel_parcel["parcelle_principale"] = pd.to_numeric(
        rel_parcel["parcelle_principale"], errors="coerce"
    ).fillna(0)
    parcel_candidates = rel_parcel.groupby("parcelle_id").size()
    best_parcel = rel_parcel.sort_values(
        ["parcelle_id", "parcelle_principale"], ascending=[True, False]
    ).drop_duplicates("parcelle_id")
    best_parcel = best_parcel.rename(
        columns={"batiment_groupe_id": "bdnb_building_id_parcel"}
    )
    output = output.merge(
        best_parcel[["parcelle_id", "bdnb_building_id_parcel"]],
        how="left",
        left_on="id_parcelle",
        right_on="parcelle_id",
        validate="many_to_one",
    )
    output["bdnb_parcel_candidate_count"] = (
        output["id_parcelle"].map(parcel_candidates).fillna(0).astype("int16")
    )
    output["bdnb_building_id"] = output["bdnb_building_id_address"].fillna(
        output["bdnb_building_id_parcel"]
    )
    output["bdnb_match_method"] = np.select(
        [
            output["bdnb_building_id_address"].notna(),
            output["bdnb_building_id_parcel"].notna(),
        ],
        ["ban_relation", "principal_parcel"],
        default="unmatched",
    )
    output["bdnb_match_confidence"] = np.select(
        [
            output["bdnb_building_id_address"].notna()
            & output["bdnb_address_reliability"].ge(15)
            & output["bdnb_address_candidate_count"].eq(1),
            output["bdnb_building_id_address"].notna(),
            output["bdnb_building_id_parcel"].notna()
            & output["bdnb_parcel_candidate_count"].eq(1),
            output["bdnb_building_id_parcel"].notna(),
        ],
        [1.0, 0.8, 0.7, 0.5],
        default=0.0,
    )
    audit = output[
        [
            "id_mutation",
            "canonical_address_key",
            "ban_address_id",
            "ban_match_method",
            "ban_match_distance_m",
            "ban_match_confidence",
            "bdnb_building_id",
            "bdnb_match_method",
            "bdnb_match_confidence",
            "bdnb_address_candidate_count",
            "bdnb_parcel_candidate_count",
        ]
    ].copy()
    quality = {
        "ban_exact_match_rate": float(output["ban_address_id"].notna().mean()),
        "bdnb_building_match_rate": float(output["bdnb_building_id"].notna().mean()),
        "bdnb_high_confidence_rate": float(output["bdnb_match_confidence"].ge(0.8).mean()),
    }
    return output, quality, audit


def add_building_features(data: pd.DataFrame, archive: zipfile.ZipFile) -> pd.DataFrame:
    output = data.copy()
    ffo = read_bdnb(
        archive,
        "batiment_groupe_ffo_bat",
        [
            "batiment_groupe_id",
            "nb_niveau",
            "annee_construction",
            "usage_niveau_1_txt",
            "mat_mur_txt",
            "mat_toit_txt",
            "nb_log",
        ],
    ).rename(
        columns={
            "nb_niveau": "building_level_count",
            "annee_construction": "building_year",
            "nb_log": "building_dwelling_count",
        }
    )
    bdtopo = read_bdnb(
        archive,
        "batiment_groupe_bdtopo_bat",
        ["batiment_groupe_id", "l_nature", "l_usage_1", "l_etat", "hauteur_mean"],
    ).rename(columns={"hauteur_mean": "building_height_m", "l_etat": "building_state"})
    buildings = read_bdnb(
        archive,
        "batiment_groupe",
        ["geom_groupe", "batiment_groupe_id", "s_geom_groupe"],
    ).rename(columns={"s_geom_groupe": "building_footprint_m2"})
    centroids = np.asarray([wkt_centroid(value) for value in buildings["geom_groupe"]])
    buildings["building_x"] = centroids[:, 0]
    buildings["building_y"] = centroids[:, 1]
    rnc = read_bdnb(
        archive,
        "batiment_groupe_rnc",
        [
            "batiment_groupe_id",
            "periode_construction_max",
            "nb_lot_garpark",
            "nb_lot_tot",
            "nb_log",
        ],
    ).rename(
        columns={
            "periode_construction_max": "building_period_rnc",
            "nb_lot_garpark": "building_parking_lot_count",
            "nb_lot_tot": "building_total_lot_count",
            "nb_log": "building_dwelling_count_rnc",
        }
    )
    for frame in (ffo, bdtopo, rnc):
        frame.drop_duplicates("batiment_groupe_id", inplace=True)
    building_features = buildings.merge(ffo, how="left", on="batiment_groupe_id")
    building_features = building_features.merge(bdtopo, how="left", on="batiment_groupe_id")
    building_features = building_features.merge(rnc, how="left", on="batiment_groupe_id")
    output = output.merge(
        building_features,
        how="left",
        left_on="bdnb_building_id",
        right_on="batiment_groupe_id",
        validate="many_to_one",
    )
    # ``bdnb_building_id`` is the canonical key kept in Gold.  The right-hand
    # join key is identical and would otherwise collide with the DPE relation
    # key in the following temporal merge.
    output.drop(columns=["batiment_groupe_id"], inplace=True)
    for column in (
        "building_level_count",
        "building_year",
        "building_dwelling_count",
        "building_height_m",
        "building_footprint_m2",
        "building_parking_lot_count",
        "building_total_lot_count",
        "building_dwelling_count_rnc",
    ):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output["building_age_at_sale"] = output["year"] - output["building_year"]
    output["building_age_at_sale"] = output["building_age_at_sale"].where(
        output["building_age_at_sale"].between(0, 1_000)
    )
    output["building_year_missing"] = output["building_year"].isna().astype("int8")
    output["apartment_floor"] = np.nan
    output["apartment_floor_available"] = np.int8(0)
    output["elevator_available"] = np.nan
    output["elevator_source_available"] = np.int8(0)
    return output


def add_point_in_time_dpe(data: pd.DataFrame, archive: zipfile.ZipFile) -> pd.DataFrame:
    relation = read_bdnb(
        archive,
        "rel_batiment_groupe_dpe_logement",
        ["batiment_groupe_id", "identifiant_dpe"],
    ).drop_duplicates()
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
    dpe = read_bdnb(archive, "dpe_logement", dpe_columns)
    dpe["date_etablissement_dpe"] = pd.to_datetime(
        dpe["date_etablissement_dpe"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    dpe = dpe.dropna(subset=["date_etablissement_dpe"])
    events = relation.merge(dpe, how="inner", on="identifiant_dpe", validate="many_to_one")
    events = events.sort_values(["date_etablissement_dpe", "batiment_groupe_id"])

    output = data.copy()
    output["_sale_order"] = np.arange(len(output))
    output["date_mutation"] = pd.to_datetime(output["date_mutation"])
    matched_parts: list[pd.DataFrame] = []
    # merge_asof by year-sized blocks avoids the very large Cartesian merge of
    # all diagnostics for each sale while retaining strict backward dates.
    eligible = output.loc[output["bdnb_building_id"].notna()].copy()
    eligible = eligible.sort_values(["date_mutation", "bdnb_building_id"])
    matched = pd.merge_asof(
        eligible,
        events,
        left_on="date_mutation",
        right_on="date_etablissement_dpe",
        left_by="bdnb_building_id",
        right_by="batiment_groupe_id",
        direction="backward",
        allow_exact_matches=True,
    )
    matched_parts.append(matched)
    unmatched = output.loc[output["bdnb_building_id"].isna()].copy()
    matched_parts.append(unmatched)
    output = (
        pd.concat(matched_parts, ignore_index=True, sort=False)
        .sort_values("_sale_order")
        .reset_index(drop=True)
    )
    output.drop(columns=["_sale_order", "batiment_groupe_id"], inplace=True)
    output["dpe_available_at_sale"] = output["identifiant_dpe"].notna().astype("int8")
    output["dpe_age_days"] = (
        output["date_mutation"] - output["date_etablissement_dpe"]
    ).dt.days
    output["dpe_surface_difference_ratio"] = (
        (
            pd.to_numeric(output["surface_habitable_logement"], errors="coerce")
            - output["surface_reelle_bati"]
        ).abs()
        / output["surface_reelle_bati"]
    )
    output["dpe_match_confidence"] = np.where(
        output["dpe_available_at_sale"].eq(1),
        np.where(output["dpe_surface_difference_ratio"].le(0.15), 1.0, 0.6),
        0.0,
    )
    for column in (
        "annee_construction_dpe",
        "nombre_niveau_logement",
        "nombre_niveau_immeuble",
        "surface_habitable_immeuble",
        "surface_habitable_logement",
        "conso_5_usages_ep_m2",
        "emission_ges_5_usages_m2",
    ):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def add_bpe_features(data: pd.DataFrame, archive: zipfile.ZipFile) -> pd.DataFrame:
    bpe = read_bdnb(
        archive, "batiment_groupe_bpe", ["batiment_groupe_id", "l_type_equipement"]
    ).drop_duplicates("batiment_groupe_id")
    buildings = read_bdnb(
        archive, "batiment_groupe", ["batiment_groupe_id", "geom_groupe"]
    )
    points = np.asarray([wkt_centroid(value) for value in buildings["geom_groupe"]])
    buildings["point_x"] = points[:, 0]
    buildings["point_y"] = points[:, 1]
    bpe = bpe.merge(buildings, how="left", on="batiment_groupe_id", validate="one_to_one")
    bpe["equipment_weight"] = bpe["l_type_equipement"].fillna("").map(
        lambda value: max(1, len(re.findall(r'"([^"]+)"', str(value))))
    )
    reference_xy = bpe[["point_x", "point_y"]].to_numpy(dtype=float)
    target_xy = data[["x_l93", "y_l93"]].to_numpy(dtype=float)
    output = data.copy()
    output["distance_shop_service_m"] = nearest_distance(target_xy, reference_xy)
    output["shop_service_count_500m"] = radius_count(
        target_xy,
        reference_xy,
        500,
        bpe["equipment_weight"].to_numpy(dtype=float),
    )
    output["shop_service_count_1000m"] = radius_count(
        target_xy,
        reference_xy,
        1_000,
        bpe["equipment_weight"].to_numpy(dtype=float),
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=PROJECT_ROOT / "data/gold/phase2_sale_features.parquet"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "data/gold/phase3_sale_features.parquet"
    )
    parser.add_argument(
        "--audit", type=Path, default=PROJECT_ROOT / "data/silver/phase3_match_audit.parquet"
    )
    parser.add_argument(
        "--quality", type=Path, default=PROJECT_ROOT / "reports/phase3/feature_quality.json"
    )
    args = parser.parse_args()

    data = pd.read_parquet(args.input)
    initial_ids = data["id_mutation"].copy()
    with zipfile.ZipFile(BDNB_ARCHIVE) as archive:
        data, match_quality, audit = link_addresses_and_buildings(data, archive)
        print("Linked BAN and BDNB", flush=True)
        data = add_building_features(data, archive)
        print("Added building features", flush=True)
        data = add_point_in_time_dpe(data, archive)
        print("Added point-in-time DPE features", flush=True)
        data = add_bpe_features(data, archive)
        print("Added BDNB BPE shop/service features", flush=True)

    data = add_transport_features(data)
    print("Added transport features", flush=True)
    data = add_school_features(data)
    print("Added school features", flush=True)
    data = add_green_features(data)
    print("Added green-space features", flush=True)
    data = add_noise_features(data)
    print("Added noise features", flush=True)

    if len(data) != len(initial_ids) or not data["id_mutation"].equals(initial_ids):
        raise RuntimeError("Phase 3 joins changed the Gold row identity/order")
    if data["id_mutation"].duplicated().any():
        raise RuntimeError("Phase 3 produced duplicate mutation IDs")
    leakage = data["date_etablissement_dpe"].notna() & (
        data["date_etablissement_dpe"] > data["date_mutation"]
    )
    if leakage.any():
        raise RuntimeError("Future DPE information leaked into historical sales")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.quality.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(args.output, index=False, compression="zstd")
    audit.to_parquet(args.audit, index=False, compression="zstd")

    phase3_columns = [column for column in data.columns if column not in pd.read_parquet(args.input).columns]
    missingness = {
        column: round(float(data[column].isna().mean()), 6) for column in phase3_columns
    }
    quality = {
        "schema_version": 1,
        "rows": len(data),
        "columns": len(data.columns),
        "input_sha256": sha256(args.input),
        "bdnb_archive_sha256": sha256(BDNB_ARCHIVE),
        "output_sha256": sha256(args.output),
        "match_quality": match_quality,
        "dpe_available_at_sale_rate": float(data["dpe_available_at_sale"].mean()),
        "future_dpe_leakage_rows": int(leakage.sum()),
        "apartment_floor_source_available": False,
        "elevator_source_available": False,
        "retrospective_static_features": [
            "BAN address/building identity",
            "BDNB physical building attributes",
            "2022 strategic noise map",
            "current transport stations",
            "current schools, green spaces and BPE services",
        ],
        "point_in_time_features": [
            "DPE fields with date_etablissement_dpe <= date_mutation",
            "all Phase 2 historical comparable features",
        ],
        "phase3_columns": phase3_columns,
        "missing_rate": missingness,
    }
    args.quality.write_text(
        json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved {args.output}: {len(data):,} rows, {len(data.columns)} columns")
    print(f"Saved {args.audit} and {args.quality}")


if __name__ == "__main__":
    main()
