"""Acquire versioned Paris external datasets used by the Phase 3 feature build.

Only authoritative/open-data endpoints are used. Large files are cached and are
never downloaded again unless ``--force`` is supplied. Every artifact is
registered with its URL, retrieval time, byte size and SHA-256 checksum.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from urllib.parse import urlencode
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from paris_avm.artifacts import write_json_atomic
from paris_avm.paths import PROJECT_ROOT


BAN_SNAPSHOT_DATE = "2026-08-12"
BAN_SHA256 = "38dd8f34090c4a9d8bc04ddf56868f911625de7455592f0e70516288a76a86a7"
BAN_URL = (
    f"https://adresse.data.gouv.fr/data/ban/adresses/{BAN_SNAPSHOT_DATE}/"
    "csv/adresses-75.csv.gz"
)
BAN_PATH = (
    PROJECT_ROOT
    / f"data/bronze/ban/snapshot={BAN_SNAPSHOT_DATE}/adresses-75.csv.gz"
)
BDNB_RELEASE = "2026-02-a"
BDNB_SHA256 = "9476d90130607bc33b72cf578bf931210de525c6748dd21a338aac9e8ce972df"
BDNB_METADATA_SHA256 = (
    "6db5f37b06ca0ae71a987f070cf23ba64f9c23f59a9f3d3364fc8dd632280865"
)
BDNB_BASE_URL = (
    f"https://open-data.s3.fr-par.scw.cloud/bdnb_millesime_{BDNB_RELEASE}/"
    f"millesime_{BDNB_RELEASE}_dep75"
)
BDNB_URL = (
    f"{BDNB_BASE_URL}/open_data_millesime_{BDNB_RELEASE}_dep75_csv.zip"
)
BDNB_METADATA_URL = (
    f"{BDNB_BASE_URL}/open_data_millesime_{BDNB_RELEASE}_dep75_csv_metadata.yml"
)
BDNB_ARCHIVE = (
    PROJECT_ROOT
    / f"data/bronze/bdnb/release={BDNB_RELEASE}/"
    f"open_data_millesime_{BDNB_RELEASE}_dep75_csv.zip"
)
BDNB_METADATA_PATH = (
    PROJECT_ROOT / f"data/bronze/bdnb/release={BDNB_RELEASE}/metadata.yml"
)
DPE_DATASET_ID = "meg-83tjwtg8dyz4vv7h1dqe"
DPE_API = f"https://data.ademe.fr/data-fair/api/v1/datasets/{DPE_DATASET_ID}"

ODS_EXPORT = "{host}/api/explore/v2.1/catalog/datasets/{dataset}/exports/csv"
CSV_SOURCES = {
    "idfm_stations": {
        "host": "https://data.iledefrance.fr",
        "dataset": "gares-et-stations-du-reseau-ferre-dile-de-france-par-ligne",
        "producer": "Ile-de-France Mobilites",
        "license": "Licence Ouverte 2.0",
    },
    "schools_elementary": {
        "host": "https://opendata.paris.fr",
        "dataset": "etablissements-scolaires-ecoles-elementaires",
        "producer": "Ville de Paris - DASCO",
        "license": "ODbL",
    },
    "schools_maternelle": {
        "host": "https://opendata.paris.fr",
        "dataset": "etablissements-scolaires-maternelles",
        "producer": "Ville de Paris - DASCO",
        "license": "ODbL",
    },
    "schools_colleges": {
        "host": "https://opendata.paris.fr",
        "dataset": "etablissements-scolaires-colleges",
        "producer": "Ville de Paris - DASCO",
        "license": "ODbL",
    },
    "green_spaces": {
        "host": "https://opendata.paris.fr",
        "dataset": "espaces_verts",
        "producer": "Ville de Paris - DEVE",
        "license": "ODbL",
    },
}

NOISE_WFS_BASE = "https://ogc.geo-ide.developpement-durable.gouv.fr/wxs"
NOISE_WFS_MAP = (
    "/opt/data/stack/mapfiles/1.4/org_3954051/"
    "fr-120066022-orphan-63989501-bf01-4322-b948-5cd1b8b03e05.internet.map"
)
NOISE_LAYERS = {
    "noise_road_lden": "ms:N_BRUIT_ZBR_INFRA_R_A_LD_S_075",
    "noise_rail_ratp_lden": "ms:N_BRUIT_ZBR_INFRA_RATP_F_A_LD_S_075",
    "noise_rail_sncf_lden": "ms:N_BRUIT_ZBR_INFRA_SNCF_F_A_LD_S_075",
}

DPE_FIELDS = [
    "numero_dpe",
    "date_etablissement_dpe",
    "etiquette_dpe",
    "etiquette_ges",
    "conso_5_usages_par_m2_ep",
    "emission_ges_5_usages_par_m2",
    "annee_construction",
    "type_batiment",
    "surface_habitable_logement",
    "surface_habitable_immeuble",
    "numero_etage_appartement",
    "nombre_appartement",
    "nombre_niveau_immeuble",
    "nombre_niveau_logement",
    "type_energie_principale_chauffage",
    "identifiant_ban",
    "adresse_ban",
    "code_postal_ban",
    "code_insee_ban",
    "score_ban",
    "coordonnee_cartographique_x_ban",
    "coordonnee_cartographique_y_ban",
]


def session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
    )
    output = requests.Session()
    output.headers["User-Agent"] = "paris-avm-phase3/1.0 (open-data research)"
    output.mount("https://", HTTPAdapter(max_retries=retry))
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(
    client: requests.Session,
    url: str,
    path: Path,
    force: bool,
    expected_sha256: str | None = None,
) -> None:
    if path.exists() and path.stat().st_size and not force:
        if expected_sha256 is not None:
            actual_sha256 = sha256(path)
            if actual_sha256.lower() != expected_sha256.lower():
                raise RuntimeError(
                    f"Checksum mismatch for cached {path}: expected "
                    f"{expected_sha256}, received {actual_sha256}; use --force "
                    "to reacquire it"
                )
        print(f"Cached: {path}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        with client.get(url, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for block in response.iter_content(1024 * 1024):
                    if block:
                        handle.write(block)
        if not temporary.stat().st_size:
            raise RuntimeError(f"Downloaded source is empty: {url}")
        if expected_sha256 is not None:
            actual_sha256 = sha256(temporary)
            if actual_sha256.lower() != expected_sha256.lower():
                raise RuntimeError(
                    f"Checksum mismatch for {url}: expected {expected_sha256}, "
                    f"received {actual_sha256}"
                )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Downloaded: {path} ({path.stat().st_size:,} bytes)", flush=True)


def acquire_dpe(
    client: requests.Session,
    path: Path,
    force: bool,
    ban_ids: set[str] | None = None,
) -> dict[str, Any]:
    if path.exists() and path.stat().st_size and not force:
        print(f"Cached: {path}", flush=True)
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            rows = sum(1 for _ in handle)
        # The cached file may contain only records matching ``ban_ids``. Its
        # row count therefore cannot reconstruct the upstream Paris total.
        return {"rows": rows, "api_total": None, "cache_hit": True}

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    # Query exact Paris postcodes independently. This avoids the expensive
    # wildcard score/sort used by Data Fair and bounds each paginated query
    # while retaining every Paris record.
    count = 0
    api_total = 0
    temporary.unlink(missing_ok=True)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", newline="\n") as output:
            for arrondissement in range(1, 21):
                postcode = f"750{arrondissement:02d}"
                url: str | None = f"{DPE_API}/lines"
                params: dict[str, Any] | None = {
                    "size": 10_000,
                    "qs": f"code_postal_ban:{postcode}",
                    "select": ",".join(DPE_FIELDS),
                }
                arrondissement_count = 0
                arrondissement_total = 0
                while url:
                    response = client.get(url, params=params, timeout=(30, 180))
                    response.raise_for_status()
                    payload = response.json()
                    arrondissement_total = int(
                        payload.get("total", arrondissement_total)
                    )
                    for record in payload.get("results", []):
                        record.pop("_score", None)
                        identifier = str(record.get("identifiant_ban") or "")
                        if ban_ids is None or identifier in ban_ids:
                            output.write(
                                json.dumps(
                                    record,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                )
                            )
                            output.write("\n")
                            count += 1
                        arrondissement_count += 1
                    url = payload.get("next")
                    params = None
                if arrondissement_count != arrondissement_total:
                    raise RuntimeError(
                        f"Incomplete DPE response for {postcode}: received "
                        f"{arrondissement_count:,} of {arrondissement_total:,} records"
                    )
                api_total += arrondissement_total
                print(
                    f"DPE {postcode}: scanned {arrondissement_count:,}, "
                    f"retained {count:,}",
                    flush=True,
                )
        if not api_total:
            raise RuntimeError("ADEME returned no DPE records for Paris")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"rows": count, "api_total": api_total, "cache_hit": False}


def artifact_record(
    source_id: str,
    producer: str,
    license_name: str,
    url: str,
    path: Path,
    retrieved_at: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        manifest_path = path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        manifest_path = path.as_posix()
    record: dict[str, Any] = {
        "source_id": source_id,
        "producer": producer,
        "license": license_name,
        "url": url,
        "path": manifest_path,
        "retrieved_at_utc": retrieved_at,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if extra:
        record.update(extra)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-dpe", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/bronze/phase3")
    parser.add_argument(
        "--manifest", type=Path, default=PROJECT_ROOT / "reports/phase3/source_manifest.json"
    )
    args = parser.parse_args()

    client = session()
    retrieved_at = datetime.now(timezone.utc).isoformat()
    artifacts: list[dict[str, Any]] = []

    ban_path = BAN_PATH
    download_file(
        client,
        BAN_URL,
        ban_path,
        args.force,
        expected_sha256=BAN_SHA256,
    )
    artifacts.append(
        artifact_record(
            "ban_paris",
            "Base Adresse Nationale / IGN",
            "Licence Ouverte 2.0",
            BAN_URL,
            ban_path,
            retrieved_at,
            {
                "snapshot_date": BAN_SNAPSHOT_DATE,
                "snapshot_semantics": "archived static address reference",
            },
        )
    )

    bdnb_path = BDNB_ARCHIVE
    bdnb_meta_path = BDNB_METADATA_PATH
    download_file(
        client,
        BDNB_URL,
        bdnb_path,
        args.force,
        expected_sha256=BDNB_SHA256,
    )
    download_file(
        client,
        BDNB_METADATA_URL,
        bdnb_meta_path,
        args.force,
        expected_sha256=BDNB_METADATA_SHA256,
    )
    artifacts.append(
        artifact_record(
            "bdnb_paris_2026_02_a",
            "CSTB",
            "Licence Ouverte 2.0",
            BDNB_URL,
            bdnb_path,
            retrieved_at,
            {
                "release": BDNB_RELEASE,
                "snapshot_semantics": "retrospective static reconstruction",
            },
        )
    )
    artifacts.append(
        artifact_record(
            "bdnb_paris_2026_02_a_metadata",
            "CSTB",
            "Licence Ouverte 2.0",
            BDNB_METADATA_URL,
            bdnb_meta_path,
            retrieved_at,
        )
    )

    if not args.skip_dpe:
        dpe_path = args.output / "dpe_paris_selected.jsonl.gz"
        # Only DPEs at BAN addresses present in the sale cohort can ever join.
        # Filtering during streaming reduces the local snapshot by an order of
        # magnitude without changing any training feature.
        import pandas as pd

        sales = pd.read_parquet(
            PROJECT_ROOT / "data/gold/phase2_sale_features.parquet",
            columns=["address_id"],
        )
        sale_address_ids = set(sales["address_id"].dropna().astype(str))
        dpe_stats = acquire_dpe(client, dpe_path, args.force, sale_address_ids)
        artifacts.append(
            artifact_record(
                "dpe_existing_paris",
                "ADEME",
                "Licence Ouverte 2.0",
                f"{DPE_API}/lines?qs=code_postal_ban:75*",
                dpe_path,
                retrieved_at,
                {
                    **dpe_stats,
                    "selected_fields": DPE_FIELDS,
                    "snapshot_semantics": "point-in-time by date_etablissement_dpe",
                },
            )
        )

    for source_id, source in CSV_SOURCES.items():
        url = ODS_EXPORT.format(host=source["host"], dataset=source["dataset"])
        params = "?lang=fr&timezone=Europe%2FParis&use_labels=false&delimiter=%3B"
        path = args.output / f"{source_id}.csv"
        download_file(client, url + params, path, args.force)
        artifacts.append(
            artifact_record(
                source_id,
                source["producer"],
                source["license"],
                url + params,
                path,
                retrieved_at,
                {"snapshot_semantics": "current static context"},
            )
        )

    for source_id, layer in NOISE_LAYERS.items():
        query = urlencode(
            {
                "map": NOISE_WFS_MAP,
                "SERVICE": "WFS",
                "VERSION": "2.0.0",
                "REQUEST": "GetFeature",
                "TYPENAMES": layer,
                "SRSNAME": "EPSG:2154",
            }
        )
        url = f"{NOISE_WFS_BASE}?{query}"
        path = args.output / f"{source_id}.gml"
        download_file(client, url, path, args.force)
        artifacts.append(
            artifact_record(
                source_id,
                "DRIEAT Ile-de-France / CEREMA / RATP / SNCF",
                "Licence Ouverte 2.0",
                url,
                path,
                retrieved_at,
                {"snapshot_semantics": "2022 strategic Lden noise map (static)"},
            )
        )

    manifest = {
        "schema_version": 1,
        "retrieved_at_utc": retrieved_at,
        "geography": "Paris, department 75",
        "artifacts": artifacts,
        "notes": [
            "DPE must be joined with date_etablissement_dpe <= valuation date.",
            "BAN and BDNB are pinned static snapshots; transport and amenity "
            "sources are current static snapshots.",
            "No public elevator field was found in the selected open BDNB/DPE schemas.",
            "BDNB BPE supplies shop/service POIs; Paris Open Data supplies schools and parks.",
        ],
    }
    write_json_atomic(args.manifest, manifest)
    print(f"Saved manifest: {args.manifest}", flush=True)


if __name__ == "__main__":
    main()
