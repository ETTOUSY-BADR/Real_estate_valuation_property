"""Download immutable annual nationwide Geo-DVF snapshots for Phase 4."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import requests

from paris_avm.artifacts import write_json_atomic
from paris_avm.paths import PROJECT_ROOT


SOURCE_ROOT = "https://files.data.gouv.fr/geo-dvf/latest/csv"
DEFAULT_YEARS = tuple(range(2021, 2026))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_atomic(url: str, destination: Path, *, timeout: int = 120) -> None:
    """Stream a source file and publish it only after a complete download."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with partial.open("wb") as output:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        output.write(block)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def acquire(
    years: list[int], output_dir: Path, manifest_path: Path, *, force: bool = False
) -> dict[str, object]:
    """Acquire selected full-France yearly archives and write their provenance."""
    if not years:
        raise ValueError("At least one year is required")
    records: list[dict[str, object]] = []
    for year in sorted(set(years)):
        if year < 2014 or year > datetime.now(timezone.utc).year:
            raise ValueError(f"Unsupported Geo-DVF year: {year}")
        url = f"{SOURCE_ROOT}/{year}/full.csv.gz"
        destination = output_dir / f"france_{year}.csv.gz"
        if force or not destination.exists():
            print(f"Downloading {url}", flush=True)
            download_atomic(url, destination)
        records.append(
            {
                "year": year,
                "source_url": url,
                "path": str(destination.resolve()),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )

    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset": "Geolocated DVF - nationwide France",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_index": f"{SOURCE_ROOT}/",
        "coverage_note": (
            "DVF does not publish Bas-Rhin (67), Haut-Rhin (68), Moselle (57), "
            "or Mayotte (976); all other departments present upstream are retained."
        ),
        "files": records,
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=list(DEFAULT_YEARS))
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "data/raw/dvf_france"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/raw/dvf_france/manifest.json",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = acquire(args.years, args.output_dir, args.manifest, force=args.force)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
