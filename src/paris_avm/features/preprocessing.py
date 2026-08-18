"""Run the complete DVF preprocessing and enrichment workflow.

The default command builds property, time, spatial and leakage-safe historical
comparable features, then adds BAN, BDNB, point-in-time DPE and neighborhood
context.  Each stage publishes atomically and validates its feature contract.
"""

from __future__ import annotations

import argparse
import gc
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

from paris_avm.artifacts import write_json_atomic
from paris_avm.features.phase2 import build_phase2_features, file_sha256
from paris_avm.features.phase3 import build_phase3_features
from paris_avm.features.schema import contract_summary
from paris_avm.modeling.train_phase1 import DEFAULT_DATA_FILES
from paris_avm.paths import PROJECT_ROOT


DEFAULT_PHASE2_OUTPUT = PROJECT_ROOT / "data/gold/phase2_sale_features.parquet"
DEFAULT_PHASE3_OUTPUT = PROJECT_ROOT / "data/gold/phase3_sale_features.parquet"
DEFAULT_PHASE2_QUALITY = PROJECT_ROOT / "reports/phase2/feature_quality.json"
DEFAULT_PHASE3_QUALITY = PROJECT_ROOT / "reports/phase3/feature_quality.json"
DEFAULT_MATCH_AUDIT = PROJECT_ROOT / "data/silver/phase3_match_audit.parquet"
DEFAULT_RUN_REPORT = PROJECT_ROOT / "reports/preprocessing/run_summary.json"


def run_preprocessing(
    data_files: list[Path],
    *,
    phase2_output: Path = DEFAULT_PHASE2_OUTPUT,
    phase3_output: Path = DEFAULT_PHASE3_OUTPUT,
    phase2_quality: Path = DEFAULT_PHASE2_QUALITY,
    phase3_quality: Path = DEFAULT_PHASE3_QUALITY,
    match_audit: Path = DEFAULT_MATCH_AUDIT,
    run_report: Path = DEFAULT_RUN_REPORT,
    availability_lag_days: int = 90,
    chunk_size: int = 512,
    phase2_only: bool = False,
    reuse_phase2: bool = False,
) -> dict[str, object]:
    """Build points 1--5 and write a compact run-level provenance report."""
    started = perf_counter()
    if reuse_phase2:
        if not phase2_output.exists() or not phase2_quality.exists():
            raise FileNotFoundError(
                "--reuse-phase2 requires an existing Phase 2 table and quality report"
            )
        phase2_report = json.loads(phase2_quality.read_text(encoding="utf-8"))
        if phase2_report.get("output_sha256") != file_sha256(phase2_output):
            raise RuntimeError("Existing Phase 2 checksum does not match its quality report")
        phase2_rows = int(phase2_report["row_count"])
    else:
        phase2_data, phase2_report = build_phase2_features(
            data_files,
            phase2_output,
            phase2_quality,
            availability_lag_days=availability_lag_days,
            chunk_size=chunk_size,
        )
        phase2_rows = len(phase2_data)
    summary: dict[str, object] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "phase2_complete" if phase2_only else "complete",
        "source_files": [str(path.resolve()) for path in data_files],
        "availability_lag_days": availability_lag_days,
        "feature_contract": contract_summary(),
        "phase2": {
            "rows": phase2_rows,
            "output": str(phase2_output.resolve()),
            "output_sha256": phase2_report["output_sha256"],
            "quality_report": str(phase2_quality.resolve()),
        },
    }

    if not phase2_only:
        # Phase 3 joins several large BDNB tables.  Do not retain the in-memory
        # Phase 2 frame after it has been atomically published and summarized.
        if not reuse_phase2:
            del phase2_data
            gc.collect()
        phase3_data, phase3_report = build_phase3_features(
            phase2_output,
            phase3_output,
            match_audit,
            phase3_quality,
        )
        summary["phase3"] = {
            "rows": len(phase3_data),
            "columns": len(phase3_data.columns),
            "output": str(phase3_output.resolve()),
            "output_sha256": phase3_report["output_sha256"],
            "quality_report": str(phase3_quality.resolve()),
            "match_audit": str(match_audit.resolve()),
        }

    summary["runtime_seconds"] = round(perf_counter() - started, 3)
    write_json_atomic(run_report, summary)
    print(f"Saved preprocessing run report: {run_report}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", type=Path, default=DEFAULT_DATA_FILES)
    parser.add_argument("--phase2-output", type=Path, default=DEFAULT_PHASE2_OUTPUT)
    parser.add_argument("--phase3-output", type=Path, default=DEFAULT_PHASE3_OUTPUT)
    parser.add_argument("--phase2-quality", type=Path, default=DEFAULT_PHASE2_QUALITY)
    parser.add_argument("--phase3-quality", type=Path, default=DEFAULT_PHASE3_QUALITY)
    parser.add_argument("--match-audit", type=Path, default=DEFAULT_MATCH_AUDIT)
    parser.add_argument("--run-report", type=Path, default=DEFAULT_RUN_REPORT)
    parser.add_argument("--availability-lag-days", type=int, default=90)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument(
        "--phase2-only",
        action="store_true",
        help="Build DVF, time, spatial and comparable features without enrichment.",
    )
    parser.add_argument(
        "--reuse-phase2",
        action="store_true",
        help="Reuse a checksum-verified Phase 2 table and run enrichment only.",
    )
    args = parser.parse_args()
    if args.availability_lag_days < 1:
        parser.error("--availability-lag-days must be positive")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")

    run_preprocessing(
        args.data,
        phase2_output=args.phase2_output,
        phase3_output=args.phase3_output,
        phase2_quality=args.phase2_quality,
        phase3_quality=args.phase3_quality,
        match_audit=args.match_audit,
        run_report=args.run_report,
        availability_lag_days=args.availability_lag_days,
        chunk_size=args.chunk_size,
        phase2_only=args.phase2_only,
        reuse_phase2=args.reuse_phase2,
    )


if __name__ == "__main__":
    main()
