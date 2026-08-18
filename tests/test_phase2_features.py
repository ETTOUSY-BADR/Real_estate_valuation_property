"""Phase 2 feature-build contract and publication tests."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from paris_avm.features import phase2


class Phase2FeatureTests(unittest.TestCase):
    @staticmethod
    def valid_frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "id_mutation": ["sale-1", "sale-2"],
                "date_mutation": pd.to_datetime(["2021-01-04", "2022-02-05"]),
                "feature": [1.5, 2.5],
            }
        )

    def run_main(
        self,
        directory: str,
        frame: pd.DataFrame,
        *,
        extra_arguments: list[str] | None = None,
    ) -> tuple[Path, Path, Path]:
        root = Path(directory)
        source = root / "source.csv"
        source.write_bytes(b"source snapshot\n")
        output = root / "nested" / "phase2.parquet"
        report = root / "reports" / "quality.json"
        arguments = [
            "phase2",
            "--data",
            str(source),
            "--output",
            str(output),
            "--quality-report",
            str(report),
            *(extra_arguments or []),
        ]
        counts = {"input_rows": len(frame)}
        comparable_quality = {"availability_lag_days": 90}

        with (
            patch.object(sys, "argv", arguments),
            patch.object(phase2, "load_and_clean_many", return_value=(frame, counts)),
            patch.object(
                phase2, "add_identity_and_spatial_features", side_effect=lambda data: data
            ),
            patch.object(
                phase2,
                "add_comparable_features",
                return_value=(frame, comparable_quality),
            ),
            redirect_stdout(StringIO()),
        ):
            phase2.main()
        return source, output, report

    def test_main_publishes_gold_table_with_matching_checksum(self) -> None:
        with TemporaryDirectory() as directory:
            source, output, report_path = self.run_main(
                directory, self.valid_frame()
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))
            with output.open("rb") as handle:
                output_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
            with source.open("rb") as handle:
                source_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()

            self.assertEqual(report["output_sha256"], output_sha256)
            self.assertEqual(report["source_files"][0]["sha256"], source_sha256)
            self.assertEqual(report["source_files"][0]["bytes"], source.stat().st_size)
            self.assertEqual(report["row_count"], 2)
            self.assertEqual(report["feature_columns"], list(self.valid_frame().columns))
            pd.testing.assert_frame_equal(pd.read_parquet(output), self.valid_frame())
            self.assertFalse(output.with_suffix(".parquet.part").exists())
            self.assertFalse(report_path.with_suffix(".json.part").exists())

    def test_main_passes_custom_lag_and_chunk_size(self) -> None:
        with TemporaryDirectory() as directory:
            frame = self.valid_frame()
            root = Path(directory)
            source = root / "source.csv"
            source.write_text("source", encoding="utf-8")
            output = root / "phase2.parquet"
            report = root / "quality.json"
            arguments = [
                "phase2",
                "--data",
                str(source),
                "--output",
                str(output),
                "--quality-report",
                str(report),
                "--availability-lag-days",
                "120",
                "--chunk-size",
                "64",
            ]

            with (
                patch.object(sys, "argv", arguments),
                patch.object(
                    phase2, "load_and_clean_many", return_value=(frame, {})
                ),
                patch.object(
                    phase2,
                    "add_identity_and_spatial_features",
                    side_effect=lambda data: data,
                ),
                patch.object(
                    phase2,
                    "add_comparable_features",
                    return_value=(frame, {}),
                ) as comparable,
                redirect_stdout(StringIO()),
            ):
                phase2.main()

            comparable.assert_called_once_with(
                frame, availability_lag_days=120, chunk_size=64
            )

    def test_duplicate_mutation_ids_are_rejected_before_publication(self) -> None:
        frame = self.valid_frame()
        frame.loc[1, "id_mutation"] = frame.loc[0, "id_mutation"]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "duplicate mutation"):
                self.run_main(directory, frame)
            self.assertFalse((root / "nested" / "phase2.parquet").exists())
            self.assertFalse((root / "reports" / "quality.json").exists())

    def test_nonchronological_rows_are_rejected_before_publication(self) -> None:
        frame = self.valid_frame().iloc[::-1].reset_index(drop=True)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "not chronologically ordered"):
                self.run_main(directory, frame)
            self.assertFalse((root / "nested" / "phase2.parquet").exists())
            self.assertFalse((root / "reports" / "quality.json").exists())

    def test_nonpositive_availability_lag_is_rejected_before_loading(self) -> None:
        with (
            patch.object(
                sys,
                "argv",
                ["phase2", "--availability-lag-days", "0"],
            ),
            patch.object(phase2, "load_and_clean_many") as load,
            redirect_stderr(StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            phase2.main()

        self.assertEqual(raised.exception.code, 2)
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
