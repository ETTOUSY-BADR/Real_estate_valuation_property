"""End-to-end preprocessing orchestration tests."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from paris_avm.features import preprocessing


class PreprocessingTests(unittest.TestCase):
    def paths(self, directory: str) -> dict[str, Path]:
        root = Path(directory)
        return {
            "phase2_output": root / "gold" / "phase2.parquet",
            "phase3_output": root / "gold" / "phase3.parquet",
            "phase2_quality": root / "reports" / "phase2.json",
            "phase3_quality": root / "reports" / "phase3.json",
            "match_audit": root / "silver" / "audit.parquet",
            "run_report": root / "reports" / "run.json",
        }

    def test_complete_workflow_runs_phase2_then_phase3(self) -> None:
        with TemporaryDirectory() as directory:
            paths = self.paths(directory)
            source = Path(directory) / "75_2022.csv"
            source.write_text("source", encoding="utf-8")
            phase2_frame = pd.DataFrame({"id_mutation": ["sale-1"]})
            phase3_frame = pd.DataFrame(
                {"id_mutation": ["sale-1"], "added_feature": [1.0]}
            )

            with (
                patch.object(
                    preprocessing,
                    "build_phase2_features",
                    return_value=(phase2_frame, {"output_sha256": "phase2-hash"}),
                ) as phase2_build,
                patch.object(
                    preprocessing,
                    "build_phase3_features",
                    return_value=(phase3_frame, {"output_sha256": "phase3-hash"}),
                ) as phase3_build,
            ):
                summary = preprocessing.run_preprocessing([source], **paths)

            phase2_build.assert_called_once_with(
                [source],
                paths["phase2_output"],
                paths["phase2_quality"],
                availability_lag_days=90,
                chunk_size=512,
            )
            phase3_build.assert_called_once_with(
                paths["phase2_output"],
                paths["phase3_output"],
                paths["match_audit"],
                paths["phase3_quality"],
            )
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["phase3"]["columns"], 2)
            published = json.loads(paths["run_report"].read_text(encoding="utf-8"))
            self.assertEqual(published["phase2"]["output_sha256"], "phase2-hash")
            self.assertEqual(published["phase3"]["output_sha256"], "phase3-hash")

    def test_phase2_only_does_not_run_external_enrichment(self) -> None:
        with TemporaryDirectory() as directory:
            paths = self.paths(directory)
            source = Path(directory) / "75.csv"
            source.write_text("source", encoding="utf-8")

            with (
                patch.object(
                    preprocessing,
                    "build_phase2_features",
                    return_value=(
                        pd.DataFrame({"id_mutation": ["sale-1"]}),
                        {"output_sha256": "phase2-hash"},
                    ),
                ),
                patch.object(preprocessing, "build_phase3_features") as phase3_build,
            ):
                summary = preprocessing.run_preprocessing(
                    [source], phase2_only=True, **paths
                )

            phase3_build.assert_not_called()
            self.assertEqual(summary["status"], "phase2_complete")
            self.assertNotIn("phase3", summary)

    def test_reuse_phase2_verifies_checksum_and_skips_rebuild(self) -> None:
        with TemporaryDirectory() as directory:
            paths = self.paths(directory)
            paths["phase2_output"].parent.mkdir(parents=True)
            paths["phase2_quality"].parent.mkdir(parents=True)
            paths["phase2_output"].write_bytes(b"phase2")
            paths["phase2_quality"].write_text(
                json.dumps({"row_count": 12, "output_sha256": "verified-hash"}),
                encoding="utf-8",
            )

            with (
                patch.object(preprocessing, "file_sha256", return_value="verified-hash"),
                patch.object(preprocessing, "build_phase2_features") as phase2_build,
                patch.object(
                    preprocessing,
                    "build_phase3_features",
                    return_value=(
                        pd.DataFrame({"id_mutation": ["sale-1"]}),
                        {"output_sha256": "phase3-hash"},
                    ),
                ) as phase3_build,
            ):
                summary = preprocessing.run_preprocessing(
                    [], reuse_phase2=True, **paths
                )

            phase2_build.assert_not_called()
            phase3_build.assert_called_once()
            self.assertEqual(summary["phase2"]["rows"], 12)

    def test_reuse_phase2_rejects_checksum_mismatch(self) -> None:
        with TemporaryDirectory() as directory:
            paths = self.paths(directory)
            paths["phase2_output"].parent.mkdir(parents=True)
            paths["phase2_quality"].parent.mkdir(parents=True)
            paths["phase2_output"].write_bytes(b"phase2")
            paths["phase2_quality"].write_text(
                json.dumps({"row_count": 12, "output_sha256": "expected"}),
                encoding="utf-8",
            )

            with (
                patch.object(preprocessing, "file_sha256", return_value="actual"),
                patch.object(preprocessing, "build_phase3_features") as phase3_build,
                self.assertRaisesRegex(RuntimeError, "checksum"),
            ):
                preprocessing.run_preprocessing([], reuse_phase2=True, **paths)

            phase3_build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
