"""Failure-safety checks for Phase 3 benchmark artifacts."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from paris_avm.artifacts import write_csv_atomic, write_json_atomic
from paris_avm.modeling import benchmark_phase3
from paris_avm.modeling.benchmark_phase3 import dump_joblib_atomic


class Phase3BenchmarkTests(unittest.TestCase):
    def test_benchmark_uses_shared_atomic_report_writers(self) -> None:
        self.assertIs(benchmark_phase3.write_csv_atomic, write_csv_atomic)
        self.assertIs(benchmark_phase3.write_json_atomic, write_json_atomic)

    def test_model_is_promoted_only_after_successful_serialization(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "selected_model.joblib"
            path.write_bytes(b"old model")
            artifact = {"model": object()}

            def write_file(_: object, output: Path) -> None:
                output.write_bytes(b"new model")

            with patch(
                "paris_avm.modeling.benchmark_phase3.joblib.dump",
                side_effect=write_file,
            ) as dump:
                dump_joblib_atomic(artifact, path)

            temporary = path.with_suffix(".joblib.part")
            self.assertEqual(path.read_bytes(), b"new model")
            self.assertFalse(temporary.exists())
            dump.assert_called_once_with(artifact, temporary)

    def test_model_serialization_failure_preserves_previous_file(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "selected_model.joblib"
            path.write_bytes(b"valid model")
            temporary = path.with_suffix(".joblib.part")
            temporary.write_bytes(b"stale partial")

            def fail_after_partial_write(_: object, output: Path) -> None:
                output.write_bytes(b"incomplete model")
                raise OSError("disk write failed")

            with patch(
                "paris_avm.modeling.benchmark_phase3.joblib.dump",
                side_effect=fail_after_partial_write,
            ):
                with self.assertRaisesRegex(OSError, "disk write failed"):
                    dump_joblib_atomic({"model": object()}, path)

            self.assertEqual(path.read_bytes(), b"valid model")
            self.assertFalse(temporary.exists())


if __name__ == "__main__":
    unittest.main()
