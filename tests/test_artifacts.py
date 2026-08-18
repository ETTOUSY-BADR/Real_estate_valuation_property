"""Comprehensive tests for failure-safe artifact publication."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock

import pandas as pd

from paris_avm.artifacts import (
    temporary_path,
    write_atomic,
    write_csv_atomic,
    write_json_atomic,
    write_parquet_atomic,
)
from paris_avm.features import phase2


class AtomicArtifactTests(unittest.TestCase):
    def test_temporary_path_keeps_the_artifact_suffix(self) -> None:
        path = Path("nested/archive.tar.gz")
        self.assertEqual(temporary_path(path), Path("nested/archive.tar.gz.part"))

    def test_atomic_write_creates_parent_and_promotes_output(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "artifact.bin"

            write_atomic(path, lambda temporary: temporary.write_bytes(b"complete"))

            self.assertEqual(path.read_bytes(), b"complete")
            self.assertFalse(temporary_path(path).exists())

    def test_atomic_write_removes_stale_partial_before_writer_runs(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            temporary = temporary_path(path)
            temporary.write_bytes(b"stale")

            def writer(output: Path) -> None:
                self.assertFalse(output.exists())
                output.write_bytes(b"fresh")

            write_atomic(path, writer)

            self.assertEqual(path.read_bytes(), b"fresh")
            self.assertFalse(temporary.exists())

    def test_atomic_write_failure_preserves_previous_artifact(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            path.write_bytes(b"valid")

            def failing_writer(output: Path) -> None:
                output.write_bytes(b"partial")
                raise OSError("write interrupted")

            with self.assertRaisesRegex(OSError, "write interrupted"):
                write_atomic(path, failing_writer)

            self.assertEqual(path.read_bytes(), b"valid")
            self.assertFalse(temporary_path(path).exists())

    def test_atomic_write_failure_without_previous_artifact_leaves_no_file(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"

            def failing_writer(output: Path) -> None:
                output.write_bytes(b"partial")
                raise OSError("write interrupted")

            with self.assertRaisesRegex(OSError, "write interrupted"):
                write_atomic(path, failing_writer)

            self.assertFalse(path.exists())
            self.assertFalse(temporary_path(path).exists())

    def test_atomic_write_rejects_writer_that_produces_no_file(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            path.write_bytes(b"valid")

            with self.assertRaises(FileNotFoundError):
                write_atomic(path, lambda _: None)

            self.assertEqual(path.read_bytes(), b"valid")
            self.assertFalse(temporary_path(path).exists())

    def test_json_writer_preserves_unicode_and_format(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "report.json"

            write_json_atomic(path, {"city": "Île-de-France", "rows": 3})

            contents = path.read_text(encoding="utf-8")
            self.assertIn("Île-de-France", contents)
            self.assertIn("\n  ", contents)
            self.assertEqual(json.loads(contents), {"city": "Île-de-France", "rows": 3})

    def test_json_serialization_failure_preserves_old_file_and_cleans_partial(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text('{"status":"valid"}', encoding="utf-8")
            temporary_path(path).write_text("stale", encoding="utf-8")

            with self.assertRaises(TypeError):
                write_json_atomic(path, {"invalid": object()})

            self.assertEqual(path.read_text(encoding="utf-8"), '{"status":"valid"}')
            self.assertFalse(temporary_path(path).exists())

    def test_json_writer_supports_explicit_fallback_serialization(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"

            write_json_atomic(path, {"value": object()}, default=str)

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertIsInstance(payload["value"], str)
            self.assertIn("object", payload["value"])

    def test_parquet_writer_uses_stable_serialization_options(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "features.parquet"
            data = MagicMock()

            def writer(output: Path, **_: object) -> None:
                output.write_bytes(b"parquet")

            data.to_parquet.side_effect = writer
            write_parquet_atomic(data, path)

            data.to_parquet.assert_called_once_with(
                temporary_path(path), index=False, compression="zstd"
            )
            self.assertEqual(path.read_bytes(), b"parquet")

    def test_parquet_writer_round_trip(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "features.parquet"
            expected = pd.DataFrame(
                {"id_mutation": ["sale-1", "sale-2"], "price": [100_000, 200_000]}
            )

            write_parquet_atomic(expected, path)

            pd.testing.assert_frame_equal(pd.read_parquet(path), expected)
            self.assertFalse(temporary_path(path).exists())

    def test_csv_writer_uses_index_free_serialization(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            data = MagicMock()

            def writer(output: Path, **_: object) -> None:
                output.write_text("metric,value\nmae,10\n", encoding="utf-8")

            data.to_csv.side_effect = writer
            write_csv_atomic(data, path)

            data.to_csv.assert_called_once_with(temporary_path(path), index=False)
            self.assertEqual(path.read_text(encoding="utf-8"), "metric,value\nmae,10\n")

    def test_csv_write_failure_preserves_previous_file(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            path.write_text("valid\n", encoding="utf-8")
            data = MagicMock()

            def failing_writer(output: Path, **_: object) -> None:
                output.write_text("partial", encoding="utf-8")
                raise OSError("disk write failed")

            data.to_csv.side_effect = failing_writer
            with self.assertRaisesRegex(OSError, "disk write failed"):
                write_csv_atomic(data, path)

            self.assertEqual(path.read_text(encoding="utf-8"), "valid\n")
            self.assertFalse(temporary_path(path).exists())

    def test_csv_writer_round_trip(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "results.csv"
            expected = pd.DataFrame({"metric": ["mae", "rmse"], "value": [10.5, 20.5]})

            write_csv_atomic(expected, path)

            pd.testing.assert_frame_equal(pd.read_csv(path), expected)
            self.assertFalse(temporary_path(path).exists())

    def test_phase2_uses_shared_atomic_writers(self) -> None:
        self.assertIs(phase2.write_json_atomic, write_json_atomic)
        self.assertIs(phase2.write_parquet_atomic, write_parquet_atomic)


if __name__ == "__main__":
    unittest.main()
