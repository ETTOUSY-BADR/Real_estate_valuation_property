"""Failure-safety checks for Phase 3 feature artifacts."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock

from paris_avm.features.phase3 import write_parquet_atomic


class Phase3FeatureTests(unittest.TestCase):
    def test_parquet_is_promoted_only_after_successful_write(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "features.parquet"
            path.write_bytes(b"old parquet")
            data = MagicMock()

            def write_file(output: Path, **_: object) -> None:
                output.write_bytes(b"new parquet")

            data.to_parquet.side_effect = write_file

            write_parquet_atomic(data, path)

            self.assertEqual(path.read_bytes(), b"new parquet")
            self.assertFalse(path.with_suffix(".parquet.part").exists())
            data.to_parquet.assert_called_once_with(
                path.with_suffix(".parquet.part"), index=False, compression="zstd"
            )

    def test_parquet_write_failure_preserves_previous_file(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "features.parquet"
            path.write_bytes(b"valid parquet")
            temporary = path.with_suffix(".parquet.part")
            temporary.write_bytes(b"stale partial")
            data = MagicMock()

            def fail_after_partial_write(output: Path, **_: object) -> None:
                output.write_bytes(b"incomplete parquet")
                raise OSError("disk write failed")

            data.to_parquet.side_effect = fail_after_partial_write

            with self.assertRaisesRegex(OSError, "disk write failed"):
                write_parquet_atomic(data, path)

            self.assertEqual(path.read_bytes(), b"valid parquet")
            self.assertFalse(temporary.exists())


if __name__ == "__main__":
    unittest.main()
