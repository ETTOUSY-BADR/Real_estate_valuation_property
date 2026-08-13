"""Unit tests for Phase 3 source acquisition and provenance."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import redirect_stdout
import gzip
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock

from paris_avm.data.acquire_phase3 import acquire_dpe, download_file


class Phase3AcquisitionTests(unittest.TestCase):
    @staticmethod
    def download_client(blocks: Iterable[bytes]) -> tuple[MagicMock, MagicMock]:
        response = MagicMock()
        response.iter_content.return_value = blocks
        client = MagicMock()
        client.get.return_value.__enter__.return_value = response
        return client, response

    def test_download_replaces_cached_source_only_after_success(self) -> None:
        client, response = self.download_client([b"new ", b"", b"snapshot"])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            path.write_bytes(b"old snapshot")

            with redirect_stdout(StringIO()):
                download_file(client, "https://example.test/source", path, force=True)

            self.assertEqual(path.read_bytes(), b"new snapshot")
            self.assertFalse(path.with_suffix(".csv.part").exists())

        response.raise_for_status.assert_called_once_with()

    def test_empty_download_preserves_cached_source(self) -> None:
        client, _ = self.download_client([b"", b""])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            path.write_bytes(b"valid cached snapshot")

            with self.assertRaisesRegex(RuntimeError, "Downloaded source is empty"):
                download_file(client, "https://example.test/source", path, force=True)

            self.assertEqual(path.read_bytes(), b"valid cached snapshot")
            self.assertFalse(path.with_suffix(".csv.part").exists())

    def test_interrupted_download_preserves_cache_and_removes_partial(self) -> None:
        def interrupted_blocks() -> Iterator[bytes]:
            yield b"partial"
            raise ConnectionError("connection interrupted")

        client, _ = self.download_client(interrupted_blocks())
        with TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            path.write_bytes(b"valid cached snapshot")

            with self.assertRaisesRegex(ConnectionError, "connection interrupted"):
                download_file(client, "https://example.test/source", path, force=True)

            self.assertEqual(path.read_bytes(), b"valid cached snapshot")
            self.assertFalse(path.with_suffix(".csv.part").exists())

    def test_cached_filtered_snapshot_does_not_claim_upstream_total(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "dpe.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as output:
                output.write('{"numero_dpe":"one"}\n')
                output.write('{"numero_dpe":"two"}\n')

            client = MagicMock()
            with redirect_stdout(StringIO()):
                stats = acquire_dpe(client, path, force=False, ban_ids={"selected"})

        self.assertEqual(
            stats,
            {"rows": 2, "api_total": None, "cache_hit": True},
        )
        client.get.assert_not_called()

    def test_fresh_filtered_snapshot_distinguishes_rows_from_api_total(self) -> None:
        response = MagicMock()
        response.json.return_value = {
            "total": 2,
            "results": [
                {"numero_dpe": "selected", "identifiant_ban": "keep"},
                {"numero_dpe": "excluded", "identifiant_ban": "drop"},
            ],
        }
        client = MagicMock()
        client.get.return_value = response

        with TemporaryDirectory() as directory:
            path = Path(directory) / "dpe.jsonl.gz"
            with redirect_stdout(StringIO()):
                stats = acquire_dpe(client, path, force=False, ban_ids={"keep"})
            with gzip.open(path, "rt", encoding="utf-8") as source:
                records = [json.loads(line) for line in source]

        self.assertEqual(
            stats,
            {"rows": 20, "api_total": 40, "cache_hit": False},
        )
        self.assertEqual(len(records), 20)
        self.assertEqual({record["numero_dpe"] for record in records}, {"selected"})
        self.assertEqual(client.get.call_count, 20)


if __name__ == "__main__":
    unittest.main()
