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

from paris_avm.data.acquire_phase3 import (
    BAN_PATH,
    BAN_SHA256,
    BAN_SNAPSHOT_DATE,
    BAN_URL,
    BDNB_ARCHIVE,
    BDNB_METADATA_PATH,
    BDNB_METADATA_SHA256,
    BDNB_RELEASE,
    BDNB_SHA256,
    BDNB_URL,
    acquire_dpe,
    download_file,
    write_json_atomic,
)


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

    def test_checksum_mismatch_preserves_cached_source(self) -> None:
        client, _ = self.download_client([b"unexpected snapshot"])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            path.write_bytes(b"valid cached snapshot")

            with self.assertRaisesRegex(RuntimeError, "Checksum mismatch"):
                download_file(
                    client,
                    "https://example.test/source",
                    path,
                    force=True,
                    expected_sha256="0" * 64,
                )

            self.assertEqual(path.read_bytes(), b"valid cached snapshot")
            self.assertFalse(path.with_suffix(".csv.part").exists())

    def test_checksum_mismatch_rejects_cached_source_without_network(self) -> None:
        client = MagicMock()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            path.write_bytes(b"corrupt cached snapshot")

            with self.assertRaisesRegex(RuntimeError, "Checksum mismatch for cached"):
                download_file(
                    client,
                    "https://example.test/source",
                    path,
                    force=False,
                    expected_sha256="0" * 64,
                )

            self.assertEqual(path.read_bytes(), b"corrupt cached snapshot")
        client.get.assert_not_called()

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

    def test_manifest_is_replaced_only_after_valid_json_is_written(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "source_manifest.json"
            path.write_text('{"status":"old"}', encoding="utf-8")

            write_json_atomic(path, {"status": "complete", "artifacts": [1, 2]})

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"status": "complete", "artifacts": [1, 2]},
            )
            self.assertFalse(path.with_suffix(".json.part").exists())

    def test_manifest_serialization_failure_preserves_previous_file(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "source_manifest.json"
            path.write_text('{"status":"valid"}', encoding="utf-8")
            path.with_suffix(".json.part").write_text("stale", encoding="utf-8")

            with self.assertRaises(TypeError):
                write_json_atomic(path, {"not_json_serializable": object()})

            self.assertEqual(path.read_text(encoding="utf-8"), '{"status":"valid"}')
            self.assertFalse(path.with_suffix(".json.part").exists())

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

    def test_incomplete_dpe_response_preserves_cached_snapshot(self) -> None:
        response = MagicMock()
        response.json.return_value = {
            "total": 2,
            "results": [{"numero_dpe": "partial", "identifiant_ban": "keep"}],
        }
        client = MagicMock()
        client.get.return_value = response

        with TemporaryDirectory() as directory:
            path = Path(directory) / "dpe.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as output:
                output.write('{"numero_dpe":"cached"}\n')

            with self.assertRaisesRegex(RuntimeError, "Incomplete DPE response"):
                acquire_dpe(client, path, force=True, ban_ids={"keep"})

            with gzip.open(path, "rt", encoding="utf-8") as source:
                records = [json.loads(line) for line in source]
            self.assertEqual(records, [{"numero_dpe": "cached"}])
            self.assertFalse(path.with_suffix(".gz.part").exists())

    def test_ban_source_is_pinned_to_its_archived_snapshot(self) -> None:
        self.assertNotIn("/latest/", BAN_URL)
        self.assertIn(f"/{BAN_SNAPSHOT_DATE}/", BAN_URL)
        self.assertEqual(BAN_PATH.parent.name, f"snapshot={BAN_SNAPSHOT_DATE}")
        self.assertRegex(BAN_SHA256, r"^[0-9a-f]{64}$")

    def test_bdnb_source_is_pinned_to_verified_release(self) -> None:
        self.assertIn(f"bdnb_millesime_{BDNB_RELEASE}/", BDNB_URL)
        self.assertEqual(BDNB_ARCHIVE.parent.name, f"release={BDNB_RELEASE}")
        self.assertRegex(BDNB_SHA256, r"^[0-9a-f]{64}$")
        self.assertRegex(BDNB_METADATA_SHA256, r"^[0-9a-f]{64}$")
        metadata = BDNB_METADATA_PATH.read_text(encoding="utf-8")
        self.assertIn(f"sha256: {BDNB_SHA256}", metadata)


if __name__ == "__main__":
    unittest.main()
