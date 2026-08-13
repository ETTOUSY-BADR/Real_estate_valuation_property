"""Unit tests for Phase 3 source acquisition and provenance."""

from __future__ import annotations

from contextlib import redirect_stdout
import gzip
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock

from paris_avm.data.acquire_phase3 import acquire_dpe


class Phase3AcquisitionTests(unittest.TestCase):
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
