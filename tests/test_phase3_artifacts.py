"""Fast integrity checks for the delivered Phase 3 research artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRACKED_DELIVERABLES = [
    "docs/paper/phase3_paper.pdf",
    "reports/phase3/PHASE3_REPORT.md",
]
GENERATED_ARTIFACTS = [
    "data/gold/phase3_sale_features.parquet",
    "models/phase3/phase3_selected_model.joblib",
    "reports/phase3/feature_quality.json",
    "reports/phase3/phase3_results.json",
    "reports/phase3/source_manifest.json",
    "visuals/phase3/phase3_results_dashboard.png",
]


class Phase3TrackedDeliverableTests(unittest.TestCase):
    def test_tracked_deliverables_exist(self) -> None:
        for relative in TRACKED_DELIVERABLES:
            with self.subTest(path=relative):
                path = ROOT / relative
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)


class Phase3ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        missing = [path for path in GENERATED_ARTIFACTS if not (ROOT / path).is_file()]
        if missing:
            raise unittest.SkipTest(
                "generated Phase 3 artifacts are unavailable; run the Phase 3 "
                f"pipeline first (missing: {', '.join(missing)})"
            )
        cls.quality = json.loads(
            (ROOT / "reports/phase3/feature_quality.json").read_text(encoding="utf-8")
        )
        cls.results = json.loads(
            (ROOT / "reports/phase3/phase3_results.json").read_text(encoding="utf-8")
        )

    def test_gold_contract(self) -> None:
        self.assertEqual(self.quality["rows"], 143_009)
        self.assertEqual(self.quality["columns"], 141)
        self.assertEqual(self.quality["future_dpe_leakage_rows"], 0)
        self.assertGreater(self.quality["match_quality"]["ban_exact_match_rate"], 0.99)
        self.assertGreater(self.quality["match_quality"]["bdnb_building_match_rate"], 0.999)

    def test_chronological_gold_rows(self) -> None:
        columns = ["id_mutation", "date_mutation", "date_etablissement_dpe"]
        gold = pd.read_parquet(ROOT / "data/gold/phase3_sale_features.parquet", columns=columns)
        self.assertEqual(len(gold), 143_009)
        self.assertFalse(gold["id_mutation"].duplicated().any())
        sale_date = pd.to_datetime(gold["date_mutation"])
        dpe_date = pd.to_datetime(gold["date_etablissement_dpe"], errors="coerce")
        self.assertFalse((dpe_date.notna() & dpe_date.gt(sale_date)).any())

    def test_selected_model_contract(self) -> None:
        artifact = joblib.load(ROOT / "models/phase3/phase3_selected_model.joblib")
        self.assertEqual(artifact["model_family"], "catboost")
        self.assertEqual(artifact["training_period"], "2021-2024")
        self.assertEqual(artifact["test_period"], "2025")
        self.assertEqual(len(artifact["features"]), 100)

    def test_reported_improvement(self) -> None:
        winner = self.results["winner"]
        self.assertLess(winner["mae_eur"], self.results["reference_2025_mae_eur"])
        self.assertGreater(winner["reduction_ci95_lower_eur"], 0)

    def test_generated_artifacts_exist(self) -> None:
        for relative in GENERATED_ARTIFACTS:
            with self.subTest(path=relative):
                path = ROOT / relative
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
