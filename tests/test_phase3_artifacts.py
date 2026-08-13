"""Fast integrity checks for the delivered Phase 3 research artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import joblib
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
TRACKED_DELIVERABLES = [
    "docs/paper/phase3_paper.pdf",
    "reports/phase3/PHASE3_REPORT.md",
]
GENERATED_ARTIFACTS = [
    "data/gold/phase3_sale_features.parquet",
    "data/silver/phase3_match_audit.parquet",
    "models/phase3/phase3_selected_model.joblib",
    "reports/phase3/feature_quality.json",
    "reports/phase3/phase3_results.json",
    "reports/phase3/source_manifest.json",
    "visuals/phase3/phase3_results_dashboard.png",
]


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


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
        cls.manifest = json.loads(
            (ROOT / "reports/phase3/source_manifest.json").read_text(encoding="utf-8")
        )

    def test_gold_contract(self) -> None:
        self.assertEqual(self.quality["rows"], 143_009)
        self.assertEqual(self.quality["columns"], 141)
        self.assertEqual(self.quality["future_dpe_leakage_rows"], 0)
        self.assertGreater(self.quality["match_quality"]["ban_exact_match_rate"], 0.99)
        self.assertGreater(self.quality["match_quality"]["bdnb_building_match_rate"], 0.999)

    def test_gold_matches_recorded_digest(self) -> None:
        gold_path = ROOT / "data/gold/phase3_sale_features.parquet"
        self.assertEqual(sha256(gold_path), self.quality["output_sha256"])

    def test_source_manifest_matches_local_snapshots(self) -> None:
        self.assertEqual(self.manifest["schema_version"], 1)
        artifacts = self.manifest["artifacts"]
        self.assertGreater(len(artifacts), 0)
        self.assertEqual(len({item["source_id"] for item in artifacts}), len(artifacts))
        self.assertEqual(len({item["path"] for item in artifacts}), len(artifacts))

        bronze_root = (ROOT / "data/bronze").resolve()
        for artifact in artifacts:
            with self.subTest(source_id=artifact["source_id"]):
                path = (ROOT / artifact["path"]).resolve()
                self.assertTrue(path.is_relative_to(bronze_root))
                self.assertTrue(path.is_file())
                self.assertEqual(path.stat().st_size, artifact["bytes"])
                self.assertEqual(sha256(path), artifact["sha256"])

    def test_chronological_gold_rows(self) -> None:
        columns = ["id_mutation", "date_mutation", "date_etablissement_dpe"]
        gold = pd.read_parquet(ROOT / "data/gold/phase3_sale_features.parquet", columns=columns)
        self.assertEqual(len(gold), 143_009)
        self.assertFalse(gold["id_mutation"].duplicated().any())
        sale_date = pd.to_datetime(gold["date_mutation"])
        dpe_date = pd.to_datetime(gold["date_etablissement_dpe"], errors="coerce")
        self.assertFalse((dpe_date.notna() & dpe_date.gt(sale_date)).any())

    def test_match_audit_covers_every_gold_transaction(self) -> None:
        audit_path = ROOT / "data/silver/phase3_match_audit.parquet"
        audit = pd.read_parquet(audit_path)
        gold_ids = pd.read_parquet(
            ROOT / "data/gold/phase3_sale_features.parquet", columns=["id_mutation"]
        )["id_mutation"]
        required_columns = {
            "id_mutation",
            "ban_match_method",
            "ban_match_distance_m",
            "ban_match_confidence",
            "bdnb_match_method",
            "bdnb_match_confidence",
            "bdnb_address_candidate_count",
            "bdnb_parcel_candidate_count",
        }

        self.assertTrue(required_columns.issubset(audit.columns))
        self.assertFalse(audit["id_mutation"].isna().any())
        self.assertFalse(audit["id_mutation"].duplicated().any())
        self.assertTrue(audit["id_mutation"].equals(gold_ids))

    def test_selected_model_contract(self) -> None:
        artifact = joblib.load(ROOT / "models/phase3/phase3_selected_model.joblib")
        self.assertEqual(artifact["model_family"], "catboost")
        self.assertEqual(artifact["training_period"], "2021-2024")
        self.assertEqual(artifact["test_period"], "2025")
        features = artifact["features"]
        categorical = artifact["categorical_features"]
        self.assertEqual(len(features), 100)
        self.assertEqual(len(features), len(set(features)))
        self.assertEqual(len(categorical), len(set(categorical)))
        self.assertTrue(set(categorical).issubset(features))

        gold_columns = set(
            pq.read_schema(ROOT / "data/gold/phase3_sale_features.parquet").names
        )
        self.assertTrue(set(features).issubset(gold_columns))
        self.assertEqual(list(artifact["model"].feature_names_), features)
        expected_categorical_indices = [features.index(name) for name in categorical]
        self.assertEqual(
            artifact["model"].get_cat_feature_indices(), expected_categorical_indices
        )

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
