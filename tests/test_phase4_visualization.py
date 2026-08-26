"""Artifact-backed reporting checks for Phase 4."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from paris_avm.features.phase4 import PHASE4_MODEL_FEATURES
from paris_avm.visualization import phase4


class Phase4VisualizationTests(unittest.TestCase):
    def test_help_does_not_render_or_load_artifacts(self) -> None:
        output = StringIO()
        with patch.object(phase4, "draw_architecture") as draw:
            with redirect_stdout(output), self.assertRaises(SystemExit) as raised:
                phase4.main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("allow-incomplete", output.getvalue())
        draw.assert_not_called()

    def test_incomplete_mode_labels_missing_results_without_fabrication(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "visuals"
            report = root / "PHASE4_REPORT.md"
            phase4.main(
                [
                    "--gold", str(root / "missing-gold.parquet"),
                    "--quality", str(root / "missing-quality.json"),
                    "--benchmark", str(root / "missing-benchmark.json"),
                    "--predictions", str(root / "missing-predictions.parquet"),
                    "--region-performance", str(root / "missing-region.csv"),
                    "--department-performance", str(root / "missing-department.csv"),
                    "--feature-importance", str(root / "missing-importance.csv"),
                    "--benchmark-comparison", str(root / "missing-comparison.csv"),
                    "--validation-trials", str(root / "missing-trials.csv"),
                    "--source-manifest", str(root / "missing-manifest.json"),
                    "--model", str(root / "missing-model.joblib"),
                    "--output-dir", str(output),
                    "--report", str(report),
                    "--allow-incomplete",
                ]
            )
            contents = report.read_text(encoding="utf-8")
            self.assertIn("no claimed Phase 4 accuracy figures", contents)
            self.assertIn("not generated", contents)
            self.assertNotIn("National MAE</", contents)
            self.assertGreater((output / "phase4_national_pipeline.png").stat().st_size, 10_000)

    def test_complete_synthetic_artifacts_generate_all_figures_and_metrics(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            report_dir = root / "reports"
            report_dir.mkdir()
            output = root / "visuals"
            rows = 120
            regions = np.where(np.arange(rows) % 2 == 0, "84", "11")
            departments = np.where(regions == "84", "69", "75")
            years = 2021 + np.arange(rows) % 5
            actual = np.linspace(90_000, 1_200_000, rows)
            predicted = actual * (1 + np.sin(np.arange(rows)) * 0.08)
            gold = pd.DataFrame(
                {
                    "id_mutation": [f"sale-{index}" for index in range(rows)],
                    "date_mutation": pd.to_datetime([f"{year}-06-15" for year in years]),
                    "year": years,
                    "code_region": regions,
                    "code_departement": departments,
                    "longitude": np.where(regions == "84", 4.83, 2.35) + np.linspace(-0.2, 0.2, rows),
                    "latitude": np.where(regions == "84", 45.75, 48.86) + np.linspace(-0.1, 0.1, rows),
                    "weighted_comp_ppm2": np.where(np.arange(rows) % 7 == 0, np.nan, 4_500),
                    "similar_weighted_comp_ppm2": np.where(np.arange(rows) % 5 == 0, np.nan, 4_400),
                    "local_365d_500m_count": np.arange(rows) % 40,
                }
            )
            gold_path = root / "gold.parquet"
            gold.to_parquet(gold_path, index=False)
            predictions = pd.DataFrame(
                {
                    "id_mutation": gold["id_mutation"],
                    "date_mutation": gold["date_mutation"],
                    "code_region": regions,
                    "code_departement": departments,
                    "code_commune": np.where(regions == "84", "69123", "75101"),
                    "code_postal": np.where(regions == "84", "69002", "75001"),
                    "surface_reelle_bati": np.linspace(20, 110, rows),
                    "valeur_fonciere": actual,
                    "predicted_value": predicted,
                    "absolute_error": np.abs(actual - predicted),
                    "absolute_percentage_error": np.abs(actual - predicted) / actual * 100,
                }
            )
            predictions_path = report_dir / "predictions.parquet"
            predictions.to_parquet(predictions_path, index=False)
            metrics = {
                "mae_eur": 42_000.0,
                "rmse_eur": 70_000.0,
                "r2": 0.89,
                "mape_percent": 15.0,
                "median_absolute_percentage_error": 11.5,
                "within_10_percent": 0.46,
                "within_20_percent": 0.76,
            }
            baseline = {**metrics, "mae_eur": 50_000.0, "rmse_eur": 80_000.0, "r2": 0.85, "within_20_percent": 0.70}
            benchmark = {
                "protocol": {"untouched_test": 2025},
                "rows": {"test": rows},
                "test_metrics": metrics,
                "baseline_test_metrics": baseline,
                "mae_reduction_vs_baseline_eur": 8_000.0,
                "prediction_interval": {"observed_test_coverage": 0.89},
            }
            benchmark_path = report_dir / "benchmark.json"
            benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
            quality = {
                "row_count": rows,
                "date_min": "2021-06-15",
                "date_max": "2025-06-15",
                "observed_departments": ["69", "75"],
                "observed_regions": ["84", "11"],
            }
            quality_path = report_dir / "quality.json"
            quality_path.write_text(json.dumps(quality), encoding="utf-8")
            region = pd.DataFrame(
                {
                    "code_region": ["84", "11"],
                    "region_name": ["Auvergne-Rhône-Alpes", "Île-de-France"],
                    "rows": [60, 60],
                    "mae_eur": [38_000.0, 46_000.0],
                    "rmse_eur": [60_000.0, 78_000.0],
                    "r2": [0.90, 0.87],
                    "mape_percent": [13.0, 17.0],
                    "median_absolute_percentage_error": [10.0, 13.0],
                    "within_10_percent": [0.50, 0.42],
                    "within_20_percent": [0.80, 0.72],
                }
            )
            region_path = report_dir / "region.csv"
            region.to_csv(region_path, index=False)
            department = region.rename(columns={"code_region": "code_departement"}).copy()
            department["code_departement"] = ["69", "75"]
            department_path = report_dir / "department.csv"
            department.to_csv(department_path, index=False)
            importance_path = report_dir / "importance.csv"
            pd.DataFrame(
                {
                    "feature": list(PHASE4_MODEL_FEATURES),
                    "importance": np.linspace(10, 1, len(PHASE4_MODEL_FEATURES)),
                }
            ).to_csv(importance_path, index=False)
            comparison_path = report_dir / "comparison.csv"
            pd.DataFrame(
                [
                    {"method": "Phase 4 CatBoost", **metrics},
                    {"method": "Hierarchical geographic median baseline", **baseline},
                ]
            ).to_csv(comparison_path, index=False)
            trials_path = report_dir / "trials.csv"
            pd.DataFrame(
                [
                    {
                        "candidate": 1,
                        "configuration": '{"depth": 8, "loss_function": "MAE"}',
                        **metrics,
                    },
                    {
                        "candidate": 2,
                        "configuration": '{"depth": 9, "loss_function": "Huber:delta=1.0"}',
                        **baseline,
                    },
                ]
            ).to_csv(trials_path, index=False)
            manifest_path = root / "manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")
            model_path = root / "model.joblib"
            model_path.write_bytes(b"synthetic model artifact")
            report_path = report_dir / "PHASE4_REPORT.md"

            phase4.main(
                [
                    "--gold", str(gold_path),
                    "--quality", str(quality_path),
                    "--benchmark", str(benchmark_path),
                    "--predictions", str(predictions_path),
                    "--region-performance", str(region_path),
                    "--department-performance", str(department_path),
                    "--feature-importance", str(importance_path),
                    "--benchmark-comparison", str(comparison_path),
                    "--validation-trials", str(trials_path),
                    "--source-manifest", str(manifest_path),
                    "--model", str(model_path),
                    "--output-dir", str(output),
                    "--report", str(report_path),
                ]
            )
            expected = {
                "phase4_national_pipeline.png",
                "phase4_national_coverage.png",
                "phase4_performance_dashboard.png",
                "phase4_prediction_diagnostics.png",
                "phase4_regional_performance.png",
                "phase4_feature_importance.png",
                "phase4_comparable_coverage.png",
            }
            self.assertEqual({path.name for path in output.glob("*.png")}, expected)
            self.assertTrue(all((output / name).stat().st_size > 10_000 for name in expected))
            contents = report_path.read_text(encoding="utf-8")
            self.assertIn("EUR 42,000 MAE", contents)
            self.assertIn("76.00%", contents)
            self.assertIn("complete empirical report", contents)


if __name__ == "__main__":
    unittest.main()
