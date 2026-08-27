"""LaTeX and PDF contracts for the Phase 4 national report."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from paris_avm.reporting.phase4 import generate_latex


ROOT = Path(__file__).resolve().parents[1]


class Phase4ReportTests(unittest.TestCase):
    def test_repository_contains_compiled_empirical_report(self) -> None:
        latex = ROOT / "docs/paper/phase4_national_report.tex"
        pdf = ROOT / "docs/paper/phase4_national_report.pdf"
        self.assertTrue(latex.exists())
        self.assertTrue(pdf.exists())
        self.assertEqual(pdf.read_bytes()[:5], b"%PDF-")
        self.assertGreater(pdf.stat().st_size, 300_000)
        source = latex.read_text(encoding="utf-8")
        self.assertIn(r"\author{Badr Ettousy}", source)
        self.assertIn("Empirical benchmark edition", source)
        self.assertIn("EUR 41,936", source)
        self.assertIn("350,399 untouched 2025 apartment sales", source)
        self.assertNotIn("No accuracy number is asserted", source)

    def test_empirical_latex_is_populated_only_from_artifacts(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark_path = root / "benchmark.json"
            trials_path = root / "trials.csv"
            region_path = root / "region.csv"
            quality_path = root / "quality.json"
            output = root / "report.tex"
            metrics = {
                "mae_eur": 42_000.0,
                "rmse_eur": 70_000.0,
                "r2": 0.89,
                "median_absolute_percentage_error": 11.5,
                "within_20_percent": 0.76,
            }
            benchmark = {
                "rows": {"test": 12_345},
                "test_metrics": metrics,
                "baseline_test_metrics": {**metrics, "mae_eur": 50_000.0},
                "mae_reduction_vs_baseline_eur": 8_000.0,
                "prediction_interval": {"observed_test_coverage": 0.89},
                "paired_bootstrap": {
                    "reduction_ci95_lower_eur": 7_100.0,
                    "reduction_ci95_upper_eur": 8_900.0,
                },
            }
            benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "candidate": 1,
                        "configuration": '{"depth": 8, "loss_function": "MAE", "l2_leaf_reg": 8}',
                        "mae_eur": 43_000.0,
                        "median_absolute_percentage_error": 12.0,
                        "within_20_percent": 0.74,
                    }
                ]
            ).to_csv(trials_path, index=False)
            pd.DataFrame(
                [
                    {
                        "code_region": "84",
                        "region_name": "Auvergne-Rhône-Alpes",
                        "rows": 2_000,
                        "mae_eur": 40_000.0,
                        "median_absolute_percentage_error": 10.5,
                        "within_20_percent": 0.80,
                        "r2": 0.91,
                    }
                ]
            ).to_csv(region_path, index=False)
            quality_path.write_text(
                json.dumps(
                    {
                        "row_count": 60_000,
                        "date_min": "2021-01-01",
                        "date_max": "2025-12-31",
                        "observed_departments": ["69", "75"],
                        "observed_regions": ["84", "11"],
                    }
                ),
                encoding="utf-8",
            )

            source = generate_latex(
                output,
                benchmark_path=benchmark_path,
                trials_path=trials_path,
                region_path=region_path,
                quality_path=quality_path,
                visual_dir=root / "visuals",
                artifact_paths={"Benchmark": benchmark_path},
            )

            self.assertIn("Empirical benchmark edition", source)
            self.assertIn("EUR 42,000", source)
            self.assertIn("EUR 7,100--8,900", source)
            self.assertIn("Auvergne-Rhône-Alpes", source)
            self.assertNotIn("No accuracy number is asserted", source)


if __name__ == "__main__":
    unittest.main()
