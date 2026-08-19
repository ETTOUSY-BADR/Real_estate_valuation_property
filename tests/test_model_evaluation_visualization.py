"""Tests for frozen-test model evaluation tables."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import pandas as pd

from paris_avm.visualization.model_evaluation import (
    PHASE2_PREDICTION,
    PHASE3_PREDICTION,
    add_error_columns,
    build_segment_metrics,
)


class ModelEvaluationVisualizationTests(unittest.TestCase):
    @staticmethod
    def predictions() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "id_mutation": ["a", "b", "c", "d"],
                "date_mutation": pd.to_datetime(
                    ["2025-01-10", "2025-01-20", "2025-02-10", "2025-02-20"]
                ),
                "code_postal": [75001, 75001, 75002, 75002],
                "surface_reelle_bati": [15.0, 35.0, 55.0, 130.0],
                "valeur_fonciere": [200_000.0, 400_000.0, 600_000.0, 2_500_000.0],
                PHASE2_PREDICTION: [240_000.0, 450_000.0, 500_000.0, 2_000_000.0],
                PHASE3_PREDICTION: [210_000.0, 420_000.0, 580_000.0, 2_400_000.0],
            }
        )

    @staticmethod
    def gold() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "id_mutation": ["a", "b", "c", "d"],
                "local_365d_500m_count": [0, 25, 150, 600],
                "dpe_available_at_sale": [0, 1, 1, 0],
            }
        )

    def test_error_columns_are_paired_on_each_transaction(self) -> None:
        source = self.predictions()

        result = add_error_columns(source)

        self.assertNotIn("phase2_absolute_error", source)
        self.assertEqual(result["phase2_absolute_error"].tolist(), [40_000, 50_000, 100_000, 500_000])
        self.assertEqual(result["phase3_absolute_error"].tolist(), [10_000, 20_000, 20_000, 100_000])
        self.assertEqual(result["paired_improvement_eur"].tolist(), [30_000, 30_000, 80_000, 400_000])

    def test_segment_metrics_cover_every_requested_dimension(self) -> None:
        segments, calibration = build_segment_metrics(
            self.predictions(), self.gold()
        )

        self.assertEqual(
            set(segments["dimension"]),
            {
                "month",
                "arrondissement",
                "surface_band",
                "value_band",
                "comparable_density",
                "dpe_status",
            },
        )
        january = segments.loc[
            segments["dimension"].eq("month") & segments["segment"].eq("Jan")
        ].iloc[0]
        self.assertEqual(january["transactions"], 2)
        self.assertEqual(january["phase2_mae_eur"], 45_000.0)
        self.assertEqual(january["phase3_mae_eur"], 15_000.0)
        self.assertEqual(january["mae_reduction_eur"], 30_000.0)
        self.assertEqual(len(calibration), 10)
        self.assertEqual(int(calibration["transactions"].sum()), 4)

    def test_duplicate_gold_identity_is_rejected(self) -> None:
        gold = pd.concat([self.gold(), self.gold().iloc[[0]]], ignore_index=True)

        with self.assertRaises(pd.errors.MergeError):
            build_segment_metrics(self.predictions(), gold)

    def test_latex_and_pdf_report_are_published(self) -> None:
        root = Path(__file__).resolve().parents[1]
        latex = root / "docs/paper/model_evaluation_report.tex"
        pdf = root / "docs/paper/model_evaluation_report.pdf"

        self.assertTrue(latex.exists())
        self.assertTrue(pdf.exists())
        self.assertGreater(pdf.stat().st_size, 1_000_000)
        self.assertEqual(pdf.read_bytes()[:5], b"%PDF-")
        source = latex.read_text(encoding="utf-8")
        self.assertIn(r"\author{Badr Ettousy}", source)
        self.assertIn("model_evaluation_segment_dashboard.png", source)

    def test_report_metrics_match_refreshed_results(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (root / "docs/paper/model_evaluation_report.tex").read_text(
            encoding="utf-8"
        )
        results = json.loads(
            (root / "reports/phase3/phase3_results.json").read_text(encoding="utf-8")
        )
        winner = results["winner"]

        self.assertIn(f"EUR {winner['mae_eur']:,.0f}", source)
        self.assertIn(
            f"EUR {winner['mae_reduction_vs_baseline_eur']:,.0f}", source
        )
        self.assertIn(f"{winner['r2']:.4f}", source)

    def test_every_dedicated_report_figure_exists(self) -> None:
        root = Path(__file__).resolve().parents[1]
        figure_dir = root / "visuals/model_evaluation"
        expected = {
            "model_evaluation_feature_contract.png",
            "model_evaluation_performance_ladder.png",
            "model_evaluation_segment_dashboard.png",
            "model_evaluation_coverage_diagnostics.png",
        }

        self.assertEqual(
            {path.name for path in figure_dir.glob("*.png")}, expected
        )
        for filename in expected:
            self.assertGreater((figure_dir / filename).stat().st_size, 100_000)


if __name__ == "__main__":
    unittest.main()
