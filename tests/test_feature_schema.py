"""Tests for the model-ready preprocessing feature contract."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from paris_avm.features.schema import (
    CURRENT_TRANSACTION_DERIVED_COLUMNS,
    PHASE2_FEATURE_GROUPS,
    PHASE2_MODEL_FEATURES,
    PHASE3_FEATURE_GROUPS,
    PHASE3_MODEL_FEATURES,
    TARGET_COLUMN,
    FeatureContractError,
    contract_summary,
    validate_phase2_table,
    validate_phase3_table,
)
from paris_avm.modeling.benchmark_phase2 import FEATURE_SETS as PHASE2_FEATURE_SETS
from paris_avm.modeling.benchmark_phase3 import FEATURE_SETS as PHASE3_FEATURE_SETS


def valid_phase2_frame() -> pd.DataFrame:
    values: dict[str, list[object]] = {
        feature: [np.nan, np.nan] for feature in PHASE2_MODEL_FEATURES
    }
    values.update(
        {
            "id_mutation": ["sale-1", "sale-2"],
            "date_mutation": pd.to_datetime(["2022-01-01", "2022-06-01"]),
            TARGET_COLUMN: [300_000.0, 450_000.0],
            "surface_reelle_bati": [30.0, 45.0],
            "log_surface": np.log1p([30.0, 45.0]).tolist(),
            "month": [1, 6],
            "day_of_year": [1, 152],
            "year": [2022, 2022],
            "months_since_2021": [12, 17],
            "latitude": [48.85, 48.86],
            "longitude": [2.34, 2.35],
            "x_l93": [651_000.0, 652_000.0],
            "y_l93": [6_861_000.0, 6_862_000.0],
            "distance_paris_center_m": [500.0, 1_000.0],
            "bearing_sin": [0.1, 0.2],
            "bearing_cos": [0.9, 0.8],
            "code_postal": ["75001", "75002"],
            "h3_r8": ["r8-a", "r8-b"],
            "h3_r9": ["r9-a", "r9-b"],
        }
    )
    return pd.DataFrame(values)


class FeatureSchemaTests(unittest.TestCase):
    def test_contract_has_expected_feature_groups_and_counts(self) -> None:
        summary = contract_summary()

        self.assertEqual(summary["phase2_feature_count"], 44)
        self.assertEqual(summary["phase3_feature_count"], 100)
        self.assertEqual(
            set(summary["phase2_groups"]),
            {"property", "time", "coordinates", "spatial", "historical_comparables", "categorical"},
        )
        self.assertEqual(
            set(summary["phase3_groups"]),
            {"building", "dpe_point_in_time", "neighborhood_context", "external_identity"},
        )

    def test_model_contract_excludes_target_and_current_sale_price_per_m2(self) -> None:
        forbidden = {TARGET_COLUMN, *CURRENT_TRANSACTION_DERIVED_COLUMNS}

        self.assertTrue(forbidden.isdisjoint(PHASE2_MODEL_FEATURES))
        self.assertTrue(forbidden.isdisjoint(PHASE3_MODEL_FEATURES))
        self.assertEqual(len(PHASE3_MODEL_FEATURES), len(set(PHASE3_MODEL_FEATURES)))

    def test_contract_order_matches_benchmark_feature_order(self) -> None:
        phase2_numeric, phase2_categorical = PHASE2_FEATURE_SETS["full"]
        phase3_numeric, phase3_categorical = PHASE3_FEATURE_SETS["full_identity"]

        self.assertEqual(tuple(phase2_numeric + phase2_categorical), PHASE2_MODEL_FEATURES)
        self.assertEqual(tuple(phase3_numeric + phase3_categorical), PHASE3_MODEL_FEATURES)

    def test_valid_phase2_table_is_accepted(self) -> None:
        validate_phase2_table(valid_phase2_frame())

    def test_missing_feature_identifies_the_column(self) -> None:
        data = valid_phase2_frame().drop(columns="h3_r9")

        with self.assertRaisesRegex(FeatureContractError, "h3_r9"):
            validate_phase2_table(data)

    def test_duplicate_mutation_is_rejected(self) -> None:
        data = valid_phase2_frame()
        data.loc[1, "id_mutation"] = "sale-1"

        with self.assertRaisesRegex(FeatureContractError, "unique"):
            validate_phase2_table(data)

    def test_out_of_bounds_surface_is_rejected(self) -> None:
        data = valid_phase2_frame()
        data.loc[0, "surface_reelle_bati"] = 8.0

        with self.assertRaisesRegex(FeatureContractError, "surfaces"):
            validate_phase2_table(data)

    def test_negative_comparable_count_is_rejected(self) -> None:
        data = valid_phase2_frame()
        data.loc[0, "local_365d_500m_count"] = -1

        with self.assertRaisesRegex(FeatureContractError, "cannot be negative"):
            validate_phase2_table(data)

    def test_future_dpe_is_rejected(self) -> None:
        data = valid_phase2_frame()
        for features in PHASE3_FEATURE_GROUPS.values():
            for feature in features:
                data[feature] = np.nan
        data["date_etablissement_dpe"] = pd.to_datetime(["2021-12-01", "2022-07-01"])

        with self.assertRaisesRegex(FeatureContractError, "after the sale"):
            validate_phase3_table(data)


if __name__ == "__main__":
    unittest.main()
