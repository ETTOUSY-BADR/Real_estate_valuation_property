"""Nationwide geography, feature, and inference checks for Phase 4."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from paris_avm.features.phase4 import (
    PHASE4_MODEL_FEATURES,
    add_comparable_features,
    add_geography_features,
    load_and_clean_file,
)
from paris_avm.geography import (
    department_from_commune,
    normalize_department,
    region_from_department,
)
from paris_avm.inference.phase4 import parse_args
from paris_avm.modeling.benchmark_phase4 import hierarchical_baseline
from paris_avm.modeling.benchmark_phase4 import paired_bootstrap_summary
from paris_avm.modeling.benchmark_phase4 import safe_inverse_log_prediction


def raw_row(
    mutation: str,
    date: str,
    *,
    local_type: str = "2",
    department: str = "69",
    commune: str = "69123",
    postcode: str = "69002",
    latitude: float = 45.75,
    longitude: float = 4.83,
) -> dict[str, object]:
    return {
        "id_mutation": mutation,
        "date_mutation": date,
        "nature_mutation": "Vente",
        "valeur_fonciere": 240_000,
        "adresse_numero": 10,
        "adresse_suffixe": "",
        "adresse_nom_voie": "RUE TEST",
        "adresse_code_voie": "1234",
        "code_postal": postcode,
        "code_commune": commune,
        "code_departement": department,
        "id_parcelle": f"{commune}000AB0001",
        "code_type_local": local_type,
        "surface_reelle_bati": 60,
        "nombre_pieces_principales": 3,
        "nombre_lots": 1,
        "surface_terrain": np.nan,
        "longitude": longitude,
        "latitude": latitude,
    }


class Phase4Tests(unittest.TestCase):
    def test_department_and_region_mapping_supports_corsica_and_overseas(self) -> None:
        self.assertEqual(normalize_department("1"), "01")
        self.assertEqual(department_from_commune("2A004"), "2A")
        self.assertEqual(region_from_department("2B"), "94")
        self.assertEqual(department_from_commune("97105"), "971")
        self.assertEqual(region_from_department("971"), "01")

    def test_chunked_cleaner_keeps_national_sale_and_complete_mutations(self) -> None:
        rows = [
            raw_row("sale-lyon", "2022-01-01"),
            raw_row("mixed", "2022-01-02", local_type="2"),
            raw_row("mixed", "2022-01-02", local_type="1"),
            raw_row(
                "sale-reunion",
                "2022-01-03",
                department="974",
                commune="97411",
                postcode="97411",
                latitude=-21.01,
                longitude=55.27,
            ),
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "france_2022.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            result, counts = load_and_clean_file(path, read_chunk_rows=1)

        self.assertEqual(result["id_mutation"].tolist(), ["sale-lyon", "sale-reunion"])
        self.assertEqual(result["code_region"].tolist(), ["84", "04"])
        self.assertEqual(counts["raw_rows"], 4)
        self.assertEqual(counts["raw_mutations"], 3)

    def test_global_geography_features_are_finite_for_overseas_coordinates(self) -> None:
        data = pd.DataFrame(
            [
                {
                    **raw_row(
                        "sale-reunion",
                        "2022-01-03",
                        department="974",
                        commune="97411",
                        postcode="97411",
                        latitude=-21.01,
                        longitude=55.27,
                    ),
                    "code_region": "04",
                    "month": 1,
                    "day_of_year": 3,
                    "year": 2022,
                    "months_since_2021": 12,
                    "price_per_m2": 4_000,
                }
            ]
        )
        result = add_geography_features(data)
        self.assertTrue(np.isfinite(result[["geo_x", "geo_y", "geo_z"]]).all(axis=None))
        self.assertTrue(result.at[0, "h3_r8"])

    def test_national_comparables_obey_availability_lag(self) -> None:
        data = pd.DataFrame(
            {
                "date_mutation": pd.to_datetime(["2022-01-01", "2022-03-01", "2022-05-01"]),
                "latitude": [45.75, 45.75, 45.75],
                "longitude": [4.83, 4.83, 4.83],
                "price_per_m2": [4_000.0, 40_000.0, 5_000.0],
                "surface_reelle_bati": [60.0, 60.0, 60.0],
                "nombre_pieces_principales": [3.0, 3.0, 3.0],
                "code_departement": ["69", "69", "69"],
                "code_region": ["84", "84", "84"],
            }
        )
        result, _ = add_comparable_features(
            data, availability_lag_days=90, query_chunk_rows=1
        )
        self.assertEqual(result.at[1, "local_365d_250m_count"], 0)
        self.assertEqual(result.at[2, "local_365d_250m_count"], 1)
        self.assertEqual(result.at[2, "weighted_comp_ppm2"], 4_000.0)

    def test_model_contract_excludes_current_sale_price(self) -> None:
        self.assertNotIn("price_per_m2", PHASE4_MODEL_FEATURES)
        self.assertNotIn("valeur_fonciere", PHASE4_MODEL_FEATURES)
        self.assertIn("code_region", PHASE4_MODEL_FEATURES)

    def test_hierarchical_baseline_falls_back_from_unknown_postcode(self) -> None:
        reference = pd.DataFrame(
            {
                "code_postal": ["69001", "69002"],
                "code_commune": ["69123", "69123"],
                "code_departement": ["69", "69"],
                "code_region": ["84", "84"],
                "price_per_m2": [4_000.0, 6_000.0],
            }
        )
        target = pd.DataFrame(
            {
                "code_postal": ["69999"],
                "code_commune": ["69123"],
                "code_departement": ["69"],
                "code_region": ["84"],
                "surface_reelle_bati": [50.0],
            }
        )
        self.assertEqual(hierarchical_baseline(reference, target)[0], 250_000.0)

    def test_inference_accepts_non_paris_commune(self) -> None:
        args = parse_args(
            [
                "--surface", "60", "--rooms", "3", "--postal-code", "69002",
                "--commune-code", "69123", "--latitude", "45.75",
                "--longitude", "4.83", "--date", "2025-06-01",
            ]
        )
        self.assertEqual(department_from_commune(args.commune_code), "69")

    def test_paired_bootstrap_reports_positive_model_improvement(self) -> None:
        actual = np.linspace(100_000, 500_000, 200)
        model = actual + 5_000
        baseline = actual + 25_000

        result = paired_bootstrap_summary(
            actual, model, baseline, repetitions=50, maximum_rows=100
        )

        self.assertEqual(result["sample_rows"], 100)
        self.assertEqual(result["repetitions"], 50)
        self.assertGreater(result["reduction_ci95_lower_eur"], 0)

    def test_inverse_log_prediction_bounds_non_finite_outputs(self) -> None:
        values, clipped = safe_inverse_log_prediction(
            np.array([np.log1p(250_000), np.inf, -np.inf, np.nan])
        )
        self.assertAlmostEqual(values[0], 250_000)
        self.assertEqual(clipped, 3)
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(((values >= 10_000) & (values <= 10_000_000)).all())


if __name__ == "__main__":
    unittest.main()
