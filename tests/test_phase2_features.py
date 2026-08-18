"""Phase 2 feature-build contract and publication tests."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from paris_avm.features import phase2


class Phase2FeatureTests(unittest.TestCase):
    @staticmethod
    def valid_frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "id_mutation": ["sale-1", "sale-2"],
                "date_mutation": pd.to_datetime(["2021-01-04", "2022-02-05"]),
                "feature": [1.5, 2.5],
            }
        )

    def run_main(
        self,
        directory: str,
        frame: pd.DataFrame,
        *,
        extra_arguments: list[str] | None = None,
    ) -> tuple[Path, Path, Path]:
        root = Path(directory)
        source = root / "source.csv"
        source.write_bytes(b"source snapshot\n")
        output = root / "nested" / "phase2.parquet"
        report = root / "reports" / "quality.json"
        arguments = [
            "phase2",
            "--data",
            str(source),
            "--output",
            str(output),
            "--quality-report",
            str(report),
            *(extra_arguments or []),
        ]
        counts = {"input_rows": len(frame)}
        comparable_quality = {"availability_lag_days": 90}

        with (
            patch.object(sys, "argv", arguments),
            patch.object(phase2, "load_and_clean_many", return_value=(frame, counts)),
            patch.object(
                phase2, "add_identity_and_spatial_features", side_effect=lambda data: data
            ),
            patch.object(
                phase2,
                "add_comparable_features",
                return_value=(frame, comparable_quality),
            ),
            patch.object(phase2, "validate_phase2_table"),
            redirect_stdout(StringIO()),
        ):
            phase2.main()
        return source, output, report

    def test_main_publishes_gold_table_with_matching_checksum(self) -> None:
        with TemporaryDirectory() as directory:
            source, output, report_path = self.run_main(
                directory, self.valid_frame()
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))
            with output.open("rb") as handle:
                output_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
            with source.open("rb") as handle:
                source_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()

            self.assertEqual(report["output_sha256"], output_sha256)
            self.assertEqual(report["source_files"][0]["sha256"], source_sha256)
            self.assertEqual(report["source_files"][0]["bytes"], source.stat().st_size)
            self.assertEqual(report["row_count"], 2)
            self.assertEqual(report["feature_columns"], list(self.valid_frame().columns))
            pd.testing.assert_frame_equal(pd.read_parquet(output), self.valid_frame())
            self.assertFalse(output.with_suffix(".parquet.part").exists())
            self.assertFalse(report_path.with_suffix(".json.part").exists())

    def test_main_passes_custom_lag_and_chunk_size(self) -> None:
        with TemporaryDirectory() as directory:
            frame = self.valid_frame()
            root = Path(directory)
            source = root / "source.csv"
            source.write_text("source", encoding="utf-8")
            output = root / "phase2.parquet"
            report = root / "quality.json"
            arguments = [
                "phase2",
                "--data",
                str(source),
                "--output",
                str(output),
                "--quality-report",
                str(report),
                "--availability-lag-days",
                "120",
                "--chunk-size",
                "64",
            ]

            with (
                patch.object(sys, "argv", arguments),
                patch.object(
                    phase2, "load_and_clean_many", return_value=(frame, {})
                ),
                patch.object(
                    phase2,
                    "add_identity_and_spatial_features",
                    side_effect=lambda data: data,
                ),
                patch.object(
                    phase2,
                    "add_comparable_features",
                    return_value=(frame, {}),
                ) as comparable,
                patch.object(phase2, "validate_phase2_table"),
                redirect_stdout(StringIO()),
            ):
                phase2.main()

            comparable.assert_called_once_with(
                frame, availability_lag_days=120, chunk_size=64
            )

    def test_duplicate_mutation_ids_are_rejected_before_publication(self) -> None:
        frame = self.valid_frame()
        frame.loc[1, "id_mutation"] = frame.loc[0, "id_mutation"]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "duplicate mutation"):
                self.run_main(directory, frame)
            self.assertFalse((root / "nested" / "phase2.parquet").exists())
            self.assertFalse((root / "reports" / "quality.json").exists())

    def test_nonchronological_rows_are_rejected_before_publication(self) -> None:
        frame = self.valid_frame().iloc[::-1].reset_index(drop=True)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "not chronologically ordered"):
                self.run_main(directory, frame)
            self.assertFalse((root / "nested" / "phase2.parquet").exists())
            self.assertFalse((root / "reports" / "quality.json").exists())

    def test_nonpositive_availability_lag_is_rejected_before_loading(self) -> None:
        with (
            patch.object(
                sys,
                "argv",
                ["phase2", "--availability-lag-days", "0"],
            ),
            patch.object(phase2, "load_and_clean_many") as load,
            redirect_stderr(StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            phase2.main()

        self.assertEqual(raised.exception.code, 2)
        load.assert_not_called()

    def test_nonpositive_chunk_size_is_rejected_before_loading(self) -> None:
        with (
            patch.object(sys, "argv", ["phase2", "--chunk-size", "0"]),
            patch.object(phase2, "load_and_clean_many") as load,
            redirect_stderr(StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            phase2.main()

        self.assertEqual(raised.exception.code, 2)
        load.assert_not_called()

    def test_spatial_features_include_metric_coordinates_and_h3_cells(self) -> None:
        data = pd.DataFrame(
            {
                "longitude": [2.3499],
                "latitude": [48.8530],
                "surface_reelle_bati": [40.0],
                "nombre_pieces_principales": [2.0],
                "code_commune": ["75101"],
                "adresse_code_voie": ["1234"],
                "adresse_numero": [10.0],
                "adresse_suffixe": [" B "],
                "id_parcelle": ["75101000AB0001"],
            }
        )

        result = phase2.add_identity_and_spatial_features(data)

        self.assertLess(result.loc[0, "distance_paris_center_m"], 0.01)
        self.assertAlmostEqual(result.loc[0, "log_surface"], np.log1p(40.0))
        self.assertAlmostEqual(result.loc[0, "rooms_per_10m2"], 0.5)
        self.assertEqual(result.loc[0, "address_id"], "75101_1234_10_b")
        for resolution in (8, 9, 10):
            self.assertTrue(result.loc[0, f"h3_r{resolution}"])

    def test_dvf_cleaning_keeps_only_valid_single_apartment_sales(self) -> None:
        rows = [
            {
                "id_mutation": "sale-1",
                "date_mutation": "2022-04-15",
                "nature_mutation": "Vente",
                "valeur_fonciere": 400_000,
                "adresse_numero": 12,
                "adresse_suffixe": "",
                "adresse_nom_voie": "RUE TEST",
                "adresse_code_voie": "1234",
                "code_postal": 75001,
                "code_commune": "75101",
                "id_parcelle": "parcel-1",
                "code_type_local": "2",
                "surface_reelle_bati": 40,
                "nombre_pieces_principales": 2,
                "nombre_lots": 1,
                "surface_terrain": np.nan,
                "longitude": 2.34,
                "latitude": 48.85,
            },
            {
                "id_mutation": "sale-2",
                "date_mutation": "2022-04-16",
                "nature_mutation": "Vente",
                "valeur_fonciere": 600_000,
                "adresse_numero": 14,
                "adresse_suffixe": "",
                "adresse_nom_voie": "RUE TEST",
                "adresse_code_voie": "1234",
                "code_postal": 75001,
                "code_commune": "75101",
                "id_parcelle": "parcel-2",
                "code_type_local": "1",
                "surface_reelle_bati": 80,
                "nombre_pieces_principales": 4,
                "nombre_lots": 1,
                "surface_terrain": 20,
                "longitude": 2.34,
                "latitude": 48.85,
            },
            {
                "id_mutation": "sale-2",
                "date_mutation": "2022-04-16",
                "nature_mutation": "Vente",
                "valeur_fonciere": 600_000,
                "adresse_numero": 14,
                "adresse_suffixe": "",
                "adresse_nom_voie": "RUE TEST",
                "adresse_code_voie": "1234",
                "code_postal": 75001,
                "code_commune": "75101",
                "id_parcelle": "parcel-2",
                "code_type_local": "2",
                "surface_reelle_bati": 45,
                "nombre_pieces_principales": 2,
                "nombre_lots": 1,
                "surface_terrain": np.nan,
                "longitude": 2.34,
                "latitude": 48.85,
            },
            {
                "id_mutation": "donation-1",
                "date_mutation": "2022-04-17",
                "nature_mutation": "Donation",
                "valeur_fonciere": 300_000,
                "adresse_numero": 16,
                "adresse_suffixe": "",
                "adresse_nom_voie": "RUE TEST",
                "adresse_code_voie": "1234",
                "code_postal": 75001,
                "code_commune": "75101",
                "id_parcelle": "parcel-3",
                "code_type_local": "2",
                "surface_reelle_bati": 30,
                "nombre_pieces_principales": 1,
                "nombre_lots": 1,
                "surface_terrain": np.nan,
                "longitude": 2.34,
                "latitude": 48.85,
            },
        ]
        with TemporaryDirectory() as directory:
            source = Path(directory) / "75_2022.csv"
            pd.DataFrame(rows).to_csv(source, index=False)

            result, counts = phase2.load_and_clean_many([source])

        self.assertEqual(result["id_mutation"].tolist(), ["sale-1"])
        self.assertEqual(result.loc[0, "price_per_m2"], 10_000.0)
        self.assertEqual(result.loc[0, "month"], 4)
        self.assertEqual(result.loc[0, "year"], 2022)
        self.assertEqual(result.loc[0, "months_since_2021"], 15)
        self.assertEqual(counts["clean_model_rows"], 1)

    def test_comparables_use_only_sales_older_than_availability_lag(self) -> None:
        data = pd.DataFrame(
            {
                "date_mutation": pd.to_datetime(
                    ["2022-01-01", "2022-03-01", "2022-05-01"]
                ),
                "x_l93": [650_000.0, 650_000.0, 650_000.0],
                "y_l93": [6_860_000.0, 6_860_000.0, 6_860_000.0],
                "price_per_m2": [5_000.0, 50_000.0, 7_000.0],
                "surface_reelle_bati": [40.0, 40.0, 40.0],
                "nombre_pieces_principales": [2.0, 2.0, 2.0],
                "year": [2022, 2022, 2022],
            }
        )

        result, quality = phase2.add_comparable_features(
            data, availability_lag_days=90, chunk_size=2
        )

        self.assertEqual(result.loc[0, "local_365d_250m_count"], 0)
        self.assertEqual(result.loc[1, "local_365d_250m_count"], 0)
        self.assertEqual(result.loc[2, "local_365d_250m_count"], 1)
        self.assertEqual(result.loc[2, "local_365d_250m_median_ppm2"], 5_000.0)
        self.assertEqual(result.loc[2, "weighted_comp_ppm2"], 5_000.0)
        self.assertEqual(quality["minimum_observed_comparable_age_days"], 90)

    def test_comparable_counts_never_include_the_current_transaction(self) -> None:
        data = pd.DataFrame(
            {
                "date_mutation": pd.to_datetime(["2022-05-01"]),
                "x_l93": [650_000.0],
                "y_l93": [6_860_000.0],
                "price_per_m2": [9_000.0],
                "surface_reelle_bati": [50.0],
                "nombre_pieces_principales": [2.0],
                "year": [2022],
            }
        )

        result, _ = phase2.add_comparable_features(
            data, availability_lag_days=1, chunk_size=1
        )

        count_columns = [
            column
            for column in result
            if column.startswith("local_") and column.endswith("_count")
        ]
        self.assertTrue((result[count_columns] == 0).all(axis=None))
        self.assertTrue(pd.isna(result.loc[0, "weighted_comp_ppm2"]))


if __name__ == "__main__":
    unittest.main()
