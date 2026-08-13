"""Command-line validation for Phase 3 single-property inference."""

from __future__ import annotations

import contextlib
import io
import unittest

import pandas as pd

from paris_avm.inference.phase3 import (
    add_base_and_comparable_features,
    normalize_address_suffix,
    normalized_address_id,
    parse_args,
    validate_coordinate_consistency,
)


VALID_ARGUMENTS = [
    "--surface",
    "58",
    "--rooms",
    "3",
    "--postal-code",
    "75008",
    "--commune-code",
    "75108",
    "--street-code",
    "2576",
    "--address-number",
    "29",
    "--latitude",
    "48.878673",
    "--longitude",
    "2.302806",
    "--lots",
    "2",
    "--date",
    "2025-01-02",
]


class Phase3CliTests(unittest.TestCase):
    def parse_error(self, arguments: list[str]) -> str:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            parse_args(arguments)
        return stderr.getvalue()

    def test_documented_example_is_valid(self) -> None:
        args = parse_args(VALID_ARGUMENTS)
        self.assertEqual(args.date.strftime("%Y-%m-%d"), "2025-01-02")
        self.assertEqual(args.street_code, "2576")
        self.assertEqual(args.address_number, 29)
        self.assertEqual(normalized_address_id(args), "75108_2576_29_")

    def test_fractional_address_number_is_rejected(self) -> None:
        arguments = ["29.5" if value == "29" else value for value in VALID_ARGUMENTS]
        message = self.parse_error(arguments)
        self.assertIn("positive whole number", message)

    def test_nonpositive_address_number_is_rejected(self) -> None:
        arguments = ["0" if value == "29" else value for value in VALID_ARGUMENTS]
        message = self.parse_error(arguments)
        self.assertIn("positive whole number", message)

    def test_conflicting_arrondissement_codes_are_rejected(self) -> None:
        arguments = [
            "75020" if value == "75008" else value for value in VALID_ARGUMENTS
        ]
        message = self.parse_error(arguments)
        self.assertIn("must identify the same arrondissement", message)

    def test_common_address_suffixes_use_canonical_codes(self) -> None:
        aliases = {"BIS": "b", "ter": "t", " Quater ": "q", "A": "a"}
        for supplied, expected in aliases.items():
            with self.subTest(suffix=supplied):
                self.assertEqual(normalize_address_suffix(supplied), expected)

    def test_suffix_alias_is_used_in_address_identity(self) -> None:
        arguments = VALID_ARGUMENTS + ["--address-suffix", "BIS"]
        args = parse_args(arguments)
        self.assertEqual(args.address_suffix, "b")
        self.assertEqual(normalized_address_id(args), "75108_2576_29_b")

    def test_documented_coordinates_match_resolved_address(self) -> None:
        args = parse_args(VALID_ARGUMENTS)
        template = pd.Series(
            {
                "ban_x": 648_873.72,
                "ban_y": 6_864_512.73,
                "x_l93": 648_866.60,
                "y_l93": 6_864_520.34,
            }
        )
        self.assertLess(validate_coordinate_consistency(args, template), 25)

    def test_coordinates_far_from_resolved_address_are_rejected(self) -> None:
        arguments = [
            "48.850000" if value == "48.878673" else value
            for value in VALID_ARGUMENTS
        ]
        args = parse_args(arguments)
        template = pd.Series({"ban_x": 648_873.72, "ban_y": 6_864_512.73})
        with self.assertRaisesRegex(SystemExit, "from the resolved address"):
            validate_coordinate_consistency(args, template)

    def test_dynamic_template_features_are_recomputed(self) -> None:
        row = pd.DataFrame(
            [
                {
                    "longitude": 2.302806,
                    "latitude": 48.878673,
                    "surface_reelle_bati": 58.0,
                    "nombre_pieces_principales": 3.0,
                    "ban_x": 648_873.72,
                    "ban_y": 6_864_512.73,
                    "ban_address_id": "75108_2576_00029",
                    "ban_match_distance_m": 999.0,
                    "ban_match_confidence": 0.0,
                    "building_year": 1900.0,
                    "building_year_missing": 1,
                    "building_age_at_sale": 1.0,
                }
            ]
        )
        history = pd.DataFrame(
            {
                "date_mutation": pd.Series(dtype="datetime64[ns]"),
                "x_l93": pd.Series(dtype="float64"),
                "y_l93": pd.Series(dtype="float64"),
                "price_per_m2": pd.Series(dtype="float64"),
                "surface_reelle_bati": pd.Series(dtype="float64"),
                "nombre_pieces_principales": pd.Series(dtype="float64"),
            }
        )

        result = add_base_and_comparable_features(
            row, history, pd.Timestamp("2025-01-02")
        )

        self.assertAlmostEqual(result.at[0, "ban_match_distance_m"], 10.51, places=1)
        self.assertEqual(result.at[0, "ban_match_confidence"], 1.0)
        self.assertEqual(result.at[0, "building_year_missing"], 0)
        self.assertEqual(result.at[0, "building_age_at_sale"], 125.0)

    def test_date_outside_evaluated_period_is_rejected(self) -> None:
        arguments = [
            "2026-01-02" if value == "2025-01-02" else value
            for value in VALID_ARGUMENTS
        ]
        message = self.parse_error(arguments)
        self.assertIn("this model is only evaluated for 2025", message)

    def test_malformed_date_is_rejected_cleanly(self) -> None:
        arguments = [
            "not-a-date" if value == "2025-01-02" else value
            for value in VALID_ARGUMENTS
        ]
        message = self.parse_error(arguments)
        self.assertIn("valid date in YYYY-MM-DD format", message)


if __name__ == "__main__":
    unittest.main()
