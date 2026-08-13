"""Command-line validation for Phase 3 single-property inference."""

from __future__ import annotations

import contextlib
import io
import unittest

from paris_avm.inference.phase3 import normalized_address_id, parse_args


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
