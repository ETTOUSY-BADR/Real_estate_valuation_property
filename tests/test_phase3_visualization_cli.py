"""Command-line behavior for the Phase 3 visualization entry point."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

from paris_avm.visualization import phase3


class Phase3VisualizationCliTests(unittest.TestCase):
    def test_help_does_not_load_generated_artifacts(self) -> None:
        output = StringIO()
        with patch.object(phase3, "load_inputs") as load_inputs:
            with redirect_stdout(output), self.assertRaises(SystemExit) as raised:
                phase3.main(["--help"])

        self.assertEqual(raised.exception.code, 0)
        self.assertIn("publication figures", output.getvalue())
        load_inputs.assert_not_called()


if __name__ == "__main__":
    unittest.main()
