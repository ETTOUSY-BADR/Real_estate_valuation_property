"""Repository-level checks for the professional source layout."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ProjectStructureTests(unittest.TestCase):
    def test_expected_directories_and_metadata_exist(self) -> None:
        expected = [
            "configs",
            "data",
            "docs",
            "models",
            "reports",
            "scripts",
            "src/paris_avm",
            "tests",
            "visuals",
            "README.md",
            "pyproject.toml",
        ]
        for relative_path in expected:
            with self.subTest(path=relative_path):
                self.assertTrue((ROOT / relative_path).exists())

    def test_all_python_sources_parse(self) -> None:
        source_files = sorted((ROOT / "src").rglob("*.py"))
        self.assertGreater(len(source_files), 0)
        for source_file in source_files:
            with self.subTest(source=source_file.relative_to(ROOT)):
                ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))

    def test_public_pipeline_modules_import(self) -> None:
        modules = [
            "paris_avm.data.acquire_phase3",
            "paris_avm.features.phase2",
            "paris_avm.features.phase3",
            "paris_avm.modeling.benchmark_phase3",
            "paris_avm.inference.phase3",
            "paris_avm.visualization.phase3",
        ]
        for module in modules:
            with self.subTest(module=module):
                importlib.import_module(module)


if __name__ == "__main__":
    unittest.main()
