"""Repository-level checks for the professional source layout."""

from __future__ import annotations

import ast
import importlib
import pkgutil
import unittest
from pathlib import Path

import paris_avm


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

    def test_all_package_modules_import(self) -> None:
        modules = sorted(
            module.name
            for module in pkgutil.walk_packages(
                paris_avm.__path__, prefix=f"{paris_avm.__name__}."
            )
        )
        self.assertGreater(len(modules), 0)
        for module in modules:
            with self.subTest(module=module):
                importlib.import_module(module)


if __name__ == "__main__":
    unittest.main()
