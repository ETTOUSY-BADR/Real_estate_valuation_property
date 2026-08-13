"""Consistency checks for release and scholarly attribution metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path
import tomllib
import unittest

import paris_avm


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_AUTHOR = "Badr Ettousy"


def cff_scalar(contents: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}:\s*[\"']?([^\"'\n]+)[\"']?\s*$", contents, re.MULTILINE)
    if match is None:
        raise AssertionError(f"CITATION.cff is missing the {key!r} field")
    return match.group(1).strip()


class ProjectMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pyproject = tomllib.loads(
            (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )["project"]
        cls.config = json.loads(
            (ROOT / "configs" / "project.json").read_text(encoding="utf-8")
        )
        cls.citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")

    def test_release_version_is_consistent(self) -> None:
        expected = self.pyproject["version"]
        self.assertEqual(paris_avm.__version__, expected)
        self.assertEqual(self.config["version"], expected)
        self.assertEqual(cff_scalar(self.citation, "version"), expected)

    def test_author_is_consistent(self) -> None:
        self.assertEqual(self.pyproject["authors"], [{"name": EXPECTED_AUTHOR}])
        self.assertRegex(self.citation, r"(?m)^\s+given-names:\s+Badr\s*$")
        self.assertRegex(self.citation, r"(?m)^\s+-\s+family-names:\s+Ettousy\s*$")

        manuscripts = {}
        for path in sorted((ROOT / "docs").rglob("*.tex")):
            contents = path.read_text(encoding="utf-8")
            if r"\documentclass" in contents:
                manuscripts[path] = contents
        self.assertGreater(len(manuscripts), 0)
        for manuscript, contents in manuscripts.items():
            with self.subTest(manuscript=manuscript.relative_to(ROOT)):
                self.assertIn(rf"\author{{{EXPECTED_AUTHOR}}}", contents)
                self.assertNotRegex(contents, r"\bBadre\b")


if __name__ == "__main__":
    unittest.main()
