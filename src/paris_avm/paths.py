"""Canonical repository paths shared by command-line modules."""

import os
from pathlib import Path


def _is_checkout(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (
        path / "src" / "paris_avm"
    ).is_dir()


def resolve_project_root(start: Path | None = None) -> Path:
    """Locate the repository used for data, models, reports, and documents."""
    configured = os.environ.get("PARIS_AVM_ROOT")
    if configured:
        root = Path(configured).resolve()
        if not root.is_dir():
            raise RuntimeError(f"PARIS_AVM_ROOT is not a directory: {root}")
        return root

    working_directory = (start or Path.cwd()).resolve()
    for candidate in (working_directory, *working_directory.parents):
        if _is_checkout(candidate):
            return candidate

    source_candidate = Path(__file__).resolve().parents[2]
    if _is_checkout(source_candidate):
        return source_candidate

    # Installed commands may operate on a data workspace without source files.
    # In that case the invocation directory is the least surprising default.
    return working_directory


PROJECT_ROOT = resolve_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DVF_DIR = DATA_DIR / "raw" / "dvf"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
VISUALS_DIR = PROJECT_ROOT / "visuals"
PAPER_DIR = PROJECT_ROOT / "docs" / "paper"
