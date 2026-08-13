"""Canonical repository paths shared by command-line modules."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DVF_DIR = DATA_DIR / "raw" / "dvf"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
VISUALS_DIR = PROJECT_ROOT / "visuals"
PAPER_DIR = PROJECT_ROOT / "docs" / "paper"
