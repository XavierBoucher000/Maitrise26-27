"""Canonical filesystem paths shared by every analysis package."""

from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT.parent
DATA_ROOT = PROJECT_ROOT / "Data"
FIG_ROOT = PROJECT_ROOT / "fig"
