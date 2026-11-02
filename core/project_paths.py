"""Canonical filesystem paths shared by every analysis package."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "Data"
FIG_ROOT = PROJECT_ROOT / "fig"
