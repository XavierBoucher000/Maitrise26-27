"""Canonical output paths for all generated project figures and reports.

Patient-specific output:
    fig/<script_name>/pX/<source cycle folder>/<artifact>

System-, isotope-, or calibration-level output:
    fig/<script_name>/shared/<artifact>

The environment variables ``FIG_PATIENT`` and ``FIG_CYCLE`` allow older
scripts to target another dataset without duplicating path constants.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Union


PROJECT_DIR = Path(__file__).resolve().parent
FIG_ROOT = PROJECT_DIR / "fig"
DEFAULT_PATIENT = "p11"
DEFAULT_CYCLE = "2026-05__Studies"


def _patient_folder(value: str) -> str:
    match = re.fullmatch(
        r"(?:patient[\s_-]*|p)(\d+)", str(value).strip(), re.IGNORECASE
    )
    if match is None:
        raise ValueError(f"patient must look like 'p11', got {value!r}")
    return f"p{int(match.group(1))}"


def _cycle_folder(value: str) -> str:
    normalized = str(value).strip()
    if re.fullmatch(r"\d{4}-\d{2}__Studies(?:-\d+)?", normalized) is None:
        raise ValueError(
            "cycle must use its source folder name, for example "
            f"'2026-05__Studies', got {value!r}"
        )
    return normalized


def analysis_name(script: Union[str, Path]) -> str:
    """Return a stable analysis folder name from a script path or name."""
    return Path(script).stem


def patient_cycle_dir(
    script: Union[str, Path],
    patient: str | None = None,
    cycle: str | None = None,
) -> Path:
    """Return the canonical directory for patient-specific artifacts."""
    patient_folder = _patient_folder(
        patient or os.environ.get("FIG_PATIENT", DEFAULT_PATIENT),
    )
    cycle_folder = _cycle_folder(
        cycle or os.environ.get("FIG_CYCLE", DEFAULT_CYCLE),
    )
    return FIG_ROOT / analysis_name(script) / patient_folder / cycle_folder


def shared_dir(script: Union[str, Path]) -> Path:
    """Return the canonical directory for non-patient-specific artifacts."""
    return FIG_ROOT / analysis_name(script) / "shared"
