"""Canonical output paths for all generated project figures and reports.

Patient-specific output:
    fig/<category>/<script_name>/pX/<source cycle folder>/<artifact>

System-, isotope-, or calibration-level output:
    fig/<category>/<script_name>/shared/<artifact>

The environment variables ``FIG_PATIENT`` and ``FIG_CYCLE`` allow older
scripts to target another dataset without duplicating path constants.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from core.project_paths import PROJECT_ROOT
from typing import Union

from admin.analysis_catalog import FIGURE_CATEGORY_BY_NAME


PROJECT_DIR = PROJECT_ROOT
FIG_ROOT = PROJECT_DIR / "fig"
DEFAULT_PATIENT = "p11"
DEFAULT_CYCLE = "2026-05__Studies"

# Ordre visuel commun aux figures multi-patients. Les cycles d'un même patient
# restent adjacents; les autres patients suivent la date de leur premier cycle.
PATIENT_ORDER = ("p11", "p8", "p17")


def patient_sort_key(patient: str) -> tuple[int, str]:
    """Return the stable display order for one patient identifier."""
    normalized = _patient_folder(patient)
    try:
        rank = PATIENT_ORDER.index(normalized)
    except ValueError:
        rank = len(PATIENT_ORDER)
    return rank, normalized


def patient_cycle_sort_key(patient: str, cycle: str) -> tuple[int, str, str]:
    """Group cycles by patient, then order cycles chronologically."""
    rank, normalized = patient_sort_key(patient)
    return rank, normalized, str(cycle)


# Scientific categories keep the figure tree navigable while the script name
# remains a stable and unambiguous leaf directory.  Unknown analyses go under
# ``other`` instead of silently recreating folders at the root of ``fig``.
ANALYSIS_CATEGORIES = FIGURE_CATEGORY_BY_NAME


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


def analysis_category(script: Union[str, Path]) -> tuple[str, ...]:
    """Return the canonical category path for an analysis script."""
    return ANALYSIS_CATEGORIES.get(analysis_name(script), ("other",))


def analysis_dir(script: Union[str, Path]) -> Path:
    """Return the canonical root directory for one analysis."""
    name = analysis_name(script)
    return FIG_ROOT.joinpath(*analysis_category(script), name)


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
    return analysis_dir(script) / patient_folder / cycle_folder


def shared_dir(script: Union[str, Path]) -> Path:
    """Return the canonical directory for non-patient-specific artifacts."""
    return analysis_dir(script) / "shared"
