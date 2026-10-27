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
from typing import Union


PROJECT_DIR = Path(__file__).resolve().parent
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
ANALYSIS_CATEGORIES = {
    "attenuation_global_model_v2": ("models",),
    "attenuation_window_model": ("models",),
    "patient_effective_sensitivity": ("models",),
    "patient_sensitivity_feature_test": ("models",),
    "patient_sensitivity_feature_test_v2": ("models",),
    "patient_sensitivity_multicycle_v3": ("models",),
    "patient_sensitivity_generalized": ("models",),
    "patient_sensitivity_qspect_slopes": ("models",),
    "patient_sensitivity_reference_comparison": ("models",),
    "patient_sensitivity_urine_target": ("models",),
    "patient_sensitivity_first_point_calibration": ("models",),
    "patient_sensitivity_urine_models": ("models",),
    "generate_planar_dead_time_report": ("corrections", "dead_time"),
    "compare_tew_pixel_global": ("corrections", "scatter"),
    "plot_attenuation_windows": ("corrections", "scatter"),
    "plot_177lu_spectrum_windows": ("corrections", "scatter"),
    "attenuation_correction": ("corrections", "ct_attenuation"),
    "compare_ct_conversion_methods": ("corrections", "ct_attenuation"),
    "compare_ct_planar_simpleitk": ("corrections", "ct_attenuation"),
    "compare_ap_pa_alignment_options": ("crop_alignment",),
    "compare_crop_methods": ("crop_alignment",),
    "crop_excluded_counts_test": ("crop_alignment",),
    "planar_qspect_profile_crop": ("crop_alignment",),
    "plot_ap_pa_alignment_test": ("crop_alignment",),
    "plot_planar_qspect_coverage": ("crop_alignment",),
    "excretion_calibration_test": ("retention_excretion",),
    "imaging_excretion_fit_balance": ("retention_excretion",),
    "planar_excretion_fit_balance": ("retention_excretion",),
    "planar_total_counts_retention": ("retention_excretion",),
    "sang_urine": ("retention_excretion",),
    "acquisition_timeline": ("quality_control",),
    "metadonne": ("quality_control",),
    "metadonné": ("quality_control",),
    "view_patient_images": ("quality_control",),
    "compare_planar_qspect": ("validation",),
}


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
