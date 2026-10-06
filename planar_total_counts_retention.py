"""Simple whole-body planar counts versus urine-derived retention.

This module answers a deliberately simple question before more sophisticated
scatter/attenuation models are considered.  For each primary ``WB RAPIDE``
acquisition it evaluates the complete images with two detector-combination
modes:

* ``AP_PA``: direct sum of the AP and PA image counts;
* ``GM``: pixelwise geometric mean ``sqrt(AP * PA_aligned)``, followed by the
  sum of the GM image.

The same calculation is then reported for:

1. the 208-keV photopeak window;
2. the broad low-energy window labelled ``Low Energy Scatter`` in DICOM
   (called *general scatter* in the figures);
3. all four acquired energy windows together.

No crop, scatter correction, CT correction, dead-time correction or
camera-sensitivity calibration is applied.  When a compatible urine/diaper CSV
is available, the first three timepoints are compared with the decay-consistent
remaining activity.  Image-only results are produced for every configured
patient/cycle, including cycles without an excretion file.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import attenuation_correction as ac
import dicom_loader
import figure_layout
import planar_processing
import sang_urine


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PLANAR_DIR = PROJECT_DIR / "Data" / "p11" / "2026-05__Studies_WBP"
DEFAULT_URINE_CSV = PROJECT_DIR / "Data" / "p11" / "P-011csv.csv"
COMBINATION_AP_PA = "AP_PA"
COMBINATION_GM = "GM"
COMBINATIONS = (COMBINATION_AP_PA, COMBINATION_GM)
DEFAULT_COMBINATION = COMBINATION_AP_PA

DEFAULT_OUTPUT_DIR = (
    figure_layout.patient_cycle_dir(__file__, "p11", "2026-05__Studies")
    / DEFAULT_COMBINATION
)

DATASETS: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("p11", "2026-05__Studies"): {
        "patient_label": "P11",
        "cycle_label": "cycle de mai 2026",
        "planar_dir": PROJECT_DIR / "Data/p11/2026-05__Studies_WBP",
        "urine_csv": PROJECT_DIR / "Data/p11/P-011csv.csv",
    },
    ("p11", "2026-07__Studies"): {
        "patient_label": "P11",
        "cycle_label": "cycle de juillet 2026",
        "planar_dir": PROJECT_DIR / "Data/p11/2026-07__Studies_WBP",
        "urine_csv": None,
    },
    ("p8", "2026-06__Studies"): {
        "patient_label": "P8",
        "cycle_label": "cycle de juin 2026",
        "planar_dir": PROJECT_DIR / "Data/p8/2026-06__Studies_WBP",
        "urine_csv": PROJECT_DIR / "Data/p8/P-008.csv",
    },
}

EXPECTED_WINDOWS = tuple(planar_processing.ENERGY_ORDER)
PHOTOPEAK = "Photopeak"
GENERAL_SCATTER = "Low Energy Scatter"


def _finite_float(value: Any) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Expected a finite numeric value, got {value!r}")
    return result


def _csv_number(value: str) -> Optional[float]:
    text = str(value).strip().replace(" ", "").replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _next_nonempty(row: Sequence[str], start: int) -> str:
    for value in row[start + 1 :]:
        if str(value).strip():
            return str(value).strip()
    return ""


def _urine_hour_key(text: str) -> Optional[int]:
    match = re.search(r"urine\s*(\d+)\s*hr", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def load_urine_retention_csv(csv_path: Path) -> Dict[str, Any]:
    """Load the common urine/diaper balance fields from P11 or P8 CSV files.

    The two spreadsheets contain the same summary table shifted by one column.
    This parser anchors columns to the ``Urine +Diapers`` header instead of
    assuming a patient-specific absolute column number.  Blood measurements
    are intentionally ignored for whole-body retention.
    """
    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))

    injected_activity_mbq: Optional[float] = None
    half_life_h: Optional[float] = None
    injection_datetime: Optional[datetime] = None
    summary_header_row: Optional[int] = None
    total_excreted_column: Optional[int] = None

    for row_index, row in enumerate(rows):
        for column, raw_value in enumerate(row):
            key = str(raw_value).strip().lower()
            following = _next_nonempty(row, column)
            if key in {"injected acctivity", "injected activity"}:
                injected_activity_mbq = _csv_number(following)
            elif "177lu half-life" in key:
                half_life_h = _csv_number(following)
            elif key == "time of injection":
                injection_datetime = datetime.strptime(
                    following, "%Y-%m-%d %H:%M"
                )
            elif "urine +diapers" in key and "sampling time" in key:
                summary_header_row = row_index
                total_excreted_column = column

    if (
        injected_activity_mbq is None
        or half_life_h is None
        or injection_datetime is None
    ):
        raise ValueError(
            f"Injected activity, Lu-177 half-life or injection time is missing in {csv_path}"
        )
    if summary_header_row is None or total_excreted_column is None:
        raise ValueError(f"Urine/diaper summary header is missing in {csv_path}")

    label_column = total_excreted_column - 6
    urine_at_sampling_column = total_excreted_column - 1
    sampling_datetime_column = total_excreted_column - 4
    elapsed_column = total_excreted_column + 2
    urine: List[Dict[str, Any]] = []
    for row in rows[summary_header_row + 1 :]:
        label = row[label_column].strip() if label_column < len(row) else ""
        hour_key = _urine_hour_key(label)
        total_mbq = (
            _csv_number(row[total_excreted_column])
            if total_excreted_column < len(row)
            else None
        )
        urine_mbq = (
            _csv_number(row[urine_at_sampling_column])
            if urine_at_sampling_column < len(row)
            else None
        )
        elapsed_h = (
            _csv_number(row[elapsed_column])
            if elapsed_column < len(row)
            else None
        )
        if hour_key is None or total_mbq is None or elapsed_h is None:
            continue
        sampling_text = (
            row[sampling_datetime_column].strip()
            if sampling_datetime_column < len(row)
            else ""
        )
        urine_value = float(urine_mbq) if urine_mbq is not None else math.nan
        urine.append(
            {
                "label": label,
                "nominal_collection_h": hour_key,
                "elapsed_h": float(elapsed_h),
                "urine_activity_mbq": urine_value,
                "diaper_activity_mbq": (
                    float(total_mbq) - urine_value
                    if np.isfinite(urine_value)
                    else math.nan
                ),
                "total_excreted_activity_mbq": float(total_mbq),
                "sampling_datetime": (
                    datetime.strptime(sampling_text, "%Y-%m-%d %H:%M")
                    if sampling_text
                    else None
                ),
            }
        )
    urine.sort(key=lambda item: item["elapsed_h"])
    if not urine:
        raise ValueError(f"No urine/diaper interval was found in {csv_path}")

    return {
        "source_csv": csv_path,
        "injected_activity_mbq": float(injected_activity_mbq),
        "half_life_h": float(half_life_h),
        "injection_datetime": injection_datetime,
        "blood": [],
        "urine": urine,
    }


def total_ap_pa_counts_by_window(scan: Mapping[str, Any]) -> Dict[str, float]:
    """Return full-image AP+PA raw counts for each of the four windows.

    Exactly one AP and one PA frame are required per window.  Requiring the
    expected eight primary frames prevents a derived EPP-Alpha object or an
    incomplete acquisition from entering this analysis silently.
    """
    grouped: Dict[str, Dict[str, List[float]]] = {
        window: {"AP": [], "PA": []} for window in EXPECTED_WINDOWS
    }
    for image in scan.get("images", []):
        window = planar_processing.normalize_energy_label(
            image.get("energy_window") or image.get("energy_window_name")
        )
        view = str(image.get("view", "")).upper()
        if window not in grouped or view not in {"AP", "PA"}:
            continue
        pixel_sum = image.get("pixel_sum")
        if pixel_sum is None:
            pixels = image.get("image")
            if pixels is None:
                raise ValueError(f"Missing pixels for {window} {view}")
            pixel_sum = np.asarray(pixels, dtype=np.float64).sum()
        grouped[window][view].append(_finite_float(pixel_sum))

    missing = [
        f"{window} {view}"
        for window in EXPECTED_WINDOWS
        for view in ("AP", "PA")
        if len(grouped[window][view]) != 1
    ]
    if missing:
        raise ValueError(
            "The primary planar acquisition must contain exactly one frame for "
            "each AP/PA energy window; invalid entries: " + ", ".join(missing)
        )

    return {
        window: grouped[window]["AP"][0] + grouped[window]["PA"][0]
        for window in EXPECTED_WINDOWS
    }


def total_gm_counts_by_window(
    scan: Mapping[str, Any],
    align_pa_to_ap: bool = planar_processing.ALIGN_PA_TO_AP,
) -> Dict[str, float]:
    """Return full-image pixelwise AP/PA geometric-mean counts by window.

    For each energy window, the PA image is first placed in the AP left/right
    orientation when ``align_pa_to_ap`` is true. The detector combination is
    then performed *pixel by pixel* and the resulting GM image is summed::

        C_GM = sum_pixels sqrt(AP * PA_aligned)

    This is deliberately different from ``sqrt(sum(AP) * sum(PA))``.
    """
    grouped: Dict[str, Dict[str, List[Mapping[str, Any]]]] = {
        window: {"AP": [], "PA": []} for window in EXPECTED_WINDOWS
    }
    for image in scan.get("images", []):
        window = planar_processing.normalize_energy_label(
            image.get("energy_window") or image.get("energy_window_name")
        )
        view = str(image.get("view", "")).upper()
        if window in grouped and view in {"AP", "PA"}:
            grouped[window][view].append(image)
    invalid = [
        f"{window} {view}"
        for window in EXPECTED_WINDOWS
        for view in ("AP", "PA")
        if len(grouped[window][view]) != 1
    ]
    if invalid:
        raise ValueError(
            "The primary planar acquisition must contain exactly one frame for "
            "each AP/PA energy window; invalid entries: " + ", ".join(invalid)
        )

    counts: Dict[str, float] = {}
    for window in EXPECTED_WINDOWS:
        ap_pixels = grouped[window]["AP"][0].get("image")
        pa_pixels = grouped[window]["PA"][0].get("image")
        if ap_pixels is None or pa_pixels is None:
            raise ValueError(f"Missing pixel arrays for {window} GM")
        ap = np.asarray(ap_pixels, dtype=np.float64)
        pa = np.asarray(pa_pixels, dtype=np.float64)
        if ap.shape != pa.shape:
            raise ValueError(
                f"AP and PA shapes differ for {window}: {ap.shape} versus {pa.shape}"
            )
        counts[window] = float(
            ac.geometric_mean(ap, pa, align_pa_to_ap=align_pa_to_ap).sum()
        )
    return counts


def total_counts_by_window(
    scan: Mapping[str, Any],
    combination: str = DEFAULT_COMBINATION,
    align_pa_to_ap: bool = planar_processing.ALIGN_PA_TO_AP,
) -> Dict[str, float]:
    """Dispatch the requested AP/PA detector-combination definition."""
    if combination == COMBINATION_AP_PA:
        return total_ap_pa_counts_by_window(scan)
    if combination == COMBINATION_GM:
        return total_gm_counts_by_window(
            scan, align_pa_to_ap=align_pa_to_ap
        )
    raise ValueError(
        f"Unknown combination {combination!r}; expected one of {COMBINATIONS}"
    )


def _combination_description(
    combination: str,
    align_pa_to_ap: bool = planar_processing.ALIGN_PA_TO_AP,
) -> str:
    if combination == COMBINATION_AP_PA:
        return "somme des comptes AP + PA"
    if combination == COMBINATION_GM:
        orientation = "PA retourné vers AP" if align_pa_to_ap else "PA non retourné"
        return f"moyenne géométrique pixel par pixel ({orientation})"
    raise ValueError(f"Unknown combination {combination!r}")


def _combination_title(result: Mapping[str, Any]) -> str:
    combination = result.get("view_combination", DEFAULT_COMBINATION)
    if combination == COMBINATION_AP_PA:
        return "AP + PA"
    if combination == COMBINATION_GM:
        return "GM pixel par pixel (PA aligné)"
    raise ValueError(f"Unknown combination {combination!r}")


def _counts_axis_label(result: Mapping[str, Any]) -> str:
    combination = result.get("view_combination", DEFAULT_COMBINATION)
    return (
        "Full-image AP+PA counts (millions)"
        if combination == COMBINATION_AP_PA
        else "Full-image geometric-mean counts (millions)"
    )


def _timepoint_labels(datetimes: Sequence[datetime]) -> List[str]:
    if not datetimes:
        return []
    first = min(datetimes)
    return [
        f"J{int(round((acquisition - first).total_seconds() / 86400.0))}"
        for acquisition in datetimes
    ]


def extract_raw_planar_count_rows(
    planar_dir: Path,
    injection_datetime: Optional[datetime] = None,
    combination: str = DEFAULT_COMBINATION,
    align_pa_to_ap: bool = planar_processing.ALIGN_PA_TO_AP,
) -> List[Dict[str, Any]]:
    """Load primary WB RAPIDE series and calculate full-image totals."""
    scans = ac.sorted_planar_scans(Path(planar_dir))
    if not scans:
        raise FileNotFoundError(f"No primary WB RAPIDE acquisition in {planar_dir}")

    scan_datetimes = [dicom_loader.scan_datetime(scan) for scan in scans]
    if any(value is None for value in scan_datetimes):
        raise ValueError("Every planar scan must contain a DICOM acquisition datetime")
    datetimes = [value for value in scan_datetimes if value is not None]
    labels = _timepoint_labels(datetimes)
    first_acquisition = min(datetimes)

    rows: List[Dict[str, Any]] = []
    for label, acquisition_datetime, scan in zip(labels, datetimes, scans):
        counts = total_counts_by_window(
            scan,
            combination=combination,
            align_pa_to_ap=align_pa_to_ap,
        )
        timing = planar_processing.planar_timing_from_dicom(scan["images"])
        duration_s = _finite_float(timing.actual_frame_duration_s)
        if duration_s <= 0.0:
            raise ValueError("ActualFrameDuration must be positive")
        photopeak = counts[PHOTOPEAK]
        general_scatter = counts[GENERAL_SCATTER]
        all_windows = float(sum(counts[window] for window in EXPECTED_WINDOWS))
        rows.append(
            {
                "timepoint": label,
                "scan_name": scan["scan_name"],
                "view_combination": combination,
                "align_pa_to_ap": bool(align_pa_to_ap),
                "acquisition_datetime": acquisition_datetime,
                "hours_after_first_planar": (
                    acquisition_datetime - first_acquisition
                ).total_seconds()
                / 3600.0,
                "days_after_first_planar": (
                    acquisition_datetime - first_acquisition
                ).total_seconds()
                / 86400.0,
                "hours_after_injection": (
                    (acquisition_datetime - injection_datetime).total_seconds()
                    / 3600.0
                    if injection_datetime is not None
                    else math.nan
                ),
                "days_after_injection": (
                    (acquisition_datetime - injection_datetime).total_seconds()
                    / 86400.0
                    if injection_datetime is not None
                    else math.nan
                ),
                "frame_duration_s": duration_s,
                "lower_scatter_counts": counts["Lower Scatter"],
                "photopeak_counts": photopeak,
                "upper_scatter_counts": counts["Upper Scatter"],
                "general_scatter_counts": general_scatter,
                "all_four_windows_counts": all_windows,
                "photopeak_cps": photopeak / duration_s,
                "general_scatter_cps": general_scatter / duration_s,
                "all_four_windows_cps": all_windows / duration_s,
                "photopeak_over_general_scatter": (
                    photopeak / general_scatter
                    if general_scatter > 0.0
                    else math.nan
                ),
            }
        )
    return rows


def ratio_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    """Summarize the uncorrected photopeak/general-scatter count ratio."""
    ratios = np.asarray(
        [row["photopeak_over_general_scatter"] for row in rows], dtype=float
    )
    if ratios.size == 0 or np.any(~np.isfinite(ratios)):
        raise ValueError("Photopeak/general-scatter ratios must be finite")
    mean = float(np.mean(ratios))
    sample_sd = float(np.std(ratios, ddof=1)) if ratios.size > 1 else 0.0
    return {
        "mean": mean,
        "sample_sd": sample_sd,
        "cv_percent": 100.0 * sample_sd / mean if mean != 0.0 else math.nan,
        "minimum": float(np.min(ratios)),
        "maximum": float(np.max(ratios)),
        "first_to_last_change_percent": 100.0 * (ratios[-1] / ratios[0] - 1.0),
    }


def analyze_planar_counts(
    planar_dir: Path,
    injection_datetime: Optional[datetime] = None,
    combination: str = DEFAULT_COMBINATION,
    align_pa_to_ap: bool = planar_processing.ALIGN_PA_TO_AP,
) -> Dict[str, Any]:
    """Analyze full-image planar counts without requiring urine data."""
    rows = extract_raw_planar_count_rows(
        Path(planar_dir),
        injection_datetime,
        combination=combination,
        align_pa_to_ap=align_pa_to_ap,
    )
    return {
        "rows": rows,
        "ratio_summary": ratio_summary(rows),
        "urine_available": False,
        "comparison_timepoints": 0,
        "fits": {},
        "view_combination": combination,
        "align_pa_to_ap": bool(align_pa_to_ap),
    }


def add_urine_retention_reference(
    rows: Sequence[Mapping[str, Any]],
    urine_data: Mapping[str, Any],
    urine_balance: Mapping[str, np.ndarray],
) -> List[Dict[str, Any]]:
    """Add same-time physical activity, measured excretion and retention."""
    times_h = np.asarray([row["hours_after_injection"] for row in rows], dtype=float)
    physical_available = float(urine_data["injected_activity_mbq"]) * (
        sang_urine.decay_factor(times_h, float(urine_data["half_life_h"]))
    )
    cumulative_excreted = sang_urine.cumulative_excreted_activity(
        times_h, dict(urine_data), dict(urine_balance)
    )
    remaining = sang_urine.predicted_body_activity(
        times_h, dict(urine_data), dict(urine_balance)
    )
    decay = sang_urine.decay_factor(times_h, float(urine_data["half_life_h"]))

    output: List[Dict[str, Any]] = []
    for index, source in enumerate(rows):
        row = dict(source)
        row.update(
            {
                "physical_available_activity_mbq": float(physical_available[index]),
                "measured_cumulative_excreted_activity_mbq": float(
                    cumulative_excreted[index]
                ),
                "urine_derived_remaining_activity_mbq": float(remaining[index]),
                "urine_derived_remaining_injection_equivalent_mbq": float(
                    remaining[index] / decay[index]
                ),
            }
        )
        output.append(row)
    return output


def through_origin_summary(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    """Fit ``y = slope*x`` and return descriptive statistics.

    The zero intercept encodes the physical expectation of zero patient counts
    at zero activity.  With only three points, these statistics are descriptive
    and are not a validation of a calibration model.
    """
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if x_array.shape != y_array.shape or x_array.ndim != 1 or x_array.size < 2:
        raise ValueError("x and y must be same-length one-dimensional arrays")
    if np.any(~np.isfinite(x_array)) or np.any(~np.isfinite(y_array)):
        raise ValueError("x and y must be finite")
    denominator = float(np.dot(x_array, x_array))
    if denominator <= 0.0:
        raise ValueError("Cannot fit a zero-origin slope with zero x values")
    slope = float(np.dot(x_array, y_array) / denominator)
    prediction = slope * x_array
    residual_sum = float(np.sum((y_array - prediction) ** 2))
    total_sum = float(np.sum((y_array - np.mean(y_array)) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0.0 else math.nan
    counts_per_mbq = y_array / x_array
    mean_counts_per_mbq = float(np.mean(counts_per_mbq))
    sample_sd = (
        float(np.std(counts_per_mbq, ddof=1)) if counts_per_mbq.size > 1 else 0.0
    )
    return {
        "slope_counts_per_mbq": slope,
        "r_squared_zero_origin": r_squared,
        "mean_pointwise_counts_per_mbq": mean_counts_per_mbq,
        "sd_pointwise_counts_per_mbq": sample_sd,
        "cv_pointwise_counts_per_mbq_percent": (
            100.0 * sample_sd / mean_counts_per_mbq
            if mean_counts_per_mbq != 0.0
            else math.nan
        ),
    }


def analyze_total_counts_retention(
    planar_dir: Path = DEFAULT_PLANAR_DIR,
    urine_csv: Path = DEFAULT_URINE_CSV,
    comparison_timepoints: int = 3,
    combination: str = DEFAULT_COMBINATION,
    align_pa_to_ap: bool = planar_processing.ALIGN_PA_TO_AP,
) -> Dict[str, Any]:
    """Calculate image counts and the decay-consistent urine comparison."""
    urine_data = load_urine_retention_csv(Path(urine_csv))
    urine_balance = sang_urine.calculate_urine_mass_balance(urine_data)
    rows = extract_raw_planar_count_rows(
        Path(planar_dir),
        urine_data["injection_datetime"],
        combination=combination,
        align_pa_to_ap=align_pa_to_ap,
    )
    rows = add_urine_retention_reference(rows, urine_data, urine_balance)
    if comparison_timepoints < 2 or comparison_timepoints > len(rows):
        raise ValueError(
            f"comparison_timepoints must be between 2 and {len(rows)}"
        )

    comparison_rows = rows[:comparison_timepoints]
    activity = [
        row["urine_derived_remaining_activity_mbq"] for row in comparison_rows
    ]
    count_fields = {
        "photopeak": "photopeak_counts",
        "general_scatter": "general_scatter_counts",
        "all_four_windows": "all_four_windows_counts",
    }
    fits = {
        name: through_origin_summary(
            activity, [row[field] for row in comparison_rows]
        )
        for name, field in count_fields.items()
    }

    return {
        "urine_data": urine_data,
        "urine_balance": urine_balance,
        "rows": rows,
        "comparison_timepoints": comparison_timepoints,
        "fits": fits,
        "ratio_summary": ratio_summary(rows),
        "urine_available": True,
        "view_combination": combination,
        "align_pa_to_ap": bool(align_pa_to_ap),
    }


def plot_counts_vs_retention(
    result: Mapping[str, Any], output_path: Path
) -> Path:
    """Plot the three raw count definitions against remaining activity."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(result["rows"][: result["comparison_timepoints"]])
    x_mbq = np.asarray(
        [row["urine_derived_remaining_activity_mbq"] for row in rows], dtype=float
    )
    x_gbq = x_mbq / 1000.0
    panels = (
        ("photopeak", "photopeak_counts", "Photopeak", "#1f77b4"),
        (
            "general_scatter",
            "general_scatter_counts",
            "General scatter\n(55–166 keV)",
            "#d62728",
        ),
        (
            "all_four_windows",
            "all_four_windows_counts",
            "Four windows combined",
            "#2ca02c",
        ),
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), layout="constrained")
    for axis, (fit_name, field, title, color) in zip(axes, panels):
        y_million = np.asarray([row[field] for row in rows], dtype=float) / 1e6
        axis.scatter(
            x_gbq,
            y_million,
            s=64,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        for row, x_value, y_value in zip(rows, x_gbq, y_million):
            axis.annotate(
                str(row["timepoint"]),
                (x_value, y_value),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                color="#111111",
            )
        x_line = np.linspace(0.0, max(x_gbq) * 1.05, 100)
        slope = result["fits"][fit_name]["slope_counts_per_mbq"]
        y_line_million = slope * (x_line * 1000.0) / 1e6
        axis.plot(x_line, y_line_million, color=color, linewidth=1.8)
        axis.set_title(title)
        axis.set_xlabel("Remaining activity from urine balance (GBq)")
        axis.set_ylabel(_counts_axis_label(result))
        axis.grid(True, linestyle="--", alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.text(
            0.04,
            0.94,
            (
                f"Origin-constrained fit\n"
                f"R² = {result['fits'][fit_name]['r_squared_zero_origin']:.3f}\n"
                f"CV(counts/MBq) = "
                f"{result['fits'][fit_name]['cv_pointwise_counts_per_mbq_percent']:.1f}%"
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color="#111111",
        )
    fig.suptitle(
        f"{result.get('patient_label', 'Patient')} — "
        f"{result.get('cycle_label', 'cycle')}\n"
        "Whole-body planar counts versus urine-derived retention\n"
        f"Mode : {_combination_title(result)}"
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_ratio_vs_time(result: Mapping[str, Any], output_path: Path) -> Path:
    """Plot raw photopeak/general-scatter count ratio at every timepoint."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(result["rows"])
    days = np.asarray(
        [row["days_after_first_planar"] for row in rows], dtype=float
    )
    ratios = np.asarray(
        [row["photopeak_over_general_scatter"] for row in rows], dtype=float
    )
    summary = result["ratio_summary"]

    fig, axis = plt.subplots(figsize=(8.2, 5.0), layout="constrained")
    axis.axhline(
        summary["mean"],
        color="#111111",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {summary['mean']:.3f}",
    )
    axis.plot(
        days,
        ratios,
        marker="o",
        markersize=7,
        color="#6a3d9a",
        linewidth=2.2,
        label=f"Photopeak / general scatter (CV = {summary['cv_percent']:.1f}%)",
    )
    for row, x_value, y_value in zip(rows, days, ratios):
        axis.annotate(
            str(row["timepoint"]),
            (x_value, y_value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#111111",
        )
    axis.set_xlabel("Time after first planar acquisition (days)")
    axis.set_ylabel("Raw count ratio: photopeak / general scatter")
    axis.set_title(
        f"{result.get('patient_label', 'Patient')} — "
        f"{result.get('cycle_label', 'cycle')}\n"
        "Stability of the uncorrected spectral ratio\n"
        f"Mode : {_combination_title(result)}"
    )
    axis.grid(True, linestyle="--", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, loc="best")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_raw_counts_vs_time(result: Mapping[str, Any], output_path: Path) -> Path:
    """Plot the three requested full-image count definitions over time."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(result["rows"])
    days = np.asarray(
        [row["days_after_first_planar"] for row in rows], dtype=float
    )
    series = (
        ("photopeak_counts", "Photopeak", "#1f77b4", "o"),
        (
            "general_scatter_counts",
            "General scatter (55–166 keV)",
            "#d62728",
            "s",
        ),
        (
            "all_four_windows_counts",
            "Four windows combined",
            "#2ca02c",
            "^",
        ),
    )

    fig, axis = plt.subplots(figsize=(8.4, 5.2), layout="constrained")
    for field, label, color, marker in series:
        values = np.asarray([row[field] for row in rows], dtype=float) / 1e6
        axis.plot(
            days,
            values,
            marker=marker,
            markersize=6.5,
            linewidth=2.0,
            color=color,
            label=label,
        )
    axis.set_xlabel("Time after first planar acquisition (days)")
    axis.set_ylabel(_counts_axis_label(result))
    axis.set_title(
        f"{result.get('patient_label', 'Patient')} — "
        f"{result.get('cycle_label', 'cycle')}\n"
        "Full-image planar counts without correction\n"
        f"Mode : {_combination_title(result)}"
    )
    axis.grid(True, linestyle="--", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, loc="best")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_normalized_counts_and_retention(
    result: Mapping[str, Any], output_path: Path
) -> Path:
    """Compare first-timepoint-normalized counts and urine-derived retention."""
    if not result.get("urine_available"):
        raise ValueError("Urine-derived retention is unavailable for this cycle")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(result["rows"][: result["comparison_timepoints"]])
    days = np.asarray(
        [row["days_after_first_planar"] for row in rows], dtype=float
    )
    series = (
        (
            "urine_derived_remaining_activity_mbq",
            "Urine-derived remaining activity",
            "#111111",
            "D",
        ),
        ("photopeak_counts", "Photopeak counts", "#1f77b4", "o"),
        (
            "general_scatter_counts",
            "General-scatter counts",
            "#d62728",
            "s",
        ),
        (
            "all_four_windows_counts",
            "Four-window counts",
            "#2ca02c",
            "^",
        ),
    )

    fig, axis = plt.subplots(figsize=(8.4, 5.2), layout="constrained")
    for field, label, color, marker in series:
        values = np.asarray([row[field] for row in rows], dtype=float)
        normalized = values / values[0]
        axis.plot(
            days,
            normalized,
            marker=marker,
            markersize=6.5,
            linewidth=2.0,
            color=color,
            label=label,
        )
    axis.set_xlabel("Time after first planar acquisition (days)")
    axis.set_ylabel("Value normalized to first planar acquisition")
    axis.set_title(
        f"{result.get('patient_label', 'Patient')} — "
        f"{result.get('cycle_label', 'cycle')}\n"
        "Temporal agreement between counts and urine-derived retention\n"
        f"Mode : {_combination_title(result)}"
    )
    axis.grid(True, linestyle="--", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, loc="best")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def write_values_csv(result: Mapping[str, Any], output_path: Path) -> Path:
    """Write one auditable flat row per planar timepoint."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in result["rows"]]
    for index, row in enumerate(rows):
        row["used_in_counts_vs_activity_analysis"] = (
            index < result["comparison_timepoints"]
        )
        row["acquisition_datetime"] = row["acquisition_datetime"].isoformat(
            sep=" "
        )
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        f"{float(value):.8g}"
                        if isinstance(value, (float, np.floating))
                        else value
                    )
                    for key, value in row.items()
                }
            )
    return output_path


def write_report(result: Mapping[str, Any], output_path: Path) -> Path:
    """Document definitions, results and limits in a compact text report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(result["rows"])
    n_compare = int(result["comparison_timepoints"])
    summary = result["ratio_summary"]
    urine_available = bool(result.get("urine_available"))
    patient_label = str(result.get("patient_label", "Patient"))
    cycle_label = str(result.get("cycle_label", "cycle"))
    combination = str(result.get("view_combination", DEFAULT_COMBINATION))
    align_pa_to_ap = bool(
        result.get("align_pa_to_ap", planar_processing.ALIGN_PA_TO_AP)
    )
    lines = [
        f"{patient_label} - {cycle_label}: whole-body planar counts",
        "=" * 68,
        "",
        "Definitions",
        "-----------",
        f"* Detector combination: {_combination_description(combination, align_pa_to_ap)}.",
        "* Counts use the complete primary WB RAPIDE images; no crop is used.",
        "* Photopeak = DICOM Photopeak window (approximately 187.2-228.8 keV).",
        "* General scatter = DICOM Low Energy Scatter window (approximately",
        "  55.45-166.35 keV). It is not the TEW scatter estimate.",
        "* Four windows = lower scatter + photopeak + upper scatter + general",
        "  scatter, after applying the same detector-combination definition to",
        "  each window.",
        "* No scatter, attenuation, dead-time or sensitivity correction is applied.",
    ]
    if combination == COMBINATION_GM:
        lines.extend(
            [
                "* GM is calculated pixel by pixel, then summed over the image:",
                "  sum_pixels sqrt(AP * PA_aligned). It is not the geometric mean",
                "  of the two already-summed detector totals.",
            ]
        )
    if urine_available:
        last_collection_h = float(result["urine_balance"]["times_h"][-1])
        lines.extend(
            [
                "* Remaining activity at a planar time is the physically decayed",
                "  injected activity minus completed measured urine/diaper excretion",
                "  at that same time. Blood concentration is not used.",
                "",
                f"Counts-versus-activity subset: first {n_compare} timepoints.",
                f"Last measured urine/diaper collection: {last_collection_h:.2f} h.",
                "Later points are excluded from the urine comparison when later",
                "excretion is unmeasured; they remain valid for image-only results.",
                "",
                "First timepoints: zero-origin descriptive fits",
                "------------------------------------------------",
                " metric             | slope counts/MBq | R2 origin | CV of counts/MBq",
                "------------------- | ---------------- | --------- | ----------------",
            ]
        )
        labels = {
            "photopeak": "Photopeak",
            "general_scatter": "General scatter",
            "all_four_windows": "Four windows",
        }
        for key in ("photopeak", "general_scatter", "all_four_windows"):
            fit = result["fits"][key]
            lines.append(
                f"{labels[key]:19s} | {fit['slope_counts_per_mbq']:16.3f} | "
                f"{fit['r_squared_zero_origin']:9.4f} | "
                f"{fit['cv_pointwise_counts_per_mbq_percent']:15.2f}%"
            )
    else:
        lines.extend(
            [
                "",
                "Urine-derived retention",
                "-------------------------",
                "Unavailable for this cycle. Counts and spectral ratios are still",
                "reported, but no counts-versus-activity fit is calculated.",
            ]
        )
    lines.extend(
        [
            "",
            "Photopeak/general-scatter ratio: all timepoints",
            "------------------------------------------------",
            f"Mean: {summary['mean']:.6f}",
            f"Sample SD: {summary['sample_sd']:.6f}",
            f"CV: {summary['cv_percent']:.2f}%",
            f"Range: {summary['minimum']:.6f} to {summary['maximum']:.6f}",
            f"First-to-last change: {summary['first_to_last_change_percent']:+.2f}%",
            "",
            "Values",
            "------",
            " TP | time from first h | photopeak | general scatter | four windows | P/G",
            "--- | ----------------- | --------- | --------------- | ------------ | -----",
        ]
    )
    for row in rows:
        lines.append(
            f"{row['timepoint']:>3s} | {row['hours_after_first_planar']:17.2f} | "
            f"{row['photopeak_counts']:9.0f} | "
            f"{row['general_scatter_counts']:15.0f} | "
            f"{row['all_four_windows_counts']:12.0f} | "
            f"{row['photopeak_over_general_scatter']:.5f}"
        )
    if urine_available:
        lines.extend(
            [
                "",
                "Urine-derived remaining activity at planar times",
                "--------------------------------------------------",
                " TP | time after injection h | remaining MBq",
                "--- | ---------------------- | -------------",
            ]
        )
        for row in rows:
            lines.append(
                f"{row['timepoint']:>3s} | {row['hours_after_injection']:22.2f} | "
                f"{row['urine_derived_remaining_activity_mbq']:13.2f}"
            )
    lines.extend(
        [
            "",
            "Interpretation limit",
            "--------------------",
            "Counts and ratios are descriptive image measurements. Where urine is",
            "available, the three-point fit is a screening result rather than a",
            "validated calibration. Ratio stability should be judged from both its",
            "CV and its systematic first-to-last change.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def run_analysis(
    planar_dir: Path = DEFAULT_PLANAR_DIR,
    urine_csv: Optional[Path] = DEFAULT_URINE_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    comparison_timepoints: int = 3,
    patient_label: str = "P11",
    cycle_label: str = "cycle de mai 2026",
    combination: str = DEFAULT_COMBINATION,
    align_pa_to_ap: bool = planar_processing.ALIGN_PA_TO_AP,
) -> Dict[str, Any]:
    """Run one cycle, with urine retention when its CSV is available."""
    if urine_csv is None:
        result = analyze_planar_counts(
            planar_dir=planar_dir,
            combination=combination,
            align_pa_to_ap=align_pa_to_ap,
        )
    else:
        result = analyze_total_counts_retention(
            planar_dir=planar_dir,
            urine_csv=urine_csv,
            comparison_timepoints=comparison_timepoints,
            combination=combination,
            align_pa_to_ap=align_pa_to_ap,
        )
    result["patient_label"] = patient_label
    result["cycle_label"] = cycle_label
    result["planar_dir"] = Path(planar_dir)
    result["urine_csv"] = Path(urine_csv) if urine_csv is not None else None
    output_dir = Path(output_dir)
    paths = {
        "counts_vs_time_figure": plot_raw_counts_vs_time(
            result, output_dir / "raw_full_image_counts_vs_time.png"
        ),
        "ratio_figure": plot_ratio_vs_time(
            result, output_dir / "photopeak_general_scatter_ratio_vs_time.png"
        ),
        "values_csv": write_values_csv(
            result, output_dir / "raw_full_image_counts_values.csv"
        ),
        "report": write_report(
            result, output_dir / "raw_full_image_counts_report.txt"
        ),
    }
    if result["urine_available"]:
        paths.update(
            {
                "counts_vs_retention_figure": plot_counts_vs_retention(
                    result, output_dir / "raw_counts_vs_urine_retention.png"
                ),
                "normalized_time_figure": plot_normalized_counts_and_retention(
                    result,
                    output_dir / "normalized_counts_vs_urine_retention_time.png",
                ),
            }
        )
    result["paths"] = paths
    return result


def run_all_datasets(
    comparison_timepoints: int = 3,
) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """Generate AP_PA and GM subfolders for every configured patient/cycle."""
    results: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for (patient_id, cycle_id), configured in DATASETS.items():
        cycle_output_dir = figure_layout.patient_cycle_dir(
            __file__, patient_id, cycle_id
        )
        for combination in COMBINATIONS:
            results[(patient_id, cycle_id, combination)] = run_analysis(
                planar_dir=Path(configured["planar_dir"]),
                urine_csv=(
                    Path(configured["urine_csv"])
                    if configured["urine_csv"] is not None
                    else None
                ),
                output_dir=cycle_output_dir / combination,
                comparison_timepoints=comparison_timepoints,
                patient_label=str(configured["patient_label"]),
                cycle_label=str(configured["cycle_label"]),
                combination=combination,
            )
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-id", choices=("p11", "p8"))
    parser.add_argument("--cycle-id")
    parser.add_argument("--planar-dir", type=Path)
    parser.add_argument("--urine-csv", type=Path)
    parser.add_argument("--without-urine", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--combination", choices=COMBINATIONS)
    parser.add_argument("--comparison-timepoints", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    custom_single = any(
        value is not None
        for value in (
            args.patient_id,
            args.cycle_id,
            args.planar_dir,
            args.urine_csv,
            args.output_dir,
            args.combination,
        )
    ) or args.without_urine
    if not custom_single:
        results = run_all_datasets(args.comparison_timepoints)
    else:
        patient_id = args.patient_id or "p11"
        cycle_id = args.cycle_id or "2026-05__Studies"
        configured = DATASETS.get((patient_id, cycle_id), {})
        planar_dir = args.planar_dir or configured.get("planar_dir")
        if planar_dir is None:
            raise ValueError("--planar-dir is required for an unconfigured cycle")
        if args.without_urine:
            urine_csv = None
        elif args.urine_csv is not None:
            urine_csv = args.urine_csv
        else:
            urine_csv = configured.get("urine_csv")
        cycle_output_dir = args.output_dir or figure_layout.patient_cycle_dir(
            __file__, patient_id, cycle_id
        )
        combinations = (args.combination,) if args.combination else COMBINATIONS
        results = {}
        for combination in combinations:
            result = run_analysis(
                planar_dir=Path(planar_dir),
                urine_csv=Path(urine_csv) if urine_csv is not None else None,
                output_dir=Path(cycle_output_dir) / combination,
                comparison_timepoints=args.comparison_timepoints,
                patient_label=str(
                    configured.get("patient_label", patient_id.upper())
                ),
                cycle_label=str(configured.get("cycle_label", cycle_id)),
                combination=combination,
            )
            results[(patient_id, cycle_id, combination)] = result

    for (patient_id, cycle_id, combination), result in results.items():
        ratio = result["ratio_summary"]
        urine_note = "with urine" if result["urine_available"] else "image only"
        print(
            f"{patient_id}/{cycle_id}/{combination} ({urine_note}): "
            f"photopeak/general scatter "
            f"mean={ratio['mean']:.6f}, CV={ratio['cv_percent']:.2f}%, "
            f"first-to-last={ratio['first_to_last_change_percent']:+.2f}%"
        )
        for name, path in result["paths"].items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
