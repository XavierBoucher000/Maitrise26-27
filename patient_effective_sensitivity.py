"""Evaluate patient-specific planar sensitivity without CT attenuation correction.

The quantity calculated here is an *effective patient sensitivity*:

    S_eff(t) = planar count rate(t) / Q/SPECT activity(t)

It is not the intrinsic camera calibration factor.  Because no CT attenuation
correction is applied, S_eff includes patient attenuation, body habitus,
activity distribution, crop coverage, and any remaining acquisition bias.

Two otherwise identical measurements are reported:

* raw: photopeak AP/PA geometric mean without scatter subtraction;
* TEW: TEW correction applied independently to AP and PA, followed by their
  geometric mean.

The Q/SPECT activity is shifted to the planar acquisition time using physical
Lu-177 decay only.  This correction is small for the paired acquisitions and
does not model biological clearance between the two scans.
"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

import attenuation_correction as ac
import dicom_loader
import planar_processing
import planar_qspect_crop
import planar_qspect_profile_crop
import qspect_processing


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "fig" / "patient_effective_sensitivity"
DEFAULT_PATIENT_LABEL = "Patient 1"
LU177_PHYSICAL_HALF_LIFE_H = 159.5
REFERENCE_CAMERA_SENSITIVITY_CPS_PER_MBQ = (
    planar_processing.CAMERA_SENSITIVITY_CPS_PER_MBQ
)


def activity_at_target_time_physical_decay(
    activity_mbq: float,
    reference_time: datetime,
    target_time: datetime,
    half_life_h: float = LU177_PHYSICAL_HALF_LIFE_H,
) -> float:
    """Move an activity from ``reference_time`` to ``target_time`` by decay only."""
    activity_mbq = float(activity_mbq)
    half_life_h = float(half_life_h)
    if not np.isfinite(activity_mbq) or activity_mbq <= 0.0:
        raise ValueError("activity_mbq must be finite and positive")
    if not np.isfinite(half_life_h) or half_life_h <= 0.0:
        raise ValueError("half_life_h must be finite and positive")
    elapsed_h = (target_time - reference_time).total_seconds() / 3600.0
    return activity_mbq * math.pow(2.0, -elapsed_h / half_life_h)


def effective_sensitivity_cps_per_mbq(
    counts: float,
    local_dwell_time_s: float,
    reference_activity_mbq: float,
) -> float:
    """Return count rate divided by reference activity in cps/MBq."""
    counts = float(counts)
    local_dwell_time_s = float(local_dwell_time_s)
    reference_activity_mbq = float(reference_activity_mbq)
    if not np.isfinite(counts) or counts < 0.0:
        raise ValueError("counts must be finite and nonnegative")
    if not np.isfinite(local_dwell_time_s) or local_dwell_time_s <= 0.0:
        raise ValueError("local_dwell_time_s must be finite and positive")
    if not np.isfinite(reference_activity_mbq) or reference_activity_mbq <= 0.0:
        raise ValueError("reference_activity_mbq must be finite and positive")
    return counts / local_dwell_time_s / reference_activity_mbq


def stability_metrics(
    values: Sequence[float],
    times_days: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Return descriptive intra-patient stability metrics for a sensitivity series."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("values must be a non-empty finite one-dimensional series")
    if np.any(array <= 0.0):
        raise ValueError("sensitivity values must be positive")

    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    metrics = {
        "n": float(array.size),
        "mean_cps_per_mbq": mean,
        "sd_cps_per_mbq": sd,
        "cv_percent": 100.0 * sd / mean,
        "minimum_cps_per_mbq": float(np.min(array)),
        "maximum_cps_per_mbq": float(np.max(array)),
        "relative_range_percent": 100.0 * float(np.ptp(array)) / mean,
        "last_over_first": float(array[-1] / array[0]),
    }

    if times_days is None or array.size < 2:
        metrics["slope_cps_per_mbq_per_day"] = math.nan
        metrics["linear_r_squared"] = math.nan
        return metrics

    times = np.asarray(times_days, dtype=np.float64)
    if times.shape != array.shape or not np.all(np.isfinite(times)):
        raise ValueError("times_days must contain one finite value per sensitivity")
    if float(np.ptp(times)) <= 0.0:
        raise ValueError("times_days must span more than one time")

    slope, intercept = np.polyfit(times, array, 1)
    fitted = slope * times + intercept
    residual_sum_squares = float(np.sum((array - fitted) ** 2))
    total_sum_squares = float(np.sum((array - mean) ** 2))
    r_squared = (
        1.0 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 0.0
        else 1.0
    )
    metrics["slope_cps_per_mbq_per_day"] = float(slope)
    metrics["linear_r_squared"] = float(r_squared)
    return metrics


def _pair_acquisitions_by_time(
    planar_scans: Sequence[Dict[str, Any]],
    qspect_series: Sequence[Dict[str, Any]],
    max_time_difference_h: float,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Pair each planar scan to the nearest unused Q/SPECT acquisition."""
    remaining = list(qspect_series)
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for scan in planar_scans:
        planar_time = dicom_loader.scan_datetime(scan)
        if planar_time is None:
            raise ValueError(f"Missing planar acquisition time for {scan['scan_name']}")
        candidates = [
            item for item in remaining if item.get("acquisition_datetime") is not None
        ]
        if not candidates:
            raise ValueError("No unused Q/SPECT acquisition remains for time pairing")
        qspect = min(
            candidates,
            key=lambda item: abs(
                (item["acquisition_datetime"] - planar_time).total_seconds()
            ),
        )
        difference_h = abs(
            (qspect["acquisition_datetime"] - planar_time).total_seconds()
        ) / 3600.0
        if difference_h > max_time_difference_h:
            raise ValueError(
                f"Nearest Q/SPECT is {difference_h:.2f} h from planar scan; "
                f"maximum allowed is {max_time_difference_h:.2f} h"
            )
        pairs.append((scan, qspect))
        remaining.remove(qspect)
    return pairs


def _emission_images(
    scan: Dict[str, Any],
    align_pa_to_ap: bool,
) -> Dict[str, Any]:
    """Build raw and per-view-TEW AP/PA geometric-mean images."""
    groups = ac.group_by_energy_and_view(scan["images"])
    photopeak = groups.get("Photopeak")
    if not photopeak or "AP" not in photopeak or "PA" not in photopeak:
        raise ValueError("Photopeak AP and PA images are required")

    raw_ap = np.asarray(photopeak["AP"]["image"], dtype=np.float64)
    raw_pa = np.asarray(photopeak["PA"]["image"], dtype=np.float64)
    tew_ap = ac.tew_correct_view_image(groups, "AP")
    tew_pa = ac.tew_correct_view_image(groups, "PA")
    raw_gm = ac.geometric_mean(raw_ap, raw_pa, align_pa_to_ap=align_pa_to_ap)
    tew_gm = ac.geometric_mean(tew_ap, tew_pa, align_pa_to_ap=align_pa_to_ap)
    timing = planar_processing.planar_timing_from_dicom(scan["images"])
    return {
        "raw_ap": raw_ap,
        "raw_pa": raw_pa,
        "tew_ap": tew_ap,
        "tew_pa": tew_pa,
        "raw_gm": raw_gm,
        "tew_gm": tew_gm,
        "timing": timing,
    }


def calculate_patient_effective_sensitivity(
    planar_dir: Path = planar_processing.default_planar_study_dir(),
    qspect_dir: Path = qspect_processing.default_qspect_dir(),
    crop_strategy: str = "profile_qspect",
    threshold_fraction: float = planar_qspect_crop.PLANAR_QSPECT_CROP_THRESHOLD,
    align_pa_to_ap: bool = planar_processing.ALIGN_PA_TO_AP,
    half_life_h: float = LU177_PHYSICAL_HALF_LIFE_H,
    max_time_difference_h: float = 6.0,
) -> List[Dict[str, Any]]:
    """Calculate raw and TEW effective sensitivities without loading any CT."""
    if crop_strategy not in {"profile_qspect", "fixed_day0", "individual"}:
        raise ValueError(
            "crop_strategy must be 'profile_qspect', 'fixed_day0' or 'individual'"
        )

    planar_scans = ac.sorted_planar_scans(Path(planar_dir))
    qspect_series = qspect_processing.load_qspect_study(Path(qspect_dir))
    if not planar_scans or not qspect_series:
        raise ValueError("Planar and Q/SPECT acquisitions are required")
    pairs = _pair_acquisitions_by_time(
        planar_scans,
        qspect_series,
        max_time_difference_h=max_time_difference_h,
    )

    prepared = [
        (scan, qspect, _emission_images(scan, align_pa_to_ap=align_pa_to_ap))
        for scan, qspect in pairs
    ]
    first_planar_time = dicom_loader.scan_datetime(prepared[0][0])
    if first_planar_time is None:
        raise ValueError("First planar acquisition time is unavailable")

    fixed_bounds: Optional[Tuple[int, int]] = None
    profile_matches: Optional[List[Dict[str, Any]]] = None
    if crop_strategy == "profile_qspect":
        profile_matches = planar_qspect_profile_crop.resolve_profile_crop_matches(
            [scan for scan, _qspect, _images in prepared],
            [qspect for _scan, qspect, _images in prepared],
            threshold_fraction=threshold_fraction,
            align_pa_to_ap=align_pa_to_ap,
        )
    if crop_strategy == "fixed_day0":
        first_scan, first_qspect, first_images = prepared[0]
        first_record = {
            "images": first_scan["images"],
            "correction": {"corrected_image": {"image": first_images["tew_gm"]}},
        }
        first_crop = planar_qspect_crop.compute_planar_crop_for_qspect(
            first_images["tew_gm"],
            first_record,
            first_qspect,
            threshold_fraction=threshold_fraction,
        )
        fixed_bounds = (int(first_crop["crop_top"]), int(first_crop["crop_bottom"]))

    rows: List[Dict[str, Any]] = []
    for index, (scan, qspect, images) in enumerate(prepared):
        planar_time = dicom_loader.scan_datetime(scan)
        qspect_time = qspect.get("acquisition_datetime")
        if planar_time is None or qspect_time is None:
            raise ValueError("Paired acquisition times are required")
        qspect_activity_mbq = (
            None
            if qspect.get("total_activity_bq") is None
            else float(qspect["total_activity_bq"]) / 1e6
        )
        if qspect_activity_mbq is None or qspect_activity_mbq <= 0.0:
            raise ValueError(f"Invalid Q/SPECT activity for {qspect.get('label', index)}")

        planar_record = {
            "images": scan["images"],
            "correction": {"corrected_image": {"image": images["tew_gm"]}},
        }
        selected_bounds = fixed_bounds
        match = None if profile_matches is None else profile_matches[index]
        if match is not None:
            selected_bounds = (int(match["crop_top"]), int(match["crop_bottom"]))
        crop = planar_qspect_crop.compute_planar_crop_for_qspect(
            images["tew_gm"],
            planar_record,
            qspect,
            threshold_fraction=threshold_fraction,
            fixed_crop_bounds=selected_bounds,
        )
        if match is not None:
            crop["crop_strategy"] = "profile_qspect"
        top = int(crop["crop_top"])
        bottom = int(crop["crop_bottom"])
        raw_counts = float(np.sum(images["raw_gm"][top:bottom, :]))
        tew_counts = float(np.sum(images["tew_gm"][top:bottom, :]))
        dwell_s = float(images["timing"].local_dwell_time_s)
        raw_cps = raw_counts / dwell_s
        tew_cps = tew_counts / dwell_s
        activity_at_planar_mbq = activity_at_target_time_physical_decay(
            qspect_activity_mbq,
            reference_time=qspect_time,
            target_time=planar_time,
            half_life_h=half_life_h,
        )
        raw_sensitivity = effective_sensitivity_cps_per_mbq(
            raw_counts,
            dwell_s,
            activity_at_planar_mbq,
        )
        tew_sensitivity = effective_sensitivity_cps_per_mbq(
            tew_counts,
            dwell_s,
            activity_at_planar_mbq,
        )
        time_difference_h = (qspect_time - planar_time).total_seconds() / 3600.0
        rows.append(
            {
                "index": index,
                "label": str(qspect.get("label", f"Timepoint {index + 1}")),
                "planar_scan_name": str(scan["scan_name"]),
                "planar_datetime": planar_time,
                "qspect_datetime": qspect_time,
                "planar_day": (planar_time - first_planar_time).total_seconds() / 86400.0,
                "qspect_minus_planar_h": time_difference_h,
                "qspect_activity_measured_mbq": qspect_activity_mbq,
                "qspect_activity_at_planar_time_mbq": activity_at_planar_mbq,
                "physical_time_adjustment_percent": 100.0
                * (activity_at_planar_mbq / qspect_activity_mbq - 1.0),
                "local_dwell_time_s": dwell_s,
                "crop_strategy": crop["crop_strategy"],
                "crop_top": top,
                "crop_bottom": bottom,
                "crop_height_cm": float(crop["crop_height_cm"]),
                "raw_photopeak_gm_counts": raw_counts,
                "tew_gm_counts": tew_counts,
                "raw_photopeak_gm_cps": raw_cps,
                "tew_gm_cps": tew_cps,
                "raw_effective_sensitivity_cps_per_mbq": raw_sensitivity,
                "tew_effective_sensitivity_cps_per_mbq": tew_sensitivity,
                "tew_retained_fraction": tew_counts / raw_counts,
                "tew_count_reduction_percent": 100.0 * (1.0 - tew_counts / raw_counts),
                "reference_camera_sensitivity_cps_per_mbq": (
                    REFERENCE_CAMERA_SENSITIVITY_CPS_PER_MBQ
                ),
                "ct_attenuation_correction_applied": False,
                "dead_time_correction_applied": False,
                "align_pa_to_ap": bool(align_pa_to_ap),
                "profile_match_correlation": (
                    np.nan if match is None else float(match["correlation"])
                ),
                "profile_match_accepted": (
                    None if match is None else bool(match["match_accepted"])
                ),
            }
        )
    return rows


def _csv_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    return value


def write_values_csv(rows: Sequence[Dict[str, Any]], output_path: Path) -> Path:
    """Write all per-timepoint values used in the sensitivity calculation."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("At least one sensitivity row is required")
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row[key]) for key in fieldnames})
    return output_path


def _metric_block(label: str, metrics: Dict[str, float]) -> List[str]:
    return [
        f"{label}:",
        f"  mean = {metrics['mean_cps_per_mbq']:.4f} cps/MBq",
        f"  SD = {metrics['sd_cps_per_mbq']:.4f} cps/MBq",
        f"  CV = {metrics['cv_percent']:.2f} %",
        f"  min-max = {metrics['minimum_cps_per_mbq']:.4f} - "
        f"{metrics['maximum_cps_per_mbq']:.4f} cps/MBq",
        f"  relative range = {metrics['relative_range_percent']:.2f} % of mean",
        f"  last/first = {metrics['last_over_first']:.4f}",
        f"  linear slope = {metrics['slope_cps_per_mbq_per_day']:.4f} "
        "cps/MBq/day",
        f"  linear R^2 = {metrics['linear_r_squared']:.4f}",
    ]


def write_report(
    rows: Sequence[Dict[str, Any]],
    output_path: Path,
    patient_label: str,
) -> Path:
    """Write a human-readable report with the stability results and limitations."""
    days = [float(row["planar_day"]) for row in rows]
    raw = [float(row["raw_effective_sensitivity_cps_per_mbq"]) for row in rows]
    tew = [float(row["tew_effective_sensitivity_cps_per_mbq"]) for row in rows]
    raw_metrics = stability_metrics(raw, days)
    tew_metrics = stability_metrics(tew, days)
    lines = [
        f"Patient-specific effective planar sensitivity — {patient_label}",
        "=" * (48 + len(patient_label)),
        "",
        "Definition:",
        "  S_eff = planar geometric-mean count rate / Q/SPECT activity",
        "  Units = cps/MBq",
        "  This is a patient-effective sensitivity, not an intrinsic camera calibration.",
        "",
        "Processing:",
        "  - no CT attenuation correction",
        "  - no dead-time correction",
        "  - raw curve: photopeak AP/PA geometric mean",
        "  - TEW curve: TEW(AP), TEW(PA), then AP/PA geometric mean",
        f"  - crop strategy: {rows[0]['crop_strategy']}",
        "  - Q/SPECT activity shifted to planar time by Lu-177 physical decay only",
        f"  - comparison line: {REFERENCE_CAMERA_SENSITIVITY_CPS_PER_MBQ:.2f} cps/MBq",
        "",
        "Timepoint values:",
        " label | day | QSP-planar h | QSP at planar MBq | raw cps/MBq | TEW cps/MBq | TEW reduction %",
        "------ | --- | ------------ | ----------------- | ----------- | ----------- | ---------------",
    ]
    for row in rows:
        lines.append(
            f"{row['label']:>6} | {row['planar_day']:5.2f} | "
            f"{row['qspect_minus_planar_h']:12.2f} | "
            f"{row['qspect_activity_at_planar_time_mbq']:17.1f} | "
            f"{row['raw_effective_sensitivity_cps_per_mbq']:11.4f} | "
            f"{row['tew_effective_sensitivity_cps_per_mbq']:11.4f} | "
            f"{row['tew_count_reduction_percent']:15.2f}"
        )
    lines.extend(["", *_metric_block("Raw photopeak", raw_metrics), ""])
    lines.extend([*_metric_block("TEW", tew_metrics), ""])
    lines.extend(
        [
            "Interpretation limits:",
            "  - profile_qspect fixes the Q/SPECT physical coverage and translates it",
            "    by bounded longitudinal-profile correlation with automatic QC fallback.",
            "  - physical decay matching does not correct biological clearance during the",
            "    time gap between planar and Q/SPECT acquisitions.",
            "  - changes across time can also arise from count-rate losses, redistribution,",
            "    scatter noise, crop mismatch, or acquisition normalization.",
            "  - one patient provides an intra-patient proof of concept only.",
        ]
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(True, linestyle="--", alpha=0.30)
    axis.spines[["top", "right"]].set_visible(False)


def plot_effective_sensitivity(
    rows: Sequence[Dict[str, Any]],
    output_path: Path,
    patient_label: str,
) -> Path:
    """Plot absolute raw and TEW patient-effective sensitivity over time."""
    days = np.asarray([row["planar_day"] for row in rows], dtype=np.float64)
    raw = np.asarray(
        [row["raw_effective_sensitivity_cps_per_mbq"] for row in rows],
        dtype=np.float64,
    )
    tew = np.asarray(
        [row["tew_effective_sensitivity_cps_per_mbq"] for row in rows],
        dtype=np.float64,
    )
    raw_metrics = stability_metrics(raw, days)
    tew_metrics = stability_metrics(tew, days)

    fig, axis = plt.subplots(figsize=(9.2, 5.7), facecolor="white", layout="constrained")
    axis.plot(
        days,
        raw,
        color="#7f7f7f",
        marker="o",
        linewidth=2.2,
        label=f"Sans correction du diffusé (CV={raw_metrics['cv_percent']:.1f} %)",
    )
    axis.plot(
        days,
        tew,
        color="#1f77b4",
        marker="s",
        linewidth=2.2,
        label=f"Avec TEW (CV={tew_metrics['cv_percent']:.1f} %)",
    )
    axis.axhline(
        REFERENCE_CAMERA_SENSITIVITY_CPS_PER_MBQ,
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        label=f"Sensibilité caméra de référence ({REFERENCE_CAMERA_SENSITIVITY_CPS_PER_MBQ:.2f} cps/MBq)",
    )
    axis.set_xlabel("Temps après la première acquisition planaire (jours)")
    axis.set_ylabel("Sensibilité effective sans CT (cps/MBq)")
    axis.set_title(f"Stabilité intra-patient de la sensibilité effective — {patient_label}")
    axis.legend(frameon=False, fontsize=8.5)
    _style_axis(axis)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_relative_stability(
    rows: Sequence[Dict[str, Any]],
    output_path: Path,
    patient_label: str,
) -> Path:
    """Plot each sensitivity relative to its own patient-specific temporal mean."""
    days = np.asarray([row["planar_day"] for row in rows], dtype=np.float64)
    raw = np.asarray(
        [row["raw_effective_sensitivity_cps_per_mbq"] for row in rows],
        dtype=np.float64,
    )
    tew = np.asarray(
        [row["tew_effective_sensitivity_cps_per_mbq"] for row in rows],
        dtype=np.float64,
    )
    raw_relative = 100.0 * raw / float(np.mean(raw))
    tew_relative = 100.0 * tew / float(np.mean(tew))

    fig, axis = plt.subplots(figsize=(9.2, 5.5), facecolor="white", layout="constrained")
    axis.axhspan(90.0, 110.0, color="#2ca02c", alpha=0.08, label="Bande indicative ±10 %")
    axis.axhline(100.0, color="black", linestyle="--", linewidth=1.4)
    axis.plot(
        days,
        raw_relative,
        color="#7f7f7f",
        marker="o",
        linewidth=2.2,
        label="Sans correction du diffusé",
    )
    axis.plot(
        days,
        tew_relative,
        color="#1f77b4",
        marker="s",
        linewidth=2.2,
        label="Avec TEW",
    )
    axis.set_xlabel("Temps après la première acquisition planaire (jours)")
    axis.set_ylabel("Sensibilité / moyenne du patient (%)")
    axis.set_title(f"Variation temporelle relative de la sensibilité — {patient_label}")
    axis.legend(frameon=False, fontsize=8.5)
    _style_axis(axis)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def run_patient_effective_sensitivity(
    planar_dir: Path = planar_processing.default_planar_study_dir(),
    qspect_dir: Path = qspect_processing.default_qspect_dir(),
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    patient_label: str = DEFAULT_PATIENT_LABEL,
    crop_strategy: str = "profile_qspect",
    align_pa_to_ap: bool = planar_processing.ALIGN_PA_TO_AP,
) -> Dict[str, Any]:
    """Run the complete one-patient analysis and save its reusable outputs."""
    output_dir = Path(output_dir)
    rows = calculate_patient_effective_sensitivity(
        planar_dir=Path(planar_dir),
        qspect_dir=Path(qspect_dir),
        crop_strategy=crop_strategy,
        align_pa_to_ap=align_pa_to_ap,
    )
    csv_path = write_values_csv(rows, output_dir / "effective_sensitivity_values.csv")
    report_path = write_report(
        rows,
        output_dir / "effective_sensitivity_report.txt",
        patient_label,
    )
    absolute_figure = plot_effective_sensitivity(
        rows,
        output_dir / "effective_sensitivity_vs_time.png",
        patient_label,
    )
    relative_figure = plot_relative_stability(
        rows,
        output_dir / "effective_sensitivity_relative_stability.png",
        patient_label,
    )
    return {
        "rows": rows,
        "csv": csv_path,
        "report": report_path,
        "absolute_figure": absolute_figure,
        "relative_figure": relative_figure,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Patient-specific planar effective sensitivity without CT correction"
    )
    parser.add_argument(
        "--planar-dir",
        type=Path,
        default=planar_processing.default_planar_study_dir(),
    )
    parser.add_argument(
        "--qspect-dir",
        type=Path,
        default=qspect_processing.default_qspect_dir(),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--patient-label", default=DEFAULT_PATIENT_LABEL)
    parser.add_argument(
        "--crop-strategy",
        choices=("profile_qspect", "fixed_day0", "individual"),
        default="profile_qspect",
    )
    parser.add_argument(
        "--align-pa-to-ap",
        action=argparse.BooleanOptionalAction,
        default=planar_processing.ALIGN_PA_TO_AP,
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    args = _build_parser().parse_args(argv)
    result = run_patient_effective_sensitivity(
        planar_dir=args.planar_dir,
        qspect_dir=args.qspect_dir,
        output_dir=args.output_dir,
        patient_label=args.patient_label,
        crop_strategy=args.crop_strategy,
        align_pa_to_ap=args.align_pa_to_ap,
    )
    print(f"Saved: {result['absolute_figure']}")
    print(f"Saved: {result['relative_figure']}")
    print(f"Saved: {result['csv']}")
    print(f"Saved: {result['report']}")
    return result


if __name__ == "__main__":
    main()
