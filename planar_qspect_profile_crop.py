"""Experimental longitudinal crop registration between planar and Q/SPECT.

This module tests a deliberately simple one-dimensional method:

1. form the TEW-corrected AP/PA geometric-mean planar image;
2. form the coronal mean projection of the paired Q/SPECT volume;
3. sum each image horizontally to obtain a longitudinal activity profile;
4. resample the Q/SPECT profile to the planar row spacing;
5. smooth and log-compress both profiles;
6. translate a crop of Q/SPECT physical length within a bounded search range;
7. retain the translation that maximizes normalized cross-correlation.

Until a marker-based machine calibration is available, the existing Day-0
crop supplies only the initial position.  The result is therefore an
exploratory image-registration test, not a validated DICOM-coordinate crop.
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
from scipy.ndimage import gaussian_filter1d

import attenuation_correction as ac
import planar_processing
import planar_qspect_crop
import qspect_processing
from plots import view_patient_images


OUTPUT_DIR = Path(__file__).resolve().parent / "fig" / "crop_profile_matching"
DETAIL_FIGURE_PATH = OUTPUT_DIR / "planar_qspect_longitudinal_profile_matches.png"
SUMMARY_FIGURE_PATH = OUTPUT_DIR / "planar_qspect_profile_match_summary.png"
BOUNDS_COMPARISON_PATH = OUTPUT_DIR / "crop_bounds_fixed_vs_profile.png"
ACTIVITY_COMPARISON_PATH = OUTPUT_DIR / "ctac_activity_fixed_vs_profile.png"
RATIO_COMPARISON_PATH = OUTPUT_DIR / "ctac_qspect_ratio_fixed_vs_profile.png"
DIFFERENCE_COMPARISON_PATH = OUTPUT_DIR / "crop_method_differences.png"
FULL_GEOMETRY_PATH = OUTPUT_DIR / "planar_full_qspect_ct_geometry.png"
CROPPED_GEOMETRY_PATH = OUTPUT_DIR / "planar_crop_qspect_ct_geometry.png"
GEOMETRY_VALUES_PATH = OUTPUT_DIR / "planar_qspect_ct_lengths.csv"
GEOMETRY_REPORT_PATH = OUTPUT_DIR / "planar_qspect_ct_lengths_report.txt"
VALUES_PATH = OUTPUT_DIR / "planar_qspect_profile_match_values.csv"
REPORT_PATH = OUTPUT_DIR / "planar_qspect_profile_match_report.txt"

MAX_SHIFT_CM = 20.0
SMOOTHING_SIGMA_CM = 2.0
EXCLUSION_RADIUS_PX = 5
MIN_ACCEPTED_CORRELATION = 0.90
MIN_ACCEPTED_PEAK_MARGIN = 0.01
MAX_ACCEPTED_PEAK_WIDTH_CM = 5.0
PEAK_WIDTH_CORRELATION_LOSS = 0.01


def _positive_finite(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(
        np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    return np.clip(values, 0.0, None)


def horizontal_sum_profile(image: np.ndarray) -> np.ndarray:
    """Return P(row) = sum over image columns."""
    image = _positive_finite(image)
    if image.ndim != 2:
        raise ValueError("A longitudinal profile requires a 2D image")
    profile = np.sum(image, axis=1)
    if not np.any(profile > 0.0):
        raise ValueError("Cannot construct a profile from an empty image")
    return profile


def resample_profile(
    profile: np.ndarray,
    source_spacing_mm: float,
    target_spacing_mm: float,
) -> np.ndarray:
    """Resample a profile by physical position while preserving its coverage."""
    profile = _positive_finite(profile)
    if profile.ndim != 1 or profile.size < 2:
        raise ValueError("Profile must contain at least two samples")
    if source_spacing_mm <= 0.0 or target_spacing_mm <= 0.0:
        raise ValueError("Profile spacings must be positive")

    physical_length_mm = profile.size * float(source_spacing_mm)
    target_size = max(2, int(np.rint(physical_length_mm / target_spacing_mm)))
    source_positions = (np.arange(profile.size, dtype=np.float64) + 0.5) * source_spacing_mm
    target_positions = (np.arange(target_size, dtype=np.float64) + 0.5) * target_spacing_mm
    return np.interp(
        target_positions,
        source_positions,
        profile,
        left=float(profile[0]),
        right=float(profile[-1]),
    )


def robust_profile(profile: np.ndarray, smoothing_sigma_px: float) -> np.ndarray:
    """Log-compress, smooth and normalize a non-negative profile."""
    profile = _positive_finite(profile)
    positive = profile[profile > 0.0]
    if positive.size == 0:
        raise ValueError("Cannot normalize an empty profile")
    scale = float(np.percentile(positive, 95.0))
    transformed = np.log1p(profile / max(scale, 1e-12))
    if smoothing_sigma_px > 0.0:
        transformed = gaussian_filter1d(
            transformed, float(smoothing_sigma_px), mode="nearest"
        )
    maximum = float(np.max(transformed))
    return transformed / maximum if maximum > 0.0 else transformed


def normalized_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Pearson-equivalent normalized cross-correlation for equal vectors."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("Correlation requires equal-length 1D profiles")
    first = first - float(np.mean(first))
    second = second - float(np.mean(second))
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return np.nan if denominator <= 0.0 else float(np.dot(first, second) / denominator)


def search_profile_crop(
    planar_profile: np.ndarray,
    qspect_profile: np.ndarray,
    initial_top: int,
    max_shift_px: int,
    exclusion_radius_px: int = EXCLUSION_RADIUS_PX,
) -> Dict[str, Any]:
    """Find the translated Q/SPECT-length planar crop with maximum correlation."""
    planar_profile = np.asarray(planar_profile, dtype=np.float64)
    qspect_profile = np.asarray(qspect_profile, dtype=np.float64)
    crop_height = int(qspect_profile.size)
    if crop_height > planar_profile.size:
        raise ValueError("Q/SPECT physical coverage exceeds the planar matrix")
    maximum_top = planar_profile.size - crop_height
    search_min = max(0, int(initial_top) - int(max_shift_px))
    search_max = min(maximum_top, int(initial_top) + int(max_shift_px))
    candidate_tops = np.arange(search_min, search_max + 1, dtype=int)
    scores = np.asarray(
        [
            normalized_correlation(
                planar_profile[top : top + crop_height], qspect_profile
            )
            for top in candidate_tops
        ],
        dtype=np.float64,
    )
    if not np.any(np.isfinite(scores)):
        raise ValueError("No valid planar/Q/SPECT profile correlation was obtained")
    best_index = int(np.nanargmax(scores))
    best_top = int(candidate_tops[best_index])

    separated = np.abs(candidate_tops - best_top) > int(exclusion_radius_px)
    separated_scores = scores[separated & np.isfinite(scores)]
    second_peak = float(np.max(separated_scores)) if separated_scores.size else np.nan
    best_score = float(scores[best_index])
    return {
        "crop_top": best_top,
        "crop_bottom": best_top + crop_height,
        "crop_height_px": crop_height,
        "shift_px": best_top - int(initial_top),
        "correlation": best_score,
        "second_peak_correlation": second_peak,
        "peak_margin": best_score - second_peak if np.isfinite(second_peak) else np.nan,
        "search_edge_hit": bool(best_index in (0, candidate_tops.size - 1)),
        "candidate_tops": candidate_tops,
        "scores": scores,
    }


def add_match_quality_control(
    registration: Dict[str, Any],
    planar_spacing_mm: float,
    minimum_correlation: float = MIN_ACCEPTED_CORRELATION,
    minimum_peak_margin: float = MIN_ACCEPTED_PEAK_MARGIN,
    maximum_peak_width_cm: float = MAX_ACCEPTED_PEAK_WIDTH_CM,
    peak_width_correlation_loss: float = PEAK_WIDTH_CORRELATION_LOSS,
) -> Dict[str, Any]:
    """Measure peak width, accept a reliable match or use its initial bounds.

    Peak width is a deterministic quality-control width, not a statistical
    confidence interval.  It spans the contiguous translations around the
    optimum whose correlation is within ``peak_width_correlation_loss`` of the
    maximum.
    """
    result = dict(registration)
    scores = np.asarray(result["scores"], dtype=np.float64)
    tops = np.asarray(result["candidate_tops"], dtype=int)
    best_top = int(result["crop_top"])
    best_index = int(np.flatnonzero(tops == best_top)[0])
    threshold = float(result["correlation"]) - float(peak_width_correlation_loss)
    left = best_index
    right = best_index
    while left > 0 and np.isfinite(scores[left - 1]) and scores[left - 1] >= threshold:
        left -= 1
    while right + 1 < scores.size and np.isfinite(scores[right + 1]) and scores[right + 1] >= threshold:
        right += 1
    peak_width_px = int(tops[right] - tops[left])
    peak_width_cm = peak_width_px * float(planar_spacing_mm) / 10.0
    half_width_cm = 0.5 * peak_width_cm

    reasons = []
    if float(result["correlation"]) < float(minimum_correlation):
        reasons.append("correlation_below_threshold")
    if not np.isfinite(result["peak_margin"]) or float(result["peak_margin"]) < float(minimum_peak_margin):
        reasons.append("peak_margin_below_threshold")
    if bool(result["search_edge_hit"]):
        reasons.append("search_edge_hit")
    if peak_width_cm > float(maximum_peak_width_cm):
        reasons.append("peak_too_wide")

    proposed_top = int(result["crop_top"])
    proposed_bottom = int(result["crop_bottom"])
    accepted = not reasons
    if not accepted:
        result["crop_top"] = int(result["initial_top"])
        result["crop_bottom"] = int(result["initial_bottom"])
        result["shift_px"] = 0
    result.update(
        {
            "proposed_crop_top": proposed_top,
            "proposed_crop_bottom": proposed_bottom,
            "proposed_shift_px": proposed_top - int(result["initial_top"]),
            "match_accepted": bool(accepted),
            "rejection_reasons": reasons,
            "peak_width_px": peak_width_px,
            "peak_width_cm": peak_width_cm,
            "peak_half_width_cm": half_width_cm,
            "peak_width_left_top": int(tops[left]),
            "peak_width_right_top": int(tops[right]),
            "peak_width_correlation_loss": float(peak_width_correlation_loss),
            "minimum_accepted_correlation": float(minimum_correlation),
            "minimum_accepted_peak_margin": float(minimum_peak_margin),
            "maximum_accepted_peak_width_cm": float(maximum_peak_width_cm),
        }
    )
    return result


def planar_spacing_mm(scan: Dict[str, Any]) -> float:
    for image in scan["images"]:
        spacing = image.get("pixel_spacing")
        if spacing is not None and len(spacing) >= 1:
            return float(spacing[0])
    raise ValueError("Planar PixelSpacing is missing")


def qspect_slice_spacing_mm(qspect: Dict[str, Any]) -> float:
    """Estimate slice-center spacing; fall back to SliceThickness."""
    files = sorted(
        qspect["series_dir"].glob("*.dcm"),
        key=qspect_processing.sort_key_for_slice,
    )
    positions = []
    normal = None
    slice_thickness = None
    for path in files:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        if slice_thickness is None and getattr(ds, "SliceThickness", None) is not None:
            slice_thickness = float(ds.SliceThickness)
        orientation = getattr(ds, "ImageOrientationPatient", None)
        position = getattr(ds, "ImagePositionPatient", None)
        if orientation is None or position is None:
            continue
        orientation = np.asarray(orientation, dtype=np.float64)
        if normal is None:
            normal = np.cross(orientation[:3], orientation[3:])
            normal /= np.linalg.norm(normal)
        positions.append(float(np.dot(np.asarray(position, dtype=np.float64), normal)))
    if len(positions) >= 2:
        differences = np.abs(np.diff(np.sort(np.asarray(positions))))
        differences = differences[differences > 1e-6]
        if differences.size:
            return float(np.median(differences))
    if slice_thickness is None:
        raise ValueError("Cannot determine Q/SPECT slice spacing")
    return float(slice_thickness)


def volume_geometry(series_dir: Path, volume_shape: Tuple[int, ...]) -> Dict[str, float]:
    """Return DICOM center range and edge-to-edge coverage for a volume."""
    files = sorted(series_dir.glob("*.dcm"), key=qspect_processing.sort_key_for_slice)
    if not files:
        raise ValueError(f"No DICOM files found in {series_dir}")
    datasets = [
        pydicom.dcmread(path, stop_before_pixels=True, force=True) for path in files
    ]
    first = datasets[0]
    orientation = np.asarray(first.ImageOrientationPatient, dtype=np.float64)
    normal = np.cross(orientation[:3], orientation[3:])
    normal /= np.linalg.norm(normal)
    centers = np.asarray(
        [
            float(np.dot(np.asarray(ds.ImagePositionPatient, dtype=np.float64), normal))
            for ds in datasets
        ]
    )
    center_differences = np.abs(np.diff(np.sort(centers)))
    center_differences = center_differences[center_differences > 1e-6]
    thickness_mm = float(first.SliceThickness)
    spacing_mm = (
        float(np.median(center_differences))
        if center_differences.size
        else thickness_mm
    )
    center_range_mm = float(np.max(centers) - np.min(centers))
    pixel_spacing = [float(value) for value in first.PixelSpacing]
    return {
        "slice_count": float(len(datasets)),
        "slice_spacing_mm": spacing_mm,
        "slice_thickness_mm": thickness_mm,
        "center_range_mm": center_range_mm,
        "coverage_mm": center_range_mm + thickness_mm,
        "width_mm": float(volume_shape[2]) * pixel_spacing[1],
        "depth_mm": float(volume_shape[1]) * pixel_spacing[0],
        "minimum_center_coordinate_mm": float(np.min(centers)),
        "maximum_center_coordinate_mm": float(np.max(centers)),
    }


def day0_initial_crop(
    scan: Dict[str, Any],
    qspect: Dict[str, Any],
    threshold_fraction: float = planar_qspect_crop.PLANAR_QSPECT_CROP_THRESHOLD,
    align_pa_to_ap: bool = True,
) -> Tuple[int, int]:
    """Use the existing Day-0 crop only as the provisional mechanical seed."""
    planar = ac.tew_geometric_mean_for_scan(
        scan, align_pa_to_ap=align_pa_to_ap
    )["image"]
    planar_record = {
        "images": scan["images"],
        "correction": {"corrected_image": {"image": planar}},
    }
    result = planar_qspect_crop.compute_planar_crop_for_qspect(
        planar, planar_record, qspect, threshold_fraction=threshold_fraction
    )
    return int(result["crop_top"]), int(result["crop_bottom"])


def match_one_pair(
    scan: Dict[str, Any],
    qspect: Dict[str, Any],
    initial_center_y: float,
    max_shift_cm: float = MAX_SHIFT_CM,
    smoothing_sigma_cm: float = SMOOTHING_SIGMA_CM,
    align_pa_to_ap: bool = True,
) -> Dict[str, Any]:
    """Calculate the bounded planar/Q/SPECT longitudinal-profile match."""
    planar_image = ac.tew_geometric_mean_for_scan(
        scan, align_pa_to_ap=align_pa_to_ap
    )["image"]
    qspect_image = ac.qspect_coronal_projections(qspect)["mean_projection"]
    planar_spacing = planar_spacing_mm(scan)
    qspect_spacing = qspect_slice_spacing_mm(qspect)

    raw_planar_profile = horizontal_sum_profile(planar_image)
    raw_qspect_profile = horizontal_sum_profile(qspect_image)
    resampled_qspect_profile = resample_profile(
        raw_qspect_profile, qspect_spacing, planar_spacing
    )
    sigma_px = float(smoothing_sigma_cm) * 10.0 / planar_spacing
    planar_profile = robust_profile(raw_planar_profile, sigma_px)
    qspect_profile = robust_profile(resampled_qspect_profile, sigma_px)

    initial_top = int(np.rint(initial_center_y - qspect_profile.size / 2.0))
    initial_top = min(
        max(0, initial_top), planar_profile.size - qspect_profile.size
    )
    max_shift_px = int(np.rint(float(max_shift_cm) * 10.0 / planar_spacing))
    registration = search_profile_crop(
        planar_profile,
        qspect_profile,
        initial_top=initial_top,
        max_shift_px=max_shift_px,
    )
    registration.update(
        {
            "label": str(qspect["label"]),
            "scan_name": str(scan["scan_name"]),
            "planar_image": planar_image,
            "qspect_image": qspect_image,
            "planar_profile": planar_profile,
            "qspect_profile": qspect_profile,
            "initial_top": initial_top,
            "initial_bottom": initial_top + qspect_profile.size,
            "planar_spacing_mm": planar_spacing,
            "qspect_spacing_mm": qspect_spacing,
            "crop_height_cm": qspect_profile.size * planar_spacing / 10.0,
            "shift_cm": registration["shift_px"] * planar_spacing / 10.0,
            "search_limit_cm": max_shift_px * planar_spacing / 10.0,
            "smoothing_sigma_cm": float(smoothing_sigma_cm),
        }
    )
    registration = add_match_quality_control(registration, planar_spacing)
    registration["shift_cm"] = registration["shift_px"] * planar_spacing / 10.0
    registration["proposed_shift_cm"] = (
        registration["proposed_shift_px"] * planar_spacing / 10.0
    )
    return registration


def calculate_matches(
    max_shift_cm: float = MAX_SHIFT_CM,
    smoothing_sigma_cm: float = SMOOTHING_SIGMA_CM,
    initial_crop_bounds: Optional[Tuple[int, int]] = None,
    threshold_fraction: float = planar_qspect_crop.PLANAR_QSPECT_CROP_THRESHOLD,
    align_pa_to_ap: bool = True,
) -> List[Dict[str, Any]]:
    planar_dir = planar_processing.default_planar_study_dir()
    qspect_dir = qspect_processing.default_qspect_dir()
    scans = ac.sorted_planar_scans(planar_dir)
    qspect_series = qspect_processing.load_qspect_study(qspect_dir)
    count = min(len(scans), len(qspect_series))
    if count == 0:
        raise ValueError("No paired planar/Q/SPECT acquisitions were found")

    if initial_crop_bounds is None:
        seed_top, seed_bottom = day0_initial_crop(
            scans[0],
            qspect_series[0],
            threshold_fraction=threshold_fraction,
            align_pa_to_ap=align_pa_to_ap,
        )
        initial_position_source = "existing_day0_crop"
    else:
        seed_top, seed_bottom = (int(initial_crop_bounds[0]), int(initial_crop_bounds[1]))
        image_height = int(scans[0]["images"][0]["rows"])
        if seed_top < 0 or seed_bottom <= seed_top or seed_bottom > image_height:
            raise ValueError(f"Invalid initial crop bounds: {initial_crop_bounds}")
        initial_position_source = "external_mechanical_calibration"
    initial_center_y = 0.5 * (seed_top + seed_bottom)
    offsets = qspect_processing.day_offsets(qspect_series[:count])
    rows = [
        match_one_pair(
            scans[index],
            qspect_series[index],
            initial_center_y=initial_center_y,
            max_shift_cm=max_shift_cm,
            smoothing_sigma_cm=smoothing_sigma_cm,
            align_pa_to_ap=align_pa_to_ap,
        )
        for index in range(count)
    ]
    for row, day in zip(rows, offsets):
        row["day_offset"] = float(day)
        row["seed_day0_top"] = int(seed_top)
        row["seed_day0_bottom"] = int(seed_bottom)
        row["initial_position_source"] = initial_position_source
    return rows


def add_ctac_comparison(rows: List[Dict[str, Any]]) -> None:
    """Calculate fixed-J0 and profile-matched CTAC results for each pair.

    The function mutates ``rows`` by adding scalar comparison values.  Large
    CT and image arrays returned by the existing pipeline are deliberately not
    retained here.
    """
    planar_dir = planar_processing.default_planar_study_dir()
    qspect_dir = qspect_processing.default_qspect_dir()
    scans = ac.sorted_planar_scans(planar_dir)
    qspect_series = qspect_processing.load_qspect_study(qspect_dir)
    if len(rows) > min(len(scans), len(qspect_series)):
        raise ValueError("Not enough paired acquisitions for CTAC comparison")

    for row, scan, qspect in zip(rows, scans, qspect_series):
        fixed_bounds = (
            int(row["seed_day0_top"]),
            int(row["seed_day0_bottom"]),
        )
        profile_bounds = (int(row["crop_top"]), int(row["crop_bottom"]))
        fixed = ac.ct_attenuation_correct_scan(
            scan,
            qspect,
            qspect_dir,
            fixed_crop_bounds=fixed_bounds,
            align_pa_to_ap=True,
        )
        profile = ac.ct_attenuation_correct_scan(
            scan,
            qspect,
            qspect_dir,
            fixed_crop_bounds=profile_bounds,
            align_pa_to_ap=True,
        )
        qspect_activity = float(fixed["qspect_activity_mbq"])
        fixed_activity = float(fixed["ctac_local_activity_mbq"])
        profile_activity = float(profile["ctac_local_activity_mbq"])
        row.update(
            {
                "fixed_crop_top": fixed_bounds[0],
                "fixed_crop_bottom": fixed_bounds[1],
                "qspect_activity_mbq": qspect_activity,
                "fixed_ctac_activity_mbq": fixed_activity,
                "profile_ctac_activity_mbq": profile_activity,
                "fixed_ctac_over_qspect": fixed_activity / qspect_activity,
                "profile_ctac_over_qspect": profile_activity / qspect_activity,
                "activity_difference_mbq": profile_activity - fixed_activity,
                "activity_difference_percent": 100.0
                * (profile_activity - fixed_activity)
                / fixed_activity,
                "ratio_difference": (profile_activity - fixed_activity)
                / qspect_activity,
                "fixed_effective_ct_factor": float(fixed["effective_ct_factor"]),
                "profile_effective_ct_factor": float(profile["effective_ct_factor"]),
                "fixed_outside_counts_percent": 100.0
                * float(fixed["crop_excluded_counts"]["outside_fraction"]),
                "profile_outside_counts_percent": 100.0
                * float(profile["crop_excluded_counts"]["outside_fraction"]),
            }
        )


def _display_vmax(image: np.ndarray) -> float:
    positive = np.asarray(image)[np.asarray(image) > 0.0]
    return 1.0 if positive.size == 0 else float(np.percentile(positive, 99.5))


def plot_detailed_matches(
    rows: Sequence[Dict[str, Any]], output_path: Path = DETAIL_FIGURE_PATH
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(rows), 3, figsize=(13.0, 3.5 * len(rows)), facecolor="white", layout="constrained"
    )
    if len(rows) == 1:
        axes = np.asarray([axes])
    for axis_row, row in zip(axes, rows):
        image_axis, profile_axis, score_axis = axis_row
        image_axis.imshow(
            row["planar_image"], cmap="magma", vmin=0.0,
            vmax=_display_vmax(row["planar_image"]), aspect="auto"
        )
        image_axis.axhline(row["initial_top"], color="cyan", linestyle="--", linewidth=1.3)
        image_axis.axhline(row["initial_bottom"] - 1, color="cyan", linestyle="--", linewidth=1.3)
        image_axis.axhline(row["crop_top"], color="lime", linewidth=1.8)
        image_axis.axhline(row["crop_bottom"] - 1, color="lime", linewidth=1.8)
        image_axis.set_title(f"{row['label']} — crop planaire")
        image_axis.axis("off")

        relative_cm = (
            np.arange(row["crop_height_px"]) * row["planar_spacing_mm"] / 10.0
        )
        matched = row["planar_profile"][row["crop_top"] : row["crop_bottom"]]
        profile_axis.plot(matched, relative_cm, color="#1f77b4", linewidth=2.0, label="Planaire")
        profile_axis.plot(row["qspect_profile"], relative_cm, color="#d62728", linewidth=2.0, label="Q/SPECT")
        profile_axis.invert_yaxis()
        profile_axis.set_xlabel("Profil normalisé")
        profile_axis.set_ylabel("Position dans le crop (cm)")
        profile_axis.set_title(f"Profils alignés — r={row['correlation']:.3f}")
        profile_axis.grid(True, linestyle="--", alpha=0.3)
        profile_axis.legend(frameon=False)

        shifts_cm = (
            (row["candidate_tops"] - row["initial_top"])
            * row["planar_spacing_mm"] / 10.0
        )
        score_axis.plot(shifts_cm, row["scores"], color="#444444", linewidth=2.0)
        uncertainty_left_cm = (
            (row["peak_width_left_top"] - row["initial_top"])
            * row["planar_spacing_mm"] / 10.0
        )
        uncertainty_right_cm = (
            (row["peak_width_right_top"] - row["initial_top"])
            * row["planar_spacing_mm"] / 10.0
        )
        score_axis.axvspan(
            uncertainty_left_cm,
            uncertainty_right_cm,
            color="limegreen",
            alpha=0.18,
            label=f"largeur QC={row['peak_width_cm']:.1f} cm",
        )
        score_axis.axvline(row["proposed_shift_cm"], color="limegreen", linewidth=1.8)
        score_axis.set_xlabel("Translation depuis la position initiale (cm)")
        score_axis.set_ylabel("Corrélation normalisée")
        warning = " — limite atteinte" if row["search_edge_hit"] else ""
        status = "accepté" if row["match_accepted"] else "repli position initiale"
        score_axis.set_title(
            f"Décalage={row['proposed_shift_cm']:+.1f} cm — {status}{warning}"
        )
        score_axis.grid(True, linestyle="--", alpha=0.3)
        score_axis.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Cropping par corrélation des profils longitudinaux planaire–Q/SPECT",
        fontsize=16,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_summary(
    rows: Sequence[Dict[str, Any]], output_path: Path = SUMMARY_FIGURE_PATH
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    days = np.asarray([row["day_offset"] for row in rows])
    shifts = np.asarray([row["shift_cm"] for row in rows])
    correlations = np.asarray([row["correlation"] for row in rows])
    widths = np.asarray([row["peak_width_cm"] for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), facecolor="white", layout="constrained")
    axes[0].axhline(0.0, color="black", linewidth=1.0)
    axes[0].plot(days, shifts, color="#2ca02c", marker="o", linewidth=2.0)
    axes[0].set_ylabel("Translation du crop (cm)")
    axes[1].plot(days, correlations, color="#1f77b4", marker="o", linewidth=2.0)
    axes[1].set_ylabel("Corrélation maximale")
    axes[1].set_ylim(-1.0, 1.05)
    axes[2].axhline(
        MAX_ACCEPTED_PEAK_WIDTH_CM,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Limite d’acceptation",
    )
    axes[2].plot(days, widths, color="#9467bd", marker="o", linewidth=2.0)
    axes[2].set_ylabel("Largeur du maximum (cm)")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.set_xlabel("Temps après le premier Q/SPECT (jours)")
        axis.grid(True, linestyle="--", alpha=0.3)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle("Qualité du matching longitudinal planaire–Q/SPECT", fontsize=15)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_crop_bounds_comparison(
    rows: Sequence[Dict[str, Any]], output_path: Path = BOUNDS_COMPARISON_PATH
) -> Path:
    """Show the fixed Day-0 and profile-matched bounds on every planar image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2, 3, figsize=(10.5, 10.5), facecolor="white", layout="constrained"
    )
    axes = axes.ravel()
    for index, axis in enumerate(axes):
        if index >= len(rows):
            axis.axis("off")
            continue
        row = rows[index]
        axis.imshow(
            row["planar_image"], cmap="magma", vmin=0.0,
            vmax=_display_vmax(row["planar_image"]), aspect="auto"
        )
        axis.axhline(
            row["fixed_crop_top"], color="cyan", linestyle="--", linewidth=1.8,
            label="Crop fixe J0"
        )
        axis.axhline(
            row["fixed_crop_bottom"] - 1, color="cyan", linestyle="--", linewidth=1.8
        )
        axis.axhline(
            row["crop_top"], color="lime", linewidth=1.8,
            label="Profil planaire–Q/SPECT"
        )
        axis.axhline(row["crop_bottom"] - 1, color="lime", linewidth=1.8)
        axis.set_title(
            f"{row['label']} — déplacement {row['shift_cm']:+.1f} cm"
        )
        axis.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="center", frameon=False, fontsize=11)
    axes[-1].axis("off")
    fig.suptitle("Limites du crop fixe et du crop par profil", fontsize=16)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_ctac_activity_comparison(
    rows: Sequence[Dict[str, Any]], output_path: Path = ACTIVITY_COMPARISON_PATH
) -> Path:
    """Compare fixed and profile-matched CTAC activities with Q/SPECT."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    days = np.asarray([row["day_offset"] for row in rows])
    qspect = np.asarray([row["qspect_activity_mbq"] for row in rows])
    fixed = np.asarray([row["fixed_ctac_activity_mbq"] for row in rows])
    profile = np.asarray([row["profile_ctac_activity_mbq"] for row in rows])
    fig, axis = plt.subplots(figsize=(8.3, 5.4), facecolor="white", layout="constrained")
    axis.plot(days, qspect, color="black", marker="D", linestyle="--", linewidth=2.2, label="Q/SPECT")
    axis.plot(days, fixed, color="#1f77b4", marker="s", linewidth=2.0, label="Planaire CTAC — crop fixe J0")
    axis.plot(days, profile, color="#2ca02c", marker="o", linewidth=2.0, label="Planaire CTAC — crop par profil")
    axis.set_xlabel("Temps après le premier Q/SPECT (jours)")
    axis.set_ylabel("Activité estimée (MBq)")
    axis.set_title("Activité selon la méthode de crop")
    axis.grid(True, linestyle="--", alpha=0.3)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_ctac_ratio_comparison(
    rows: Sequence[Dict[str, Any]], output_path: Path = RATIO_COMPARISON_PATH
) -> Path:
    """Compare CTAC/QSPECT ratios for the two crop methods."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    days = np.asarray([row["day_offset"] for row in rows])
    fixed = np.asarray([row["fixed_ctac_over_qspect"] for row in rows])
    profile = np.asarray([row["profile_ctac_over_qspect"] for row in rows])
    fig, axis = plt.subplots(figsize=(8.3, 5.4), facecolor="white", layout="constrained")
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1.3, label="Accord parfait")
    axis.plot(days, fixed, color="#1f77b4", marker="s", linewidth=2.0, label="Crop fixe J0")
    axis.plot(days, profile, color="#2ca02c", marker="o", linewidth=2.0, label="Crop par profil")
    axis.set_xlabel("Temps après le premier Q/SPECT (jours)")
    axis.set_ylabel("Activité planaire CTAC / activité Q/SPECT")
    axis.set_title("Ratio au Q/SPECT selon la méthode de crop")
    axis.grid(True, linestyle="--", alpha=0.3)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, loc="lower left", fontsize=9)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_method_differences(
    rows: Sequence[Dict[str, Any]], output_path: Path = DIFFERENCE_COMPARISON_PATH
) -> Path:
    """Show positional and quantitative changes caused by the new crop."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    days = np.asarray([row["day_offset"] for row in rows])
    shifts = np.asarray([row["shift_cm"] for row in rows])
    activity_percent = np.asarray([row["activity_difference_percent"] for row in rows])
    ratio_difference = np.asarray([row["ratio_difference"] for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), facecolor="white", layout="constrained")
    panels = (
        (shifts, "Déplacement du crop (cm)", "#2ca02c"),
        (activity_percent, "Différence d’activité nouvelle–fixe (%)", "#d62728"),
        (ratio_difference, "Différence du ratio au Q/SPECT", "#9467bd"),
    )
    for axis, (values, ylabel, color) in zip(axes, panels):
        axis.axhline(0.0, color="black", linewidth=1.0)
        axis.plot(days, values, color=color, marker="o", linewidth=2.0)
        axis.set_xlabel("Temps après le premier Q/SPECT (jours)")
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle="--", alpha=0.3)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle("Effet du crop par profil par rapport au crop fixe J0", fontsize=15)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _add_physical_image(
    axis: plt.Axes,
    image: np.ndarray,
    width_cm: float,
    length_cm: float,
    title: str,
    cmap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Display one image with centimetres and a physically meaningful aspect."""
    if vmin is None or vmax is None:
        if cmap == "gray":
            vmin, vmax = -200.0, 300.0
        else:
            vmin, vmax = 0.0, _display_vmax(image)
    axis.imshow(
        image,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=(-width_cm / 2.0, width_cm / 2.0, length_cm, 0.0),
        aspect="equal",
    )
    axis.set_title(title)
    axis.set_xlabel("Largeur (cm)")
    axis.set_ylabel("Longueur tête–pieds (cm)")
    axis.grid(False)


def geometry_comparison_data(row: Dict[str, Any]) -> Dict[str, Any]:
    """Load the paired Day-0 volumes and calculate their physical lengths."""
    planar_dir = planar_processing.default_planar_study_dir()
    qspect_dir = qspect_processing.default_qspect_dir()
    scan = ac.sorted_planar_scans(planar_dir)[0]
    qspect = qspect_processing.load_qspect_study(qspect_dir)[0]
    ct = view_patient_images.load_ct_for_qspect_day(qspect_dir, qspect)

    planar = np.asarray(row["planar_image"], dtype=np.float64)
    crop = planar[int(row["crop_top"]) : int(row["crop_bottom"]), :]
    qspect_projection = ac.qspect_coronal_projections(qspect)["mean_projection"]
    ct_center_index = ct["volume"].shape[1] // 2
    ct_coronal = view_patient_images.orient_qspect_display(
        np.asarray(ct["volume"][:, ct_center_index, :], dtype=np.float64)
    )

    planar_spacing = [float(value) for value in scan["images"][0]["pixel_spacing"]]
    qspect_geometry = volume_geometry(qspect["series_dir"], qspect["volume"].shape)
    ct_geometry = volume_geometry(ct["series_dir"], ct["volume"].shape)
    return {
        "label": str(qspect["label"]),
        "planar": planar,
        "crop": crop,
        "qspect": qspect_projection,
        "ct": ct_coronal,
        "planar_matrix_length_cm": planar.shape[0] * planar_spacing[0] / 10.0,
        "planar_scan_length_cm": float(scan["images"][0]["scan_length"]) / 10.0,
        "planar_width_cm": planar.shape[1] * planar_spacing[1] / 10.0,
        "fixed_crop_length_cm": (
            int(row["fixed_crop_bottom"]) - int(row["fixed_crop_top"])
        )
        * planar_spacing[0]
        / 10.0,
        "profile_crop_length_cm": crop.shape[0] * planar_spacing[0] / 10.0,
        "qspect_length_cm": qspect_geometry["coverage_mm"] / 10.0,
        "qspect_center_range_cm": qspect_geometry["center_range_mm"] / 10.0,
        "qspect_width_cm": qspect_geometry["width_mm"] / 10.0,
        "ct_length_cm": ct_geometry["coverage_mm"] / 10.0,
        "ct_center_range_cm": ct_geometry["center_range_mm"] / 10.0,
        "ct_width_cm": ct_geometry["width_mm"] / 10.0,
        "qspect_geometry": qspect_geometry,
        "ct_geometry": ct_geometry,
        "ct_series_description": str(ct["series_description"]),
        "crop_top": int(row["crop_top"]),
        "crop_bottom": int(row["crop_bottom"]),
    }


def geometry_lengths_all_days(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return reproducible longitudinal lengths for every paired time point."""
    planar_dir = planar_processing.default_planar_study_dir()
    qspect_dir = qspect_processing.default_qspect_dir()
    scans = ac.sorted_planar_scans(planar_dir)
    qspect_series = qspect_processing.load_qspect_study(qspect_dir)
    output = []
    for row, scan, qspect in zip(rows, scans, qspect_series):
        ct = view_patient_images.load_ct_for_qspect_day(qspect_dir, qspect)
        qspect_geometry = volume_geometry(qspect["series_dir"], qspect["volume"].shape)
        ct_geometry = volume_geometry(ct["series_dir"], ct["volume"].shape)
        planar_spacing_mm = float(scan["images"][0]["pixel_spacing"][0])
        output.append(
            {
                "day_offset": float(row["day_offset"]),
                "label": str(qspect["label"]),
                "planar_matrix_length_cm": float(row["planar_image"].shape[0])
                * planar_spacing_mm
                / 10.0,
                "planar_scan_length_cm": float(scan["images"][0]["scan_length"])
                / 10.0,
                "fixed_crop_length_cm": (
                    int(row["fixed_crop_bottom"]) - int(row["fixed_crop_top"])
                )
                * planar_spacing_mm
                / 10.0,
                "profile_crop_length_cm": (
                    int(row["crop_bottom"]) - int(row["crop_top"])
                )
                * planar_spacing_mm
                / 10.0,
                "qspect_slice_count": int(qspect_geometry["slice_count"]),
                "qspect_length_cm": qspect_geometry["coverage_mm"] / 10.0,
                "ct_slice_count": int(ct_geometry["slice_count"]),
                "ct_length_cm": ct_geometry["coverage_mm"] / 10.0,
                "ct_series_description": str(ct["series_description"]),
            }
        )
    return output


def plot_full_geometry_comparison(
    geometry: Dict[str, Any], output_path: Path = FULL_GEOMETRY_PATH
) -> Path:
    """Compare the complete, non-cropped planar matrix with Q/SPECT and CT."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 10.0), facecolor="white", layout="constrained")
    _add_physical_image(
        axes[0], geometry["planar"], geometry["planar_width_cm"],
        geometry["planar_matrix_length_cm"],
        f"Planaire complet\nmatrice = {geometry['planar_matrix_length_cm']:.2f} cm\nScanLength = {geometry['planar_scan_length_cm']:.1f} cm",
        "magma",
    )
    _add_physical_image(
        axes[1], geometry["qspect"], geometry["qspect_width_cm"],
        geometry["qspect_length_cm"],
        f"Q/SPECT complet\nlongueur = {geometry['qspect_length_cm']:.2f} cm",
        "magma",
    )
    _add_physical_image(
        axes[2], geometry["ct"], geometry["ct_width_cm"],
        geometry["ct_length_cm"],
        f"CT complet\nlongueur = {geometry['ct_length_cm']:.2f} cm",
        "gray",
    )
    fig.suptitle(
        f"{geometry['label']} — images complètes à leur échelle physique",
        fontsize=16,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_cropped_geometry_comparison(
    geometry: Dict[str, Any], output_path: Path = CROPPED_GEOMETRY_PATH
) -> Path:
    """Replace the complete planar matrix by the profile-matched crop."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 6.2), facecolor="white", layout="constrained")
    _add_physical_image(
        axes[0], geometry["crop"], geometry["planar_width_cm"],
        geometry["profile_crop_length_cm"],
        f"Planaire croppé ({geometry['crop_top']}:{geometry['crop_bottom']})\nlongueur = {geometry['profile_crop_length_cm']:.2f} cm",
        "magma",
    )
    _add_physical_image(
        axes[1], geometry["qspect"], geometry["qspect_width_cm"],
        geometry["qspect_length_cm"],
        f"Q/SPECT complet\nlongueur = {geometry['qspect_length_cm']:.2f} cm",
        "magma",
    )
    _add_physical_image(
        axes[2], geometry["ct"], geometry["ct_width_cm"],
        geometry["ct_length_cm"],
        f"CT complet\nlongueur = {geometry['ct_length_cm']:.2f} cm",
        "gray",
    )
    fig.suptitle(
        f"{geometry['label']} — crop planaire comparé à la couverture Q/SPECT–CT",
        fontsize=16,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def write_geometry_outputs(
    geometry: Dict[str, Any], all_days: Sequence[Dict[str, Any]]
) -> Tuple[Path, Path]:
    """Save the exact displayed lengths and explain their definitions."""
    pd.DataFrame(all_days).to_csv(GEOMETRY_VALUES_PATH, index=False)
    lines = [
        "Planar, Q/SPECT and CT longitudinal coverage",
        "=============================================",
        "",
        f"Time point: {geometry['label']}",
        f"Planar matrix: {geometry['planar_matrix_length_cm']:.4f} cm",
        f"Planar DICOM ScanLength: {geometry['planar_scan_length_cm']:.4f} cm",
        f"Fixed Day-0 crop: {geometry['fixed_crop_length_cm']:.4f} cm",
        f"Profile-matched crop: {geometry['profile_crop_length_cm']:.4f} cm",
        f"Q/SPECT edge-to-edge coverage: {geometry['qspect_length_cm']:.4f} cm",
        f"Q/SPECT first-to-last slice-center range: {geometry['qspect_center_range_cm']:.4f} cm",
        f"CT edge-to-edge coverage: {geometry['ct_length_cm']:.4f} cm",
        f"CT first-to-last slice-center range: {geometry['ct_center_range_cm']:.4f} cm",
        f"CT series: {geometry['ct_series_description']}",
        "",
        "Edge-to-edge coverage = first-to-last slice-center range + SliceThickness.",
        "The planar matrix length and DICOM ScanLength are reported separately",
        "because they are not equal in this acquisition.",
        "",
        " day | label | planar matrix | ScanLength | fixed crop | profile crop | Q/SPECT | CT",
        "---- | ----- | ------------- | ---------- | ---------- | ------------ | ------- | --",
    ]
    for row in all_days:
        lines.append(
            f"{row['day_offset']:4.2f} | {row['label']:5s} | "
            f"{row['planar_matrix_length_cm']:13.2f} | {row['planar_scan_length_cm']:10.2f} | "
            f"{row['fixed_crop_length_cm']:10.2f} | {row['profile_crop_length_cm']:12.2f} | "
            f"{row['qspect_length_cm']:7.2f} | {row['ct_length_cm']:6.2f}"
        )
    GEOMETRY_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return GEOMETRY_VALUES_PATH, GEOMETRY_REPORT_PATH


def write_outputs(rows: Sequence[Dict[str, Any]]) -> Tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scalar_keys = (
        "day_offset", "label", "scan_name", "seed_day0_top", "seed_day0_bottom",
        "initial_top", "initial_bottom", "crop_top", "crop_bottom", "crop_height_px",
        "crop_height_cm", "shift_px", "shift_cm", "correlation",
        "second_peak_correlation", "peak_margin", "search_edge_hit",
        "proposed_crop_top", "proposed_crop_bottom", "proposed_shift_px",
        "proposed_shift_cm", "match_accepted", "rejection_reasons",
        "peak_width_px", "peak_width_cm", "peak_half_width_cm",
        "peak_width_correlation_loss", "minimum_accepted_correlation",
        "minimum_accepted_peak_margin", "maximum_accepted_peak_width_cm",
        "initial_position_source",
        "planar_spacing_mm", "qspect_spacing_mm", "search_limit_cm",
        "smoothing_sigma_cm",
        "fixed_crop_top", "fixed_crop_bottom", "qspect_activity_mbq",
        "fixed_ctac_activity_mbq", "profile_ctac_activity_mbq",
        "fixed_ctac_over_qspect", "profile_ctac_over_qspect",
        "activity_difference_mbq", "activity_difference_percent",
        "ratio_difference", "fixed_effective_ct_factor",
        "profile_effective_ct_factor", "fixed_outside_counts_percent",
        "profile_outside_counts_percent",
    )
    frame = pd.DataFrame(
        [{key: row[key] for key in scalar_keys} for row in rows]
    )
    frame.to_csv(VALUES_PATH, index=False)
    lines = [
        "Planar/QSPECT longitudinal-profile crop matching",
        "=================================================",
        "",
        "Method:",
        "  TEW AP/PA geometric mean and Q/SPECT coronal mean projection;",
        "  horizontal sums; resampling to planar row spacing; log compression;",
        "  Gaussian smoothing; bounded normalized cross-correlation.",
        "  The existing Day-0 crop is only a provisional initial position until",
        "  a marker-based machine calibration supplies the true fixed offset.",
        "  Acceptance requires correlation >= 0.90, peak margin >= 0.01,",
        "  no search-boundary maximum, and a QC peak width <= 5 cm.",
        "  An invalid match falls back to the initial Q/SPECT-length crop.",
        "  Peak width spans contiguous translations within 0.01 correlation of",
        "  the optimum; it is a deterministic QC width, not a confidence interval.",
        "",
        "A boundary maximum or a small peak margin indicates an unreliable match.",
        "",
        " day | label | initial crop | applied crop | shift cm | corr | margin | width cm | accepted",
        "---- | ----- | ------------ | ------------ | -------- | ---- | ------ | -------- | --------",
    ]
    for row in rows:
        lines.append(
            f"{row['day_offset']:4.2f} | {row['label']:5s} | "
            f"{row['initial_top']:4d}:{row['initial_bottom']:<4d} | "
            f"{row['crop_top']:4d}:{row['crop_bottom']:<4d} | "
            f"{row['shift_cm']:+8.2f} | {row['correlation']:.3f} | "
            f"{row['peak_margin']:.3f} | {row['peak_width_cm']:.2f} | "
            f"{row['match_accepted']}"
        )
    lines.extend(
        [
            "",
            "Quantitative comparison after the complete current CTAC pipeline:",
            "",
            " day | fixed MBq | profile MBq | Q/SPECT MBq | fixed/QSP | profile/QSP | difference %",
            "---- | --------- | ----------- | ----------- | --------- | ----------- | ------------",
        ]
    )
    for row in rows:
        lines.append(
            f"{row['day_offset']:4.2f} | {row['fixed_ctac_activity_mbq']:9.1f} | "
            f"{row['profile_ctac_activity_mbq']:11.1f} | {row['qspect_activity_mbq']:11.1f} | "
            f"{row['fixed_ctac_over_qspect']:9.3f} | {row['profile_ctac_over_qspect']:11.3f} | "
            f"{row['activity_difference_percent']:+12.2f}"
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return VALUES_PATH, REPORT_PATH


def parse_args() -> Any:
    parser = ArgumentParser(
        description="Match planar and Q/SPECT longitudinal profiles for cropping."
    )
    parser.add_argument("--initial-top", type=int, default=None)
    parser.add_argument("--initial-bottom", type=int, default=None)
    parser.add_argument("--max-shift-cm", type=float, default=MAX_SHIFT_CM)
    parser.add_argument("--smoothing-sigma-cm", type=float, default=SMOOTHING_SIGMA_CM)
    args = parser.parse_args()
    if (args.initial_top is None) != (args.initial_bottom is None):
        parser.error("--initial-top and --initial-bottom must be provided together")
    return args


def main() -> None:
    args = parse_args()
    initial_bounds = (
        None
        if args.initial_top is None
        else (int(args.initial_top), int(args.initial_bottom))
    )
    rows = calculate_matches(
        max_shift_cm=float(args.max_shift_cm),
        smoothing_sigma_cm=float(args.smoothing_sigma_cm),
        initial_crop_bounds=initial_bounds,
    )
    add_ctac_comparison(rows)
    geometry = geometry_comparison_data(rows[0])
    all_geometry_lengths = geometry_lengths_all_days(rows)
    detail_path = plot_detailed_matches(rows)
    summary_path = plot_summary(rows)
    bounds_path = plot_crop_bounds_comparison(rows)
    activity_path = plot_ctac_activity_comparison(rows)
    ratio_path = plot_ctac_ratio_comparison(rows)
    difference_path = plot_method_differences(rows)
    full_geometry_path = plot_full_geometry_comparison(geometry)
    cropped_geometry_path = plot_cropped_geometry_comparison(geometry)
    geometry_values_path, geometry_report_path = write_geometry_outputs(
        geometry, all_geometry_lengths
    )
    values_path, report_path = write_outputs(rows)
    for row in rows:
        print(
            f"{row['label']}: crop={row['crop_top']}:{row['crop_bottom']} | "
            f"shift={row['shift_cm']:+.2f} cm | r={row['correlation']:.3f} | "
            f"margin={row['peak_margin']:.3f} | width={row['peak_width_cm']:.2f} cm | "
            f"accepted={row['match_accepted']}"
        )
    print(f"Saved detailed figure: {detail_path}")
    print(f"Saved summary figure: {summary_path}")
    print(f"Saved crop comparison: {bounds_path}")
    print(f"Saved activity comparison: {activity_path}")
    print(f"Saved ratio comparison: {ratio_path}")
    print(f"Saved difference comparison: {difference_path}")
    print(f"Saved full-geometry comparison: {full_geometry_path}")
    print(f"Saved cropped-geometry comparison: {cropped_geometry_path}")
    print(f"Saved geometry values: {geometry_values_path}")
    print(f"Saved geometry report: {geometry_report_path}")
    print(f"Saved values: {values_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
