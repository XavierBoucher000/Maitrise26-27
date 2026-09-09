"""One-patient emission-window model for planar Lu-177 attenuation.

This module is a proof of concept, not a clinical or population model.  It
uses a CT-derived attenuation map as the development reference and uses
Q/SPECT only for an optional, final descriptive activity comparison.

Target
------
For a CT volume with constant voxel spacing ``delta_z`` along the assumed
anterior/posterior integration axis::

    M_CT(x, y) = sum_z(mu_208(x, y, z)) * delta_z
    F_CT(x, y) = exp(M_CT(x, y) / 2)

For each emission patch, the scalar correction target is the CT-derived
effective factor for the counts observed in that patch::

    F_eff = sum(C_TEW * F_CT) / sum(C_TEW)
    M_eff = 2 * log(F_eff)

The ridge model predicts ``M_eff`` from local planar emission-window features.
Thus the CT map is independent of emission, while the patch scalar is
count-weighted because the intended output corrects those counts.  It never
uses Q/SPECT activity as a regression target.

Inputs
------
Each non-overlapping planar patch contributes these emission-only features:

* log lower-scatter / photopeak count-density ratio;
* log upper-scatter / photopeak count-density ratio;
* log low-energy / photopeak count-density ratio;
* absolute AP/PA photopeak log-asymmetry;
* photopeak density relative to the median signal in that acquisition.

Counts are normalized by energy-window width and acquisition duration before
forming ratios.  Patches are weighted by their photopeak statistics, with the
weights normalized separately within every acquisition so Day0 cannot dominate
only because it has more patches or counts.

Validation
----------
Leave-one-day-out validation is grouped by acquisition day.  All patches from
the held-out day remain outside the fit.  Neighboring patches are not
independent patients, so results only demonstrate within-patient feasibility.

Known development limitations
-----------------------------
The current CT reference inherits an approximate CT-to-planar resize. During
development, the default crop uses the Q/SPECT physical coverage and bounded
longitudinal-profile matching for each timepoint. Q/SPECT defines geometry only;
its activity is not a regression target. An emission-only crop localizer remains
necessary before the method can be applied to a new patient without Q/SPECT.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom

import attenuation_correction as ac
import ct_attenuation_correction as ctac
import dicom_loader
import planar_processing
import planar_qspect_profile_crop
import qspect_processing
from plots import view_patient_images


FIG_ROOT = Path(__file__).resolve().parent / "fig" / "attenuation_window_model"
REPORT_PATH = FIG_ROOT / "one_patient_window_ridge_report.txt"
VALIDATION_FIGURE_PATH = FIG_ROOT / "one_patient_window_ridge_validation.png"
MAP_FIGURE_PATH = FIG_ROOT / "one_patient_window_ridge_last_day_map.png"
MODEL_PATH = FIG_ROOT / "one_patient_window_ridge_model.npz"

FEATURE_NAMES = (
    "log lower/photopeak",
    "log upper/photopeak",
    "log low-energy/photopeak",
    "abs AP/PA asymmetry",
    "log relative photopeak",
)

REQUIRED_WINDOWS = (
    "Lower Scatter",
    "Photopeak",
    "Upper Scatter",
    "Low Energy Scatter",
)


def _ct_training_priority(series_description: str) -> int:
    """Prefer the same raw ACCT representation on every acquisition day."""
    description = " ".join(series_description.upper().split())
    if description == "ACCT":
        return 0
    if description == "CT":
        return 1
    if "ACCT" in description and "TRANSFORMED" in description:
        return 2
    if "TOPOGRAM" in description:
        return 99
    return 10


def _positive_array(image: np.ndarray) -> np.ndarray:
    """Return a finite, non-negative float image."""
    image = np.asarray(image, dtype=np.float64)
    return np.clip(np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(weights))
    if total <= 0.0:
        return float(np.mean(values))
    return float(np.sum(np.asarray(values, dtype=np.float64) * weights) / total)


def _weighted_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
) -> Dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / max(float(np.sum(weights)), 1e-12)
    residual = prediction - target
    mean_target = float(np.sum(weights * target))
    sse = float(np.sum(weights * residual**2))
    sst = float(np.sum(weights * (target - mean_target) ** 2))
    minimum_log_factor = np.log(ctac.CT_FACTOR_CLIP[0])
    maximum_log_factor = np.log(ctac.CT_FACTOR_CLIP[1])
    true_factor = np.exp(np.clip(0.5 * target, minimum_log_factor, maximum_log_factor))
    predicted_factor = np.exp(
        np.clip(0.5 * prediction, minimum_log_factor, maximum_log_factor)
    )
    factor_relative_error = np.abs(predicted_factor - true_factor) / np.maximum(true_factor, 1e-12)
    return {
        "mae_optical_depth": float(np.sum(weights * np.abs(residual))),
        "rmse_optical_depth": float(np.sqrt(sse)),
        "r2_optical_depth": float(1.0 - sse / sst) if sst > 0.0 else np.nan,
        "mean_absolute_factor_error_percent": float(100.0 * np.sum(weights * factor_relative_error)),
    }


def window_density_maps(scan: Dict[str, Any]) -> Dict[str, Any]:
    """Build AP, PA and geometric-mean count-density maps for every window.

    Density has units of counts / (keV s).  Ratios therefore account for the
    unequal energy-window widths.  The duration is common to all frames in one
    acquisition, but retaining it makes the feature definition explicit.
    """
    images = scan["images"]
    groups = ac.group_by_energy_and_view(images)
    widths = planar_processing.window_widths(images)
    timing = planar_processing.planar_timing_from_dicom(images)
    duration = float(timing.actual_frame_duration_s)
    if duration <= 0.0:
        raise ValueError("A positive planar frame duration is required")

    missing = [
        energy
        for energy in REQUIRED_WINDOWS
        if energy not in groups
        or "AP" not in groups[energy]
        or "PA" not in groups[energy]
        or energy not in widths
    ]
    if missing:
        raise ValueError(f"Missing AP/PA image or width for: {', '.join(missing)}")

    maps: Dict[str, Dict[str, np.ndarray]] = {}
    for energy in REQUIRED_WINDOWS:
        width = float(widths[energy])
        if width <= 0.0:
            raise ValueError(f"Invalid energy-window width for {energy}: {width}")
        ap = _positive_array(groups[energy]["AP"]["image"])
        pa = _positive_array(groups[energy]["PA"]["image"])
        maps[energy] = {
            "ap": ap / (width * duration),
            "pa": pa / (width * duration),
            "gm": ac.geometric_mean(ap, pa) / (width * duration),
        }
    return {"maps": maps, "widths": widths, "duration_seconds": duration}


def _crop_map(image: np.ndarray, crop_top: int, crop_bottom: int) -> np.ndarray:
    crop = np.asarray(image, dtype=np.float64)[crop_top:crop_bottom, :]
    if crop.size == 0:
        raise ValueError(f"Empty crop y={crop_top}:{crop_bottom}")
    return crop


def _patch_feature_vector(
    density_crops: Dict[str, Dict[str, np.ndarray]],
    patch_slice: Tuple[slice, slice],
    valid_patch: np.ndarray,
    photopeak_reference: float,
) -> np.ndarray:
    """Calculate one robust feature vector from a planar patch."""
    pixel_count = max(int(np.count_nonzero(valid_patch)), 1)

    def patch_mean(energy: str, view: str = "gm") -> float:
        values = density_crops[energy][view][patch_slice]
        return float(np.sum(values[valid_patch]) / pixel_count)

    photo = patch_mean("Photopeak")
    lower = patch_mean("Lower Scatter")
    upper = patch_mean("Upper Scatter")
    low_energy = patch_mean("Low Energy Scatter")
    photo_ap = patch_mean("Photopeak", "ap")
    photo_pa = patch_mean("Photopeak", "pa")

    # A small adaptive pseudo-count avoids unstable logarithms in late images.
    epsilon = max(1e-12, 1e-6 * photopeak_reference)
    return np.asarray(
        [
            np.log((lower + epsilon) / (photo + epsilon)),
            np.log((upper + epsilon) / (photo + epsilon)),
            np.log((low_energy + epsilon) / (photo + epsilon)),
            abs(0.5 * np.log((photo_ap + epsilon) / (photo_pa + epsilon))),
            np.log((photo + epsilon) / (photopeak_reference + epsilon)),
        ],
        dtype=np.float64,
    )


def patch_samples_for_day(
    scan: Dict[str, Any],
    ct_row: Dict[str, Any],
    day_index: int,
    patch_shape: Tuple[int, int] = (16, 8),
    mask_threshold_fraction: float = 0.005,
    minimum_valid_fraction: float = 0.25,
) -> List[Dict[str, Any]]:
    """Build patch samples for one acquisition and its CT-derived reference map."""
    patch_height, patch_width = patch_shape
    if patch_height <= 0 or patch_width <= 0:
        raise ValueError("Patch dimensions must be positive")

    crop_top = int(ct_row["crop_top"])
    crop_bottom = int(ct_row["crop_bottom"])
    density = window_density_maps(scan)
    density_crops: Dict[str, Dict[str, np.ndarray]] = {
        energy: {
            view: _crop_map(image, crop_top, crop_bottom)
            for view, image in view_maps.items()
        }
        for energy, view_maps in density["maps"].items()
    }

    planar_crop = _positive_array(ct_row["planar_crop"])
    optical_depth = _positive_array(ct_row["ct_mu_integral_resized"])
    if planar_crop.shape != optical_depth.shape:
        raise ValueError(
            f"Planar/CT-reference shape mismatch: {planar_crop.shape} versus {optical_depth.shape}"
        )
    for energy in REQUIRED_WINDOWS:
        if density_crops[energy]["gm"].shape != planar_crop.shape:
            raise ValueError(
                f"{energy} crop shape {density_crops[energy]['gm'].shape} "
                f"does not match CT reference {planar_crop.shape}"
            )

    maximum = float(np.max(planar_crop))
    if maximum <= 0.0:
        raise ValueError(f"Empty planar crop for {ct_row['day_label']}")
    body_mask = planar_crop >= mask_threshold_fraction * maximum

    positive_photo = density_crops["Photopeak"]["gm"][body_mask]
    positive_photo = positive_photo[positive_photo > 0.0]
    if positive_photo.size == 0:
        raise ValueError(f"No positive photopeak signal for {ct_row['day_label']}")
    photopeak_reference = float(np.median(positive_photo))

    height, width = planar_crop.shape
    samples: List[Dict[str, Any]] = []
    for top in range(0, height, patch_height):
        bottom = min(top + patch_height, height)
        for left in range(0, width, patch_width):
            right = min(left + patch_width, width)
            patch_slice = (slice(top, bottom), slice(left, right))
            valid_patch = body_mask[patch_slice]
            if float(np.mean(valid_patch)) < minimum_valid_fraction:
                continue

            planar_patch = planar_crop[patch_slice]
            count_weights = planar_patch * valid_patch
            patch_counts = float(np.sum(count_weights))
            if patch_counts <= 0.0:
                continue
            target_patch = optical_depth[patch_slice]
            target_factor_patch = np.exp(
                np.clip(
                    0.5 * target_patch,
                    np.log(ctac.CT_FACTOR_CLIP[0]),
                    np.log(ctac.CT_FACTOR_CLIP[1]),
                )
            )
            effective_patch_factor = _weighted_mean(
                target_factor_patch,
                count_weights,
            )
            target = 2.0 * np.log(max(effective_patch_factor, 1.0))
            features = _patch_feature_vector(
                density_crops,
                patch_slice,
                valid_patch,
                photopeak_reference,
            )
            if not np.all(np.isfinite(features)) or not np.isfinite(target):
                continue
            samples.append(
                {
                    "features": features,
                    "target": target,
                    "patch_counts": patch_counts,
                    "day_index": day_index,
                    "day_offset": float(ct_row["day_offset"]),
                    "label": str(ct_row["day_label"]),
                    "bounds": (top, bottom, left, right),
                    "valid_mask": valid_patch.copy(),
                }
            )

    if not samples:
        raise ValueError(f"No valid patches for {ct_row['day_label']}")

    # Preserve higher-confidence patches while giving every day equal total
    # weight, regardless of the number of usable patches or total activity.
    raw_weights = np.sqrt(np.asarray([sample["patch_counts"] for sample in samples]))
    raw_weights /= max(float(np.sum(raw_weights)), 1e-12)
    for sample, sample_weight in zip(samples, raw_weights):
        sample["sample_weight"] = float(sample_weight)

    covered_mask = np.zeros_like(body_mask, dtype=bool)
    for sample in samples:
        top, bottom, left, right = sample["bounds"]
        covered_mask[top:bottom, left:right] |= sample["valid_mask"]
    total_counts = float(np.sum(planar_crop))
    ct_row["emission_mask_pixel_fraction"] = float(np.mean(body_mask))
    ct_row["emission_mask_count_fraction"] = (
        float(np.sum(planar_crop[body_mask])) / total_counts
        if total_counts > 0.0
        else np.nan
    )
    ct_row["training_support_pixel_fraction"] = float(np.mean(covered_mask))
    ct_row["training_support_count_fraction"] = (
        float(np.sum(planar_crop[covered_mask])) / total_counts
        if total_counts > 0.0
        else np.nan
    )
    return samples


def load_ct_for_planar_scan(ct_root: Path, scan: Dict[str, Any]) -> Dict[str, Any]:
    """Select the best same-day ACCT/CT series for one planar acquisition.

    Selection uses only DICOM dates, times and CT series descriptions.  It does
    not load or inspect Q/SPECT activity.
    """
    ct_dirs = view_patient_images.find_ct_series_dirs(ct_root)
    if not ct_dirs:
        raise ValueError(f"No CT series found in {ct_root}")

    planar_datetime = dicom_loader.scan_datetime(scan)
    planar_date = planar_datetime.date() if planar_datetime is not None else None
    candidates = []
    for series_dir in ct_dirs:
        series_datetime = view_patient_images.ct_series_datetime(series_dir)
        first_file = next(series_dir.glob("*.dcm"), None)
        if first_file is None:
            continue
        ct_header = pydicom.dcmread(
            first_file,
            stop_before_pixels=True,
            force=True,
        )
        description = str(getattr(ct_header, "SeriesDescription", series_dir.name))
        if "TOPOGRAM" in description.upper():
            continue
        same_day_penalty = (
            0
            if planar_date is not None
            and series_datetime is not None
            and series_datetime.date() == planar_date
            else 1
        )
        time_delta = (
            abs((series_datetime - planar_datetime).total_seconds())
            if series_datetime is not None and planar_datetime is not None
            else float("inf")
        )
        candidates.append(
            (
                same_day_penalty,
                _ct_training_priority(description),
                time_delta,
                series_dir,
            )
        )
    if not candidates:
        raise ValueError(f"No usable CT/ACCT series found in {ct_root}")

    same_day_penalty, _priority, _delta, selected_dir = min(
        candidates,
        key=lambda item: item[:3],
    )
    if same_day_penalty != 0:
        raise ValueError(
            f"No CT/ACCT acquired on the planar date {planar_date} in {ct_root}"
        )
    ct = view_patient_images.load_ct_series(selected_dir)
    ct["shape"] = ct["volume"].shape
    return ct


def build_ct_reference_rows(
    scans: Sequence[Dict[str, Any]],
    ct_root: Path,
    crop_bounds: Optional[Tuple[int, int]] = None,
    crop_strategy: str = "profile_qspect",
    conversion_method: str = ctac.DEFAULT_CT_CONVERSION_METHOD,
) -> List[Dict[str, Any]]:
    """Build raw-ACCT reference rows using profile crops by default."""
    if crop_strategy not in {"profile_qspect", "fixed"}:
        raise ValueError("crop_strategy must be 'profile_qspect' or 'fixed'")
    fixed_bounds: Optional[Tuple[int, int]] = None
    profile_matches: Optional[List[Dict[str, Any]]] = None
    if crop_strategy == "fixed":
        if crop_bounds is None:
            raise ValueError("crop_bounds are required with crop_strategy='fixed'")
        fixed_bounds = (int(crop_bounds[0]), int(crop_bounds[1]))
        if fixed_bounds[0] < 0 or fixed_bounds[1] <= fixed_bounds[0]:
            raise ValueError(f"Invalid fixed crop bounds: {crop_bounds}")
    else:
        qspect_series = qspect_processing.load_qspect_study(ct_root)
        profile_matches = planar_qspect_profile_crop.resolve_profile_crop_matches(
            scans,
            qspect_series,
            align_pa_to_ap=planar_processing.ALIGN_PA_TO_AP,
        )

    rows: List[Dict[str, Any]] = []
    first_datetime = next(
        (dicom_loader.scan_datetime(scan) for scan in scans if dicom_loader.scan_datetime(scan) is not None),
        None,
    )
    for day_index, scan in enumerate(scans):
        match = None if profile_matches is None else profile_matches[day_index]
        if match is None:
            assert fixed_bounds is not None
            crop_top, crop_bottom = fixed_bounds
        else:
            crop_top = int(match["crop_top"])
            crop_bottom = int(match["crop_bottom"])
        planar_result = ac.tew_geometric_mean_for_scan(scan)
        full_planar = _positive_array(planar_result["image"])
        if crop_bottom > full_planar.shape[0]:
            raise ValueError(
                f"Crop {(crop_top, crop_bottom)} exceeds planar height {full_planar.shape[0]} "
                f"for {scan['scan_name']}"
        )
        planar_crop = full_planar[crop_top:crop_bottom, :]
        ct = load_ct_for_planar_scan(ct_root, scan)
        ct_map = ctac.ct_attenuation_factor_map_for_method(ct, conversion_method)
        mu_integral_resized = ctac.resize_map_to_image(
            ct_map["mu_integral"],
            planar_crop.shape,
        )
        mu_integral_resized = _positive_array(mu_integral_resized)
        log_factor = np.clip(
            0.5 * mu_integral_resized,
            np.log(ctac.CT_FACTOR_CLIP[0]),
            np.log(ctac.CT_FACTOR_CLIP[1]),
        )
        factor_resized = np.exp(log_factor)
        corrected_crop = planar_crop * factor_resized
        planar_crop_counts = float(np.sum(planar_crop))
        ctac_crop_counts = float(np.sum(corrected_crop))
        effective_ct_factor = (
            ctac_crop_counts / planar_crop_counts
            if planar_crop_counts > 0.0
            else np.nan
        )
        timing = planar_result["timing"]
        scan_datetime = dicom_loader.scan_datetime(scan)
        day_offset = (
            (scan_datetime - first_datetime).total_seconds() / 86400.0
            if scan_datetime is not None and first_datetime is not None
            else float(day_index)
        )
        before_activity = planar_processing.counts_to_activity_mbq(
            planar_crop_counts,
            timing.local_dwell_time_s,
            planar_processing.CAMERA_SENSITIVITY_CPS_PER_MBQ,
        )
        after_activity = planar_processing.counts_to_activity_mbq(
            ctac_crop_counts,
            timing.local_dwell_time_s,
            planar_processing.CAMERA_SENSITIVITY_CPS_PER_MBQ,
        )
        rows.append(
            {
                "label": scan["scan_name"],
                "day_label": f"Day{int(round(day_offset))}",
                "day_offset": float(day_offset),
                "datetime": scan_datetime,
                "crop_top": crop_top,
                "crop_bottom": crop_bottom,
                "crop_strategy": crop_strategy,
                "planar_crop": planar_crop,
                "planar_crop_counts": planar_crop_counts,
                "ctac_crop_counts": ctac_crop_counts,
                "planar_crop_local_activity_mbq": float(before_activity),
                "ctac_local_activity_mbq": float(after_activity),
                "effective_ct_factor": float(effective_ct_factor),
                "ct_factor_resized": factor_resized,
                "ctac_crop": corrected_crop,
                "ct_mu_integral": ct_map["mu_integral"],
                "ct_mu_integral_resized": mu_integral_resized,
                "delta_z_cm": float(ct_map["ap_spacing_cm"]),
                "local_dwell_time_s": float(timing.local_dwell_time_s),
                "ct_series_description": str(ct["series_description"]),
                "ct_shape": tuple(ct["shape"]),
                "ct_conversion_method": conversion_method,
                "ct_conversion_protocol_limitation": ct_map.get(
                    "protocol_limitation"
                ),
                "profile_match_correlation": (
                    np.nan if match is None else float(match["correlation"])
                ),
                "profile_match_accepted": (
                    None if match is None else bool(match["match_accepted"])
                ),
            }
        )
    return rows


def attach_optional_qspect_reference(
    rows: List[Dict[str, Any]],
    qspect_root: Path,
) -> None:
    """Attach Q/SPECT activities after dataset construction for reporting only."""
    available = [
        candidate
        for candidate in qspect_processing.qspect_decay_data(qspect_root)
        if candidate.get("datetime") is not None
        and candidate.get("activity_mbq") is not None
    ]
    for row in rows:
        planar_datetime = row.get("datetime")
        if planar_datetime is None:
            row["qspect_activity_mbq"] = np.nan
            continue
        same_date = [
            (index, candidate)
            for index, candidate in enumerate(available)
            if candidate["datetime"].date() == planar_datetime.date()
        ]
        if not same_date:
            row["qspect_activity_mbq"] = np.nan
            continue
        selected_index, nearest = min(
            same_date,
            key=lambda item: abs(
                (item[1]["datetime"] - planar_datetime).total_seconds()
            ),
        )
        time_delta_seconds = abs(
            (nearest["datetime"] - planar_datetime).total_seconds()
        )
        if time_delta_seconds > 12.0 * 3600.0:
            row["qspect_activity_mbq"] = np.nan
            continue
        row["qspect_activity_mbq"] = float(nearest["activity_mbq"])
        available.pop(selected_index)


def build_dataset(
    planar_dir: Path = planar_processing.default_planar_study_dir(),
    ct_root: Path = qspect_processing.default_qspect_dir(),
    patch_shape: Tuple[int, int] = (16, 8),
    mask_threshold_fraction: float = 0.005,
    minimum_valid_fraction: float = 0.25,
    crop_bounds: Optional[Tuple[int, int]] = None,
    crop_strategy: str = "profile_qspect",
    include_qspect_reference: bool = False,
    conversion_method: str = ctac.DEFAULT_CT_CONVERSION_METHOD,
) -> Dict[str, Any]:
    """Load planar/CT acquisitions and create a patch-level dataset."""
    scans = ac.sorted_planar_scans(planar_dir)
    ct_rows = build_ct_reference_rows(
        scans,
        ct_root,
        crop_bounds=crop_bounds,
        crop_strategy=crop_strategy,
        conversion_method=conversion_method,
    )
    if include_qspect_reference:
        attach_optional_qspect_reference(ct_rows, ct_root)
    count = min(len(scans), len(ct_rows))
    if count < 2:
        raise ValueError("At least two paired acquisition days are required")

    samples: List[Dict[str, Any]] = []
    for day_index in range(count):
        samples.extend(
            patch_samples_for_day(
                scans[day_index],
                ct_rows[day_index],
                day_index,
                patch_shape=patch_shape,
                mask_threshold_fraction=mask_threshold_fraction,
                minimum_valid_fraction=minimum_valid_fraction,
            )
        )

    return {
        "samples": samples,
        "ct_rows": ct_rows[:count],
        "scans": scans[:count],
        "patch_shape": patch_shape,
        "mask_threshold_fraction": mask_threshold_fraction,
        "minimum_valid_fraction": minimum_valid_fraction,
        "crop_strategy": crop_strategy,
        "crop_bounds": (
            int(ct_rows[0]["crop_top"]), int(ct_rows[0]["crop_bottom"])
        ),
        "crop_bounds_by_day": [
            (int(row["crop_top"]), int(row["crop_bottom"])) for row in ct_rows[:count]
        ],
        "include_qspect_reference": include_qspect_reference,
    }


def samples_to_arrays(samples: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, ...]:
    return (
        np.vstack([sample["features"] for sample in samples]),
        np.asarray([sample["target"] for sample in samples], dtype=np.float64),
        np.asarray([sample["sample_weight"] for sample in samples], dtype=np.float64),
        np.asarray([sample["day_index"] for sample in samples], dtype=np.int64),
    )


def fit_weighted_ridge(
    features: np.ndarray,
    target: np.ndarray,
    sample_weight: np.ndarray,
    alpha: float = 10.0,
) -> Dict[str, np.ndarray | float]:
    """Fit standardized weighted ridge regression without scikit-learn."""
    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] != target.size:
        raise ValueError("Invalid ridge feature/target shapes")
    if alpha < 0.0:
        raise ValueError("Ridge alpha must be non-negative")

    normalized_weight = sample_weight / max(float(np.sum(sample_weight)), 1e-12)
    feature_mean = np.sum(features * normalized_weight[:, None], axis=0)
    centered = features - feature_mean
    feature_variance = np.sum(centered**2 * normalized_weight[:, None], axis=0)
    feature_scale = np.sqrt(np.maximum(feature_variance, 1e-12))
    standardized = centered / feature_scale
    design = np.column_stack([np.ones(features.shape[0]), standardized])

    # Scale weights to an average of one so ``alpha`` keeps the usual ridge
    # interpretation while relative weights still equalize acquisition days.
    fit_weight = sample_weight * (
        features.shape[0] / max(float(np.sum(sample_weight)), 1e-12)
    )
    root_weight = np.sqrt(np.maximum(fit_weight, 0.0))
    weighted_design = design * root_weight[:, None]
    weighted_target = target * root_weight
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    normal_matrix = weighted_design.T @ weighted_design + penalty
    normal_vector = weighted_design.T @ weighted_target
    try:
        coefficients = np.linalg.solve(normal_matrix, normal_vector)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(normal_matrix, normal_vector, rcond=None)[0]

    return {
        "intercept": float(coefficients[0]),
        "coefficients": coefficients[1:],
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "alpha": float(alpha),
    }


def predict_optical_depth(
    model: Dict[str, np.ndarray | float],
    features: np.ndarray,
    maximum_factor: float = ac.CT_FACTOR_CLIP[1],
) -> np.ndarray:
    standardized = (
        np.asarray(features, dtype=np.float64) - np.asarray(model["feature_mean"])
    ) / np.asarray(model["feature_scale"])
    prediction = float(model["intercept"]) + standardized @ np.asarray(model["coefficients"])
    maximum_optical_depth = 2.0 * np.log(float(maximum_factor))
    return np.clip(prediction, 0.0, maximum_optical_depth)


def save_model_bundle(
    model: Dict[str, np.ndarray | float],
    dataset: Dict[str, Any],
    output_path: Path = MODEL_PATH,
) -> Path:
    """Save the fitted emission-only model and its preprocessing settings."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        intercept=np.asarray(model["intercept"], dtype=np.float64),
        coefficients=np.asarray(model["coefficients"], dtype=np.float64),
        feature_mean=np.asarray(model["feature_mean"], dtype=np.float64),
        feature_scale=np.asarray(model["feature_scale"], dtype=np.float64),
        alpha=np.asarray(model["alpha"], dtype=np.float64),
        feature_names=np.asarray(FEATURE_NAMES),
        patch_shape=np.asarray(dataset["patch_shape"], dtype=np.int64),
        crop_bounds=np.asarray(dataset["crop_bounds"], dtype=np.int64),
        mask_threshold_fraction=np.asarray(
            dataset["mask_threshold_fraction"],
            dtype=np.float64,
        ),
        minimum_valid_fraction=np.asarray(
            dataset["minimum_valid_fraction"],
            dtype=np.float64,
        ),
    )
    return output_path


def load_model_bundle(model_path: Path = MODEL_PATH) -> Dict[str, Any]:
    """Load a model saved by :func:`save_model_bundle` without pickle."""
    with np.load(model_path, allow_pickle=False) as bundle:
        return {
            "intercept": float(bundle["intercept"]),
            "coefficients": np.asarray(bundle["coefficients"], dtype=np.float64),
            "feature_mean": np.asarray(bundle["feature_mean"], dtype=np.float64),
            "feature_scale": np.asarray(bundle["feature_scale"], dtype=np.float64),
            "alpha": float(bundle["alpha"]),
            "feature_names": tuple(str(name) for name in bundle["feature_names"]),
            "patch_shape": tuple(int(value) for value in bundle["patch_shape"]),
            "crop_bounds": tuple(int(value) for value in bundle["crop_bounds"]),
            "mask_threshold_fraction": float(bundle["mask_threshold_fraction"]),
            "minimum_valid_fraction": float(bundle["minimum_valid_fraction"]),
        }


def patch_predictions_to_factor_map(
    samples: Sequence[Dict[str, Any]],
    predictions: np.ndarray,
    reference_shape: Tuple[int, int],
    default_optical_depth: float = 0.0,
    relevant_mask: np.ndarray | None = None,
) -> np.ndarray:
    maximum_optical_depth = 2.0 * np.log(ctac.CT_FACTOR_CLIP[1])
    default_factor = np.exp(
        0.5 * np.clip(float(default_optical_depth), 0.0, maximum_optical_depth)
    )
    factor_map = np.ones(reference_shape, dtype=np.float64)
    if relevant_mask is None:
        relevant_mask = np.ones(reference_shape, dtype=bool)
    else:
        relevant_mask = np.asarray(relevant_mask, dtype=bool)
        if relevant_mask.shape != reference_shape:
            raise ValueError(
                f"Relevant-mask shape {relevant_mask.shape} does not match {reference_shape}"
            )
    factor_map[relevant_mask] = default_factor
    for sample, prediction in zip(samples, predictions):
        top, bottom, left, right = sample["bounds"]
        patch_factor = np.exp(0.5 * float(prediction))
        patch_map = factor_map[top:bottom, left:right]
        patch_mask = relevant_mask[top:bottom, left:right]
        if "valid_mask" in sample:
            patch_mask = patch_mask & np.asarray(sample["valid_mask"], dtype=bool)
        patch_map[patch_mask] = patch_factor
    return factor_map


def predict_scan_emission_only(
    scan: Dict[str, Any],
    model_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply a fitted model to one planar scan without loading CT or Q/SPECT."""
    crop_top, crop_bottom = model_bundle["crop_bounds"]
    planar_result = ac.tew_geometric_mean_for_scan(scan)
    full_planar = _positive_array(planar_result["image"])
    if crop_bottom > full_planar.shape[0]:
        raise ValueError(
            f"Saved crop {(crop_top, crop_bottom)} exceeds planar height "
            f"{full_planar.shape[0]}"
        )
    planar_crop = full_planar[crop_top:crop_bottom, :]
    placeholder_reference = {
        "crop_top": crop_top,
        "crop_bottom": crop_bottom,
        "planar_crop": planar_crop,
        "ct_mu_integral_resized": np.zeros_like(planar_crop),
        "day_label": "inference",
        "day_offset": np.nan,
    }
    samples = patch_samples_for_day(
        scan,
        placeholder_reference,
        day_index=0,
        patch_shape=tuple(model_bundle["patch_shape"]),
        mask_threshold_fraction=float(model_bundle["mask_threshold_fraction"]),
        minimum_valid_fraction=float(model_bundle["minimum_valid_fraction"]),
    )
    features = np.vstack([sample["features"] for sample in samples])
    predictions = predict_optical_depth(model_bundle, features)
    factor_map = patch_predictions_to_factor_map(
        samples,
        predictions,
        planar_crop.shape,
        default_optical_depth=float(model_bundle["intercept"]),
        relevant_mask=planar_crop > 0.0,
    )
    corrected_crop = planar_crop * factor_map
    before_counts = float(np.sum(planar_crop))
    after_counts = float(np.sum(corrected_crop))
    return {
        "planar_crop": planar_crop,
        "factor_map": factor_map,
        "corrected_crop": corrected_crop,
        "planar_crop_counts": before_counts,
        "corrected_crop_counts": after_counts,
        "effective_factor": after_counts / before_counts if before_counts > 0.0 else np.nan,
        "patch_predictions": predictions,
        "patch_samples": samples,
        "uses_ct": False,
        "uses_qspect": False,
    }


def leave_one_day_out(
    dataset: Dict[str, Any],
    alpha: float = 10.0,
) -> Dict[str, Any]:
    """Hold out every acquisition day once and collect patch/map metrics."""
    samples = dataset["samples"]
    features, target, sample_weight, day_index = samples_to_arrays(samples)
    unique_days = np.unique(day_index)
    predictions = np.full(target.shape, np.nan, dtype=np.float64)
    baseline_predictions = np.full(target.shape, np.nan, dtype=np.float64)
    folds: List[Dict[str, Any]] = []

    for held_out_day in unique_days:
        train_mask = day_index != held_out_day
        test_mask = day_index == held_out_day
        model = fit_weighted_ridge(
            features[train_mask],
            target[train_mask],
            sample_weight[train_mask],
            alpha=alpha,
        )
        fold_prediction = predict_optical_depth(model, features[test_mask])
        predictions[test_mask] = fold_prediction
        baseline_optical_depth = _weighted_mean(
            target[train_mask],
            sample_weight[train_mask],
        )
        baseline_fold_prediction = np.full(
            int(np.count_nonzero(test_mask)),
            baseline_optical_depth,
            dtype=np.float64,
        )
        baseline_predictions[test_mask] = baseline_fold_prediction
        test_samples = [sample for sample in samples if sample["day_index"] == int(held_out_day)]
        ct_row = dataset["ct_rows"][int(held_out_day)]
        planar_crop = _positive_array(ct_row["planar_crop"])
        relevant_mask = planar_crop > 0.0
        factor_map = patch_predictions_to_factor_map(
            test_samples,
            fold_prediction,
            tuple(ct_row["planar_crop"].shape),
            default_optical_depth=float(model["intercept"]),
            relevant_mask=relevant_mask,
        )
        baseline_factor_map = patch_predictions_to_factor_map(
            test_samples,
            baseline_fold_prediction,
            tuple(ct_row["planar_crop"].shape),
            default_optical_depth=baseline_optical_depth,
            relevant_mask=relevant_mask,
        )
        predicted_counts = float(np.sum(planar_crop * factor_map))
        baseline_counts = float(np.sum(planar_crop * baseline_factor_map))
        predicted_activity = planar_processing.counts_to_activity_mbq(
            predicted_counts,
            float(ct_row["local_dwell_time_s"]),
            planar_processing.CAMERA_SENSITIVITY_CPS_PER_MBQ,
        )
        baseline_activity = planar_processing.counts_to_activity_mbq(
            baseline_counts,
            float(ct_row["local_dwell_time_s"]),
            planar_processing.CAMERA_SENSITIVITY_CPS_PER_MBQ,
        )
        qspect_activity = float(ct_row.get("qspect_activity_mbq", np.nan))
        fold = {
            "held_out_day": int(held_out_day),
            "label": str(ct_row["day_label"]),
            "day_offset": float(ct_row["day_offset"]),
            "patch_count": int(np.count_nonzero(test_mask)),
            "model": model,
            "factor_map": factor_map,
            "predicted_ctac_counts": predicted_counts,
            "predicted_ctac_activity_mbq": predicted_activity,
            "baseline_ctac_counts": baseline_counts,
            "baseline_ctac_activity_mbq": baseline_activity,
            "oracle_ctac_activity_mbq": float(ct_row["ctac_local_activity_mbq"]),
            "qspect_activity_mbq": qspect_activity,
            "true_effective_factor": float(ct_row["effective_ct_factor"]),
            "predicted_effective_factor": (
                predicted_counts / float(ct_row["planar_crop_counts"])
                if float(ct_row["planar_crop_counts"]) > 0.0
                else np.nan
            ),
            "baseline_effective_factor": (
                baseline_counts / float(ct_row["planar_crop_counts"])
                if float(ct_row["planar_crop_counts"]) > 0.0
                else np.nan
            ),
        }
        fold.update(_weighted_metrics(target[test_mask], fold_prediction, sample_weight[test_mask]))
        baseline_metrics = _weighted_metrics(
            target[test_mask],
            baseline_fold_prediction,
            sample_weight[test_mask],
        )
        fold.update(
            {f"baseline_{name}": value for name, value in baseline_metrics.items()}
        )
        fold["oracle_activity_error_percent"] = 100.0 * (
            predicted_activity - fold["oracle_ctac_activity_mbq"]
        ) / fold["oracle_ctac_activity_mbq"]
        fold["baseline_oracle_activity_error_percent"] = 100.0 * (
            baseline_activity - fold["oracle_ctac_activity_mbq"]
        ) / fold["oracle_ctac_activity_mbq"]
        fold["qspect_activity_error_percent"] = (
            100.0
            * (predicted_activity - qspect_activity)
            / qspect_activity
            if np.isfinite(qspect_activity) and qspect_activity > 0.0
            else np.nan
        )
        folds.append(fold)

    if np.any(~np.isfinite(predictions)):
        raise RuntimeError("Leave-one-day-out predictions are incomplete")
    if np.any(~np.isfinite(baseline_predictions)):
        raise RuntimeError("Leave-one-day-out baseline predictions are incomplete")
    overall = _weighted_metrics(target, predictions, sample_weight)
    baseline_overall = _weighted_metrics(target, baseline_predictions, sample_weight)
    full_model = fit_weighted_ridge(features, target, sample_weight, alpha=alpha)
    return {
        "folds": folds,
        "predictions": predictions,
        "baseline_predictions": baseline_predictions,
        "target": target,
        "sample_weight": sample_weight,
        "day_index": day_index,
        "overall": overall,
        "baseline_overall": baseline_overall,
        "full_model": full_model,
        "alpha": float(alpha),
    }


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    line = lambda values: " | ".join(
        str(value).rjust(widths[index]) for index, value in enumerate(values)
    )
    return [line(headers), line(["-" * width for width in widths])] + [line(row) for row in rows]


def _format_optional(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}" if np.isfinite(value) else "n/a"


def write_report(
    dataset: Dict[str, Any],
    validation: Dict[str, Any],
    proxy_only_validation: Dict[str, Any] | None = None,
    output_path: Path = REPORT_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    has_qspect = any(
        np.isfinite(fold["qspect_activity_mbq"])
        for fold in validation["folds"]
    )
    headers = [
        "held out",
        "patches",
        "M RMSE",
        "M R2",
        "weighted model MAPE %",
        "weighted constant MAPE %",
        "Ceff CT",
        "Ceff model",
        "model err %",
        "constant err %",
    ]
    if has_qspect:
        headers.append("QSP err %")
    rows = []
    for fold in validation["folds"]:
        row = [
            fold["label"],
            str(fold["patch_count"]),
            f"{fold['rmse_optical_depth']:.3f}",
            f"{fold['r2_optical_depth']:.3f}",
            f"{fold['mean_absolute_factor_error_percent']:.1f}",
            f"{fold['baseline_mean_absolute_factor_error_percent']:.1f}",
            f"{fold['true_effective_factor']:.3f}",
            f"{fold['predicted_effective_factor']:.3f}",
            f"{fold['oracle_activity_error_percent']:.1f}",
            f"{fold['baseline_oracle_activity_error_percent']:.1f}",
        ]
        if has_qspect:
            row.append(_format_optional(fold["qspect_activity_error_percent"], digits=1))
        rows.append(row)
    coefficient_rows = [
        [feature_name, f"{coefficient:.5f}"]
        for feature_name, coefficient in zip(
            FEATURE_NAMES,
            np.asarray(validation["full_model"]["coefficients"]),
        )
    ]
    ct_series_rows = [
        [
            row["day_label"],
            f"{row['delta_z_cm']:.6f}",
            "x".join(str(size) for size in row["ct_shape"]),
            row["ct_series_description"],
        ]
        for row in dataset["ct_rows"]
    ]
    support_rows = [
        [
            row["day_label"],
            f"{100.0 * row['emission_mask_pixel_fraction']:.1f}",
            f"{100.0 * row['emission_mask_count_fraction']:.1f}",
            f"{100.0 * row['training_support_pixel_fraction']:.1f}",
            f"{100.0 * row['training_support_count_fraction']:.1f}",
        ]
        for row in dataset["ct_rows"]
    ]
    overall = validation["overall"]
    baseline_overall = validation["baseline_overall"]
    proxy_overall = (
        proxy_only_validation["overall"]
        if proxy_only_validation is not None
        else None
    )
    lines = [
        "One-patient Lu-177 emission-window attenuation model",
        "====================================================",
        "",
        "Purpose:",
        "  Predict the CT-derived effective patch correction M_eff from planar windows.",
        "  Raw ACCT is the development reference. Q/SPECT activity is not fitted.",
        "  Time is not fitted directly; changes over time enter through the observed window features.",
        "  The null comparison predicts the weighted mean M_eff of the training days.",
        "",
        "CT projection:",
        "  M_CT(x,y) = sum_z(mu_208(x,y,z)) * delta_z",
        "  where delta_z is constant within each selected CT series.",
        "  Patch target: M_eff = 2*log(sum(C_TEW*exp(M_CT/2))/sum(C_TEW)).",
        "",
        "Configuration:",
        f"  crop_strategy = {dataset['crop_strategy']}",
        "  crop_bounds_by_day = " + ", ".join(
            f"{row['day_label']}[{row['crop_top']}:{row['crop_bottom']}]"
            for row in dataset["ct_rows"]
        ),
        f"  patch_shape = {dataset['patch_shape'][0]} x {dataset['patch_shape'][1]} pixels",
        f"  mask_threshold_fraction = {dataset['mask_threshold_fraction']:.4f}",
        f"  minimum_valid_fraction = {dataset['minimum_valid_fraction']:.3f}",
        f"  ridge_alpha = {validation['alpha']:.3f}",
        f"  total_patches = {len(dataset['samples'])}",
        "",
        "CT reference series:",
    ]
    lines.extend(
        _format_table(
            ["day", "delta_AP (cm)", "CT shape", "series"],
            ct_series_rows,
        )
    )
    lines.extend(["", "Emission-defined support:"])
    lines.extend(
        _format_table(
            ["day", "mask pixels %", "mask counts %", "fit pixels %", "fit counts %"],
            support_rows,
        )
    )
    lines.extend(["", "Leave-one-day-out validation:"])
    lines.extend(_format_table(headers, rows))
    lines.extend(
        [
            "",
            "Overall patch metrics:",
            f"  optical-depth MAE = {overall['mae_optical_depth']:.4f}",
            f"  optical-depth RMSE = {overall['rmse_optical_depth']:.4f}",
            f"  optical-depth R2 = {overall['r2_optical_depth']:.4f}",
            f"  weighted attenuation-factor MAPE = {overall['mean_absolute_factor_error_percent']:.2f}%",
            f"  weighted constant-baseline factor MAPE = {baseline_overall['mean_absolute_factor_error_percent']:.2f}%",
            *(
                [
                    "  weighted attenuation-proxy-only MAPE "
                    f"(without relative photopeak) = "
                    f"{proxy_overall['mean_absolute_factor_error_percent']:.2f}%"
                ]
                if proxy_overall is not None
                else []
            ),
            "",
            "Full-data standardized ridge coefficients (descriptive only):",
        ]
    )
    lines.extend(_format_table(["feature", "coefficient"], coefficient_rows))
    lines.extend(
        [
            "",
            "Interpretation limits:",
            "  - Patches from one patient are not independent patient samples.",
            "  - The fixed patient crop is explicit and is not learned from Q/SPECT.",
            "  - Patch targets and support are count-weighted, so biodistribution and noise affect sampling.",
            "  - The relative-photopeak feature can encode biodistribution; compare its reported ablation.",
            "  - CT-to-planar mapping is approximate resizing, not DICOM registration.",
            f"  - CT target conversion = {dataset['ct_rows'][0]['ct_conversion_method']}.",
            "  - The default Catphan T2 curve is provisional until the QC reconstruction matches the patient ACCT protocol.",
            "  - AP/PA orientation and registration still require geometry-aware verification.",
            "  - Optional Q/SPECT output is descriptive only and never a regression target.",
            "  - Generalization requires phantoms and additional patients.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def plot_validation(
    dataset: Dict[str, Any],
    validation: Dict[str, Any],
    output_path: Path = VALIDATION_FIGURE_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target = validation["target"]
    prediction = validation["predictions"]
    folds = sorted(validation["folds"], key=lambda fold: fold["day_offset"])
    days = np.asarray([fold["day_offset"] for fold in folds])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), facecolor="white")
    axes[0].scatter(target, prediction, s=8, alpha=0.18, color="tab:blue")
    upper = max(float(np.max(target)), float(np.max(prediction)), 1.0)
    axes[0].plot([0.0, upper], [0.0, upper], linestyle="--", color="0.25")
    axes[0].set_xlabel("CT-derived effective optical depth")
    axes[0].set_ylabel("Predicted effective optical depth")
    axes[0].set_title("Held-out patch correction")

    axes[1].plot(
        days,
        [fold["true_effective_factor"] for fold in folds],
        marker="o",
        linewidth=2,
        label="CT-derived reference",
    )
    axes[1].plot(
        days,
        [fold["predicted_effective_factor"] for fold in folds],
        marker="s",
        linewidth=2,
        label="Window model",
    )
    axes[1].plot(
        days,
        [fold["baseline_effective_factor"] for fold in folds],
        marker="^",
        linestyle="--",
        color="0.45",
        linewidth=1.5,
        label="Constant baseline",
    )
    axes[1].set_xlabel("Time after first acquisition (days)")
    axes[1].set_ylabel("Effective attenuation factor")
    axes[1].set_title("Leave-one-day-out factor")
    axes[1].legend(frameon=False)

    axes[2].plot(
        days,
        [fold["predicted_ctac_activity_mbq"] for fold in folds],
        marker="s",
        linewidth=2,
        label="Window model",
    )
    axes[2].plot(
        days,
        [fold["oracle_ctac_activity_mbq"] for fold in folds],
        marker="o",
        linewidth=2,
        label="CT-derived reference",
    )
    axes[2].plot(
        days,
        [fold["baseline_ctac_activity_mbq"] for fold in folds],
        marker="^",
        linestyle="--",
        color="0.45",
        linewidth=1.5,
        label="Constant baseline",
    )
    qspect_activities = np.asarray(
        [fold["qspect_activity_mbq"] for fold in folds],
        dtype=np.float64,
    )
    if np.any(np.isfinite(qspect_activities)):
        axes[2].plot(
            days,
            qspect_activities,
            marker="^",
            linewidth=2,
            label="Q/SPECT reference",
        )
    axes[2].set_xlabel("Time after first acquisition (days)")
    axes[2].set_ylabel("Activity estimate (MBq)")
    axes[2].set_title("CT-derived activity comparison")
    axes[2].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_last_day_map(
    dataset: Dict[str, Any],
    validation: Dict[str, Any],
    output_path: Path = MAP_FIGURE_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fold = max(validation["folds"], key=lambda item: item["day_offset"])
    row = dataset["ct_rows"][fold["held_out_day"]]
    true_factor = np.exp(
        np.clip(
            0.5 * np.asarray(row["ct_mu_integral_resized"], dtype=np.float64),
            np.log(ctac.CT_FACTOR_CLIP[0]),
            np.log(ctac.CT_FACTOR_CLIP[1]),
        )
    )
    predicted_factor = np.asarray(fold["factor_map"], dtype=np.float64)
    planar_crop = _positive_array(row["planar_crop"])
    support_threshold = dataset["mask_threshold_fraction"] * float(np.max(planar_crop))
    evaluation_support = planar_crop >= support_threshold
    predicted_display = np.ma.masked_where(~evaluation_support, predicted_factor)
    absolute_error = np.ma.masked_where(
        ~evaluation_support,
        np.abs(predicted_factor - true_factor),
    )
    factor_cmap = plt.get_cmap("viridis").copy()
    factor_cmap.set_bad("black")
    error_cmap = plt.get_cmap("inferno").copy()
    error_cmap.set_bad("black")

    fig, axes = plt.subplots(1, 4, figsize=(13, 4.2), facecolor="white")
    axes[0].imshow(row["planar_crop"], cmap="magma", aspect="auto")
    axes[0].set_title(f"{fold['label']} planar GM")
    shared_max = float(max(np.percentile(true_factor, 99), np.percentile(predicted_factor, 99)))
    axes[1].imshow(true_factor, cmap="viridis", vmin=1.0, vmax=shared_max, aspect="auto")
    axes[1].set_title("CT attenuation factor")
    axes[2].imshow(
        predicted_display,
        cmap=factor_cmap,
        vmin=1.0,
        vmax=shared_max,
        aspect="auto",
    )
    axes[2].set_title("Predicted factor (emission support)")
    error_image = axes[3].imshow(absolute_error, cmap=error_cmap, aspect="auto")
    axes[3].set_title("Factor error (emission support)")
    fig.colorbar(error_image, ax=axes[3], fraction=0.046, pad=0.02)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def run_model(
    planar_dir: Path | None = None,
    ct_root: Path | None = None,
    patch_shape: Tuple[int, int] = (16, 8),
    mask_threshold_fraction: float = 0.005,
    minimum_valid_fraction: float = 0.25,
    alpha: float = 10.0,
    crop_bounds: Optional[Tuple[int, int]] = None,
    crop_strategy: str = "profile_qspect",
    include_qspect_reference: bool = False,
    conversion_method: str = ctac.DEFAULT_CT_CONVERSION_METHOD,
) -> Dict[str, Any]:
    dataset = build_dataset(
        planar_dir=(planar_dir or planar_processing.default_planar_study_dir()),
        ct_root=(ct_root or qspect_processing.default_qspect_dir()),
        patch_shape=patch_shape,
        mask_threshold_fraction=mask_threshold_fraction,
        minimum_valid_fraction=minimum_valid_fraction,
        crop_bounds=crop_bounds,
        crop_strategy=crop_strategy,
        include_qspect_reference=include_qspect_reference,
        conversion_method=conversion_method,
    )
    validation = leave_one_day_out(dataset, alpha=alpha)
    proxy_only_dataset = {
        **dataset,
        "samples": [
            {**sample, "features": np.asarray(sample["features"])[:4]}
            for sample in dataset["samples"]
        ],
    }
    proxy_only_validation = leave_one_day_out(proxy_only_dataset, alpha=alpha)
    report_path = write_report(
        dataset,
        validation,
        proxy_only_validation=proxy_only_validation,
    )
    figure_path = plot_validation(dataset, validation)
    map_path = plot_last_day_map(dataset, validation)
    model_path = save_model_bundle(validation["full_model"], dataset)
    print(f"Saved window-model report: {report_path}")
    print(f"Saved validation figure: {figure_path}")
    print(f"Saved last-day map figure: {map_path}")
    print(f"Saved fitted emission-only model: {model_path}")
    print(
        "Leave-one-day-out: "
        f"M RMSE={validation['overall']['rmse_optical_depth']:.3f}, "
        f"weighted factor MAPE={validation['overall']['mean_absolute_factor_error_percent']:.1f}%, "
        "proxy-only MAPE="
        f"{proxy_only_validation['overall']['mean_absolute_factor_error_percent']:.1f}%"
    )
    return {
        "dataset": dataset,
        "validation": validation,
        "proxy_only_validation": proxy_only_validation,
        "model_path": model_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict CT attenuation optical depth from planar Lu-177 energy windows."
    )
    parser.add_argument("--patch-height", type=int, default=16)
    parser.add_argument("--patch-width", type=int, default=8)
    parser.add_argument("--mask-threshold", type=float, default=0.005)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--crop-top", type=int, default=141)
    parser.add_argument("--crop-bottom", type=int, default=609)
    parser.add_argument(
        "--crop-strategy",
        choices=("profile_qspect", "fixed"),
        default="profile_qspect",
    )
    parser.add_argument("--planar-dir", type=Path)
    parser.add_argument("--ct-root", type=Path)
    parser.add_argument(
        "--ct-conversion-method",
        choices=("water_scaled", "raystation_materials", "catphan_t2_110kvp"),
        default=ctac.DEFAULT_CT_CONVERSION_METHOD,
    )
    parser.add_argument(
        "--include-qspect-reference",
        action="store_true",
        help="Add Q/SPECT to the final plot/report only; never use it for fitting.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_model(
        planar_dir=arguments.planar_dir,
        ct_root=arguments.ct_root,
        patch_shape=(arguments.patch_height, arguments.patch_width),
        mask_threshold_fraction=arguments.mask_threshold,
        minimum_valid_fraction=arguments.minimum_valid_fraction,
        alpha=arguments.alpha,
        crop_bounds=(arguments.crop_top, arguments.crop_bottom),
        crop_strategy=arguments.crop_strategy,
        include_qspect_reference=arguments.include_qspect_reference,
        conversion_method=arguments.ct_conversion_method,
    )
