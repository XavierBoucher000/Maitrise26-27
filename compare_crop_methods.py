"""Compare three longitudinal crop strategies for the one-patient CTAC study.

Strategies
----------
``variable_j``
    Existing crop recalculated independently from planar/Q/SPECT measurements.
``fixed_j0``
    Existing Day-0 crop bounds reused without movement.
``registered_j0``
    New planar-only follow-up method.  It preserves the Day-0 crop height and
    estimates only a bounded longitudinal translation relative to the Day-0
    planar multi-window profile.

Q/SPECT is used to define the development reference crop at Day 0 and to report
quality-control metrics.  It is not used to estimate the registered shifts at
later time points.
"""

import gc
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, gaussian_filter1d

import attenuation_correction as ac
import planar_processing
import qspect_processing


OUTPUT_DIR = Path(__file__).resolve().parent / "fig" / "crop_method_comparison"
BOUNDS_FIGURE_PATH = OUTPUT_DIR / "crop_bounds_four_methods.png"
METRICS_FIGURE_PATH = OUTPUT_DIR / "crop_metrics_four_methods.png"
ACTIVITY_FIGURE_PATH = OUTPUT_DIR / "activity_curves_four_crop_methods.png"
RATIO_FIGURE_PATH = OUTPUT_DIR / "qspect_ratios_four_crop_methods.png"
VALUES_PATH = OUTPUT_DIR / "crop_methods_values.csv"
REPORT_PATH = OUTPUT_DIR / "crop_methods_report.txt"

REGISTRATION_ENERGIES = (
    "Lower Scatter",
    "Photopeak",
    "Upper Scatter",
    "Low Energy Scatter",
)
REGISTRATION_MAX_SHIFT_PX = 40
REGISTRATION_SMOOTHING_SIGMA_PX = 8.0
SILHOUETTE_SMOOTHING_SIGMA_PX = 4.0
SILHOUETTE_RELATIVE_THRESHOLD = 0.01
SILHOUETTE_MIN_ROW_WIDTH_PX = 8
SILHOUETTE_MIN_LENGTH_RATIO = 0.80
SILHOUETTE_MAX_LENGTH_RATIO = 1.20
SILHOUETTE_MAX_EDGE_DISAGREEMENT_PX = 12


def planar_multiwindow_registration_proxy(
    scan: Dict[str, Any],
    energies: Sequence[str] = REGISTRATION_ENERGIES,
) -> np.ndarray:
    """Build a planar-only positioning image from normalized AP/PA windows.

    Each available geometric-mean window is percentile-normalized before the
    windows are averaged.  This prevents the broad low-energy window or a hot
    organ from controlling the registration solely because of count scale.
    """
    groups = ac.group_by_energy_and_view(scan["images"])
    normalized_images = []
    for energy in energies:
        views = groups.get(energy)
        if not views or "AP" not in views or "PA" not in views:
            continue
        image = ac.geometric_mean(
            views["AP"]["image"],
            views["PA"]["image"],
            align_pa_to_ap=True,
        )
        image = np.clip(np.asarray(image, dtype=np.float64), 0.0, None)
        positive = image[image > 0.0]
        if positive.size == 0:
            continue
        scale = float(np.percentile(positive, 99.0))
        normalized_images.append(np.clip(image / max(scale, 1e-12), 0.0, 1.0))
    if not normalized_images:
        raise ValueError("No AP/PA energy-window pair is available for crop registration")
    return np.mean(normalized_images, axis=0)


def robust_longitudinal_profile(
    image: np.ndarray,
    smoothing_sigma_px: float = REGISTRATION_SMOOTHING_SIGMA_PX,
) -> np.ndarray:
    """Return a log-compressed, smoothed and normalized head-to-feet profile."""
    image = np.nan_to_num(
        np.asarray(image, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    if image.ndim != 2:
        raise ValueError("Registration proxy must be a 2D image")
    if smoothing_sigma_px < 0.0:
        raise ValueError("smoothing_sigma_px must be non-negative")
    profile = np.sum(np.clip(image, 0.0, None), axis=1)
    positive = profile[profile > 0.0]
    if positive.size == 0:
        raise ValueError("Cannot register an empty planar profile")
    scale = float(np.percentile(positive, 95.0))
    profile = np.log1p(profile / max(scale, 1e-12))
    if smoothing_sigma_px > 0.0:
        profile = gaussian_filter1d(profile, smoothing_sigma_px, mode="nearest")
    maximum = float(profile.max())
    return profile / maximum if maximum > 0.0 else profile


def estimate_planar_longitudinal_shift(
    reference_profile: np.ndarray,
    moving_profile: np.ndarray,
    max_shift_px: int = REGISTRATION_MAX_SHIFT_PX,
) -> Dict[str, Any]:
    """Estimate a bounded integer row shift by normalized profile correlation.

    A positive result means that the corresponding signal appears at larger row
    indices in the moving acquisition and the Day-0 crop must move downward.
    """
    reference = np.asarray(reference_profile, dtype=np.float64)
    moving = np.asarray(moving_profile, dtype=np.float64)
    if reference.ndim != 1 or moving.ndim != 1 or reference.shape != moving.shape:
        raise ValueError("Reference and moving profiles must be equal-length 1D arrays")
    if max_shift_px < 0 or max_shift_px >= reference.size:
        raise ValueError("max_shift_px must be non-negative and smaller than the profile")

    shifts = np.arange(-int(max_shift_px), int(max_shift_px) + 1, dtype=int)
    scores = np.full(shifts.shape, np.nan, dtype=np.float64)
    for index, shift in enumerate(shifts):
        if shift >= 0:
            ref_overlap = reference[: reference.size - shift]
            mov_overlap = moving[shift:]
        else:
            ref_overlap = reference[-shift:]
            mov_overlap = moving[: moving.size + shift]
        support = (ref_overlap >= 0.02) | (mov_overlap >= 0.02)
        ref_signal = ref_overlap[support]
        mov_signal = mov_overlap[support]
        if ref_signal.size < 20:
            continue
        ref_centered = ref_signal - float(ref_signal.mean())
        mov_centered = mov_signal - float(mov_signal.mean())
        denominator = float(np.linalg.norm(ref_centered) * np.linalg.norm(mov_centered))
        if denominator > 0.0:
            scores[index] = float(np.dot(ref_centered, mov_centered) / denominator)

    if not np.any(np.isfinite(scores)):
        raise ValueError("No valid longitudinal crop-registration score was obtained")
    best_index = int(np.nanargmax(scores))
    return {
        "shift_y_px": int(shifts[best_index]),
        "correlation": float(scores[best_index]),
        "shifts_px": shifts,
        "scores": scores,
    }


def translate_crop_bounds(
    day0_bounds: Tuple[int, int],
    shift_y_px: int,
    image_height: int,
) -> Tuple[int, int]:
    """Translate Day-0 bounds while preserving their exact crop height."""
    top, bottom = (int(day0_bounds[0]), int(day0_bounds[1]))
    height = bottom - top
    if height <= 0 or height > image_height:
        raise ValueError("Invalid Day-0 crop bounds")
    shifted_top = top + int(shift_y_px)
    shifted_top = min(max(0, shifted_top), image_height - height)
    return shifted_top, shifted_top + height


def planar_body_envelope(
    image: np.ndarray,
    smoothing_sigma_px: float = SILHOUETTE_SMOOTHING_SIGMA_PX,
    relative_threshold: float = SILHOUETTE_RELATIVE_THRESHOLD,
    minimum_row_width_px: int = SILHOUETTE_MIN_ROW_WIDTH_PX,
) -> Dict[str, int]:
    """Estimate a low-threshold whole-body envelope from a planar proxy.

    The envelope uses spatial occupancy rather than summed activity. A row is
    retained only when several columns contain signal, which reduces the
    influence of an isolated hot organ.
    """
    image = np.clip(np.asarray(image, dtype=np.float64), 0.0, None)
    if image.ndim != 2:
        raise ValueError("Silhouette proxy must be a 2D image")
    if not 0.0 < relative_threshold < 1.0:
        raise ValueError("relative_threshold must be between 0 and 1")
    if minimum_row_width_px <= 0 or minimum_row_width_px > image.shape[1]:
        raise ValueError("minimum_row_width_px is incompatible with the image width")
    smoothed = gaussian_filter(
        image,
        sigma=(float(smoothing_sigma_px), float(smoothing_sigma_px)),
        mode="nearest",
    )
    positive = smoothed[smoothed > 0.0]
    if positive.size == 0:
        raise ValueError("Cannot estimate a silhouette from an empty image")
    scale = float(np.percentile(positive, 99.0))
    normalized = smoothed / max(scale, 1e-12)
    row_width = np.count_nonzero(normalized >= relative_threshold, axis=1)
    rows = np.flatnonzero(row_width >= int(minimum_row_width_px))
    if rows.size == 0:
        raise ValueError("No body-envelope row passed the occupancy criterion")
    top = int(rows[0])
    bottom = int(rows[-1]) + 1
    return {"top": top, "bottom": bottom, "length": bottom - top}


def estimate_silhouette_shift(
    reference_envelope: Dict[str, int],
    moving_envelope: Dict[str, int],
    max_shift_px: int = REGISTRATION_MAX_SHIFT_PX,
) -> Dict[str, Any]:
    """Estimate a shift from both body edges and reject incomplete silhouettes."""
    top_shift = int(moving_envelope["top"]) - int(reference_envelope["top"])
    bottom_shift = int(moving_envelope["bottom"]) - int(reference_envelope["bottom"])
    reference_length = int(reference_envelope["length"])
    moving_length = int(moving_envelope["length"])
    if reference_length <= 0 or moving_length <= 0:
        raise ValueError("Silhouette lengths must be positive")
    length_ratio = float(moving_length) / float(reference_length)
    edge_disagreement = abs(top_shift - bottom_shift)
    raw_shift = int(np.rint(np.median([top_shift, bottom_shift])))
    accepted = (
        SILHOUETTE_MIN_LENGTH_RATIO <= length_ratio <= SILHOUETTE_MAX_LENGTH_RATIO
        and edge_disagreement <= SILHOUETTE_MAX_EDGE_DISAGREEMENT_PX
        and abs(raw_shift) <= int(max_shift_px)
    )
    return {
        "raw_shift_y_px": raw_shift,
        "shift_y_px": raw_shift if accepted else 0,
        "accepted": bool(accepted),
        "top_shift_y_px": top_shift,
        "bottom_shift_y_px": bottom_shift,
        "edge_disagreement_px": edge_disagreement,
        "length_ratio": length_ratio,
    }


def registered_day0_rows(
    fixed_rows: List[Dict[str, Any]],
    scans: List[Dict[str, Any]],
    qspect_series: List[Dict[str, Any]],
    qspect_dir: Path,
    conversion_method: str = "water_scaled",
) -> List[Dict[str, Any]]:
    """Apply the new Day-0 crop plus planar-only longitudinal registration."""
    if not fixed_rows or not scans or not qspect_series:
        raise ValueError("Fixed rows, planar scans and Q/SPECT series are required")
    count = min(len(fixed_rows), len(scans), len(qspect_series))
    day0_bounds = (int(fixed_rows[0]["crop_top"]), int(fixed_rows[0]["crop_bottom"]))
    reference_proxy = planar_multiwindow_registration_proxy(scans[0])
    reference_profile = robust_longitudinal_profile(reference_proxy)

    rows = []
    for index in range(count):
        moving_proxy = planar_multiwindow_registration_proxy(scans[index])
        moving_profile = robust_longitudinal_profile(moving_proxy)
        registration = estimate_planar_longitudinal_shift(
            reference_profile,
            moving_profile,
        )
        bounds = translate_crop_bounds(
            day0_bounds,
            registration["shift_y_px"],
            moving_proxy.shape[0],
        )
        row = ac.ct_attenuation_correct_scan(
            scans[index],
            qspect_series[index],
            qspect_dir,
            fixed_crop_bounds=bounds,
            conversion_method=conversion_method,
            align_pa_to_ap=True,
        )
        row["crop_strategy"] = "registered_j0"
        row["day_offset"] = float(fixed_rows[index]["day_offset"])
        row["crop_shift_y_px"] = int(registration["shift_y_px"])
        row["crop_shift_cm"] = (
            registration["shift_y_px"]
            * float(scans[index]["images"][0]["pixel_spacing"][0])
            / 10.0
        )
        row["profile_registration_correlation"] = float(registration["correlation"])
        row["registration_reference_profile"] = reference_profile
        row["registration_moving_profile"] = moving_profile
        qspect_activity = float(row["qspect_activity_mbq"])
        row["ctac_over_qspect"] = float(row["ctac_local_activity_mbq"]) / qspect_activity
        rows.append(row)
    return rows


def silhouette_day0_rows(
    fixed_rows: List[Dict[str, Any]],
    scans: List[Dict[str, Any]],
    qspect_series: List[Dict[str, Any]],
    qspect_dir: Path,
    conversion_method: str = "water_scaled",
) -> List[Dict[str, Any]]:
    """Preserve the Day-0 height and move it only with a valid body silhouette."""
    if not fixed_rows or not scans or not qspect_series:
        raise ValueError("Fixed rows, planar scans and Q/SPECT series are required")
    count = min(len(fixed_rows), len(scans), len(qspect_series))
    day0_bounds = (int(fixed_rows[0]["crop_top"]), int(fixed_rows[0]["crop_bottom"]))
    proxies = [planar_multiwindow_registration_proxy(scan) for scan in scans[:count]]
    reference_envelope = planar_body_envelope(proxies[0])

    rows = []
    for index in range(count):
        moving_envelope = planar_body_envelope(proxies[index])
        registration = estimate_silhouette_shift(reference_envelope, moving_envelope)
        bounds = translate_crop_bounds(
            day0_bounds,
            registration["shift_y_px"],
            proxies[index].shape[0],
        )
        if registration["shift_y_px"] == 0:
            full_row = fixed_rows[index]
        else:
            full_row = ac.ct_attenuation_correct_scan(
                scans[index],
                qspect_series[index],
                qspect_dir,
                fixed_crop_bounds=bounds,
                conversion_method=conversion_method,
                align_pa_to_ap=True,
            )
        metrics = ac.crop_alignment_metrics(full_row)
        excluded = full_row["crop_excluded_counts"]
        rows.append(
            {
                "crop_strategy": "silhouette_j0",
                "day_offset": float(fixed_rows[index]["day_offset"]),
                "qspect_label": str(full_row["qspect_label"]),
                "crop_top": int(full_row["crop_top"]),
                "crop_bottom": int(full_row["crop_bottom"]),
                "crop_height_cm": float(full_row["crop_height_cm"]),
                "planar_crop_height_px": int(full_row["planar_crop"].shape[0]),
                "ctac_local_activity_mbq": float(full_row["ctac_local_activity_mbq"]),
                "qspect_activity_mbq": float(full_row["qspect_activity_mbq"]),
                "precomputed_alignment_metrics": metrics,
                "outside_fraction": float(excluded["outside_fraction"]),
                "crop_shift_y_px": int(registration["shift_y_px"]),
                "silhouette_raw_shift_y_px": int(registration["raw_shift_y_px"]),
                "silhouette_shift_accepted": bool(registration["accepted"]),
                "silhouette_length_ratio": float(registration["length_ratio"]),
                "silhouette_edge_disagreement_px": int(
                    registration["edge_disagreement_px"]
                ),
                "silhouette_reference_envelope": reference_envelope,
                "silhouette_moving_envelope": moving_envelope,
            }
        )
    return rows


def compare_crop_methods(
    conversion_method: str = "water_scaled",
) -> Dict[str, List[Dict[str, Any]]]:
    """Calculate variable, fixed-J0 and registered-J0 crop strategies."""
    planar_dir = planar_processing.default_planar_study_dir()
    qspect_dir = qspect_processing.default_qspect_dir()
    scans = ac.sorted_planar_scans(planar_dir)
    qspect_series = qspect_processing.load_qspect_study(qspect_dir)
    variable_rows = ac.ct_attenuation_correction_rows(
        planar_dir=planar_dir,
        qspect_dir=qspect_dir,
        crop_strategy="individual",
        conversion_method=conversion_method,
        align_pa_to_ap=True,
    )
    fixed_rows = ac.ct_attenuation_correction_rows(
        planar_dir=planar_dir,
        qspect_dir=qspect_dir,
        crop_strategy="fixed_day0",
        conversion_method=conversion_method,
        align_pa_to_ap=True,
    )
    registered_rows = registered_day0_rows(
        fixed_rows,
        scans,
        qspect_series,
        qspect_dir,
        conversion_method=conversion_method,
    )
    silhouette_rows = silhouette_day0_rows(
        fixed_rows,
        scans,
        qspect_series,
        qspect_dir,
        conversion_method=conversion_method,
    )
    return {
        "variable_j": variable_rows,
        "fixed_j0": fixed_rows,
        "registered_j0": registered_rows,
        "silhouette_j0": silhouette_rows,
    }


def compact_crop_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only scalar crop fields needed for boundary plots and console output."""
    keys = (
        "day_offset",
        "qspect_label",
        "crop_top",
        "crop_bottom",
        "crop_shift_y_px",
        "profile_registration_correlation",
        "silhouette_raw_shift_y_px",
        "silhouette_shift_accepted",
    )
    return [{key: row[key] for key in keys if key in row} for row in rows]


def calculate_comparison_outputs(
    conversion_method: str = "water_scaled",
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Calculate and compact each method sequentially to limit peak memory."""
    planar_dir = planar_processing.default_planar_study_dir()
    qspect_dir = qspect_processing.default_qspect_dir()
    scans = ac.sorted_planar_scans(planar_dir)
    qspect_series = qspect_processing.load_qspect_study(qspect_dir)
    compact_methods: Dict[str, List[Dict[str, Any]]] = {}
    scalar_rows: List[Dict[str, Any]] = []

    variable_rows = ac.ct_attenuation_correction_rows(
        planar_dir=planar_dir,
        qspect_dir=qspect_dir,
        crop_strategy="individual",
        conversion_method=conversion_method,
        align_pa_to_ap=True,
    )
    compact_methods["variable_j"] = compact_crop_rows(variable_rows)
    scalar_rows.extend(scalar_comparison_rows({"variable_j": variable_rows}))
    del variable_rows
    gc.collect()

    fixed_rows = ac.ct_attenuation_correction_rows(
        planar_dir=planar_dir,
        qspect_dir=qspect_dir,
        crop_strategy="fixed_day0",
        conversion_method=conversion_method,
        align_pa_to_ap=True,
    )
    compact_methods["fixed_j0"] = compact_crop_rows(fixed_rows)
    scalar_rows.extend(scalar_comparison_rows({"fixed_j0": fixed_rows}))

    registered_rows = registered_day0_rows(
        fixed_rows,
        scans,
        qspect_series,
        qspect_dir,
        conversion_method=conversion_method,
    )
    compact_methods["registered_j0"] = compact_crop_rows(registered_rows)
    scalar_rows.extend(scalar_comparison_rows({"registered_j0": registered_rows}))
    del registered_rows
    gc.collect()

    silhouette_rows = silhouette_day0_rows(
        fixed_rows,
        scans,
        qspect_series,
        qspect_dir,
        conversion_method=conversion_method,
    )
    compact_methods["silhouette_j0"] = compact_crop_rows(silhouette_rows)
    scalar_rows.extend(scalar_comparison_rows({"silhouette_j0": silhouette_rows}))
    return compact_methods, scalar_rows


def scalar_comparison_rows(
    methods: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Flatten the three strategies into reportable QC values."""
    output = []
    for method, rows in methods.items():
        day0_top = int(rows[0]["crop_top"])
        for row in rows:
            metrics = row.get("precomputed_alignment_metrics")
            if metrics is None:
                metrics = ac.crop_alignment_metrics(row)
            outside_fraction = row.get("outside_fraction")
            if outside_fraction is None:
                outside_fraction = row["crop_excluded_counts"]["outside_fraction"]
            qspect_activity = float(row["qspect_activity_mbq"])
            if "planar_crop_height_px" in row:
                planar_crop_height_px = int(row["planar_crop_height_px"])
            else:
                planar_crop_height_px = int(row["planar_crop"].shape[0])
            output.append(
                {
                    "method": method,
                    "day": float(row["day_offset"]),
                    "label": str(row["qspect_label"]),
                    "crop_top": int(row["crop_top"]),
                    "crop_bottom_exclusive": int(row["crop_bottom"]),
                    "crop_height_cm": float(row["crop_height_cm"]),
                    "shift_from_day0_px": int(row["crop_top"]) - day0_top,
                    "shift_from_day0_cm": (
                        (int(row["crop_top"]) - day0_top)
                        * float(row["crop_height_cm"])
                        / float(planar_crop_height_px)
                    ),
                    "profile_registration_correlation": float(
                        row.get("profile_registration_correlation", np.nan)
                    ),
                    "silhouette_shift_accepted": row.get(
                        "silhouette_shift_accepted", np.nan
                    ),
                    "silhouette_raw_shift_y_px": float(
                        row.get("silhouette_raw_shift_y_px", np.nan)
                    ),
                    "silhouette_length_ratio": float(
                        row.get("silhouette_length_ratio", np.nan)
                    ),
                    "com_difference_cm": float(metrics["com_y_difference_cm"]),
                    "gm_qspect_mean_correlation": float(
                        metrics["planar_vs_qspect_mean_corr"]
                    ),
                    "ctac_qspect_mean_correlation": float(
                        metrics["ctac_vs_qspect_mean_corr"]
                    ),
                    "outside_counts_percent": 100.0 * float(outside_fraction),
                    "ctac_activity_mbq": float(row["ctac_local_activity_mbq"]),
                    "qspect_activity_mbq": qspect_activity,
                    "ctac_over_qspect": float(row["ctac_local_activity_mbq"])
                    / qspect_activity,
                }
            )
    return output


def plot_crop_bounds(
    methods: Dict[str, List[Dict[str, Any]]],
    output_path: Path = BOUNDS_FIGURE_PATH,
) -> Path:
    """Overlay all three crop boundaries on each full planar acquisition."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    variable = methods["variable_j"]
    fixed = methods["fixed_j0"]
    registered = methods["registered_j0"]
    colors = {
        "variable_j": "#d62728",
        "fixed_j0": "#1f77b4",
        "registered_j0": "#2ca02c",
        "silhouette_j0": "#9467bd",
    }
    labels = {
        "variable_j": "Variable J",
        "fixed_j0": "Fixe J0",
        "registered_j0": "J0 + profil d'activité",
        "silhouette_j0": "J0 + silhouette avec repli",
    }
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 13.0), facecolor="white", layout="constrained")
    legend_axis = axes.ravel()[-1]
    for index, axis in enumerate(axes.ravel()):
        if index >= len(variable):
            axis.axis("off")
            continue
        full = ac.tew_geometric_mean_for_scan(ac.sorted_planar_scans(planar_processing.default_planar_study_dir())[index])["image"]
        vmax = float(np.percentile(full[full > 0.0], 99.5))
        axis.imshow(full, cmap="magma", vmin=0.0, vmax=vmax, aspect="auto")
        for method, rows in methods.items():
            row = rows[index]
            axis.axhline(row["crop_top"], color=colors[method], linewidth=1.8, label=labels[method])
            axis.axhline(row["crop_bottom"] - 1, color=colors[method], linewidth=1.8)
        axis.set_title(f"{variable[index]['qspect_label']} — limites du crop")
        axis.axis("off")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    legend_axis.legend(
        handles,
        legend_labels,
        loc="center",
        frameon=False,
        fontsize=11,
    )
    fig.suptitle("Comparaison des quatre méthodes de crop", fontsize=16)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_crop_metrics(
    scalar_rows: List[Dict[str, Any]],
    output_path: Path = METRICS_FIGURE_PATH,
) -> Path:
    """Plot crop position, Q/SPECT QC metrics and final CTAC ratio."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(scalar_rows)
    styles = {
        "variable_j": ("Variable J", "#d62728", "o"),
        "fixed_j0": ("Fixe J0", "#1f77b4", "s"),
        "registered_j0": ("J0 + profil d'activité", "#2ca02c", "^"),
        "silhouette_j0": ("J0 + silhouette avec repli", "#9467bd", "P"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2), facecolor="white", layout="constrained")
    panels = (
        ("crop_top", "Ligne supérieure du crop (pixel)"),
        ("com_difference_cm", "Écart du centre de masse (cm)"),
        ("gm_qspect_mean_correlation", "Corrélation GM–Q/SPECT moyen"),
        ("ctac_over_qspect", "CTAC planaire / Q/SPECT"),
    )
    for axis, (column, ylabel) in zip(axes.ravel(), panels):
        if column == "ctac_over_qspect":
            axis.axhline(1.0, color="black", linewidth=1.0, label="Accord Q/SPECT")
        elif column == "com_difference_cm":
            axis.axhline(0.0, color="black", linewidth=1.0)
        for method, (label, color, marker) in styles.items():
            subset = frame[frame["method"] == method].sort_values("day")
            axis.plot(subset["day"], subset[column], marker=marker, color=color, linewidth=2.0, label=label)
        axis.set_xlabel("Temps après la première acquisition (jours)")
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle="--", alpha=0.3)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Évaluation des méthodes de crop", fontsize=16)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_activity_curves(
    scalar_rows: List[Dict[str, Any]],
    output_path: Path = ACTIVITY_FIGURE_PATH,
) -> Path:
    """Plot CTAC planar activity for each crop strategy against Q/SPECT."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(scalar_rows)
    styles = {
        "variable_j": ("Planaire CTAC — crop variable J", "#d62728", "o"),
        "fixed_j0": ("Planaire CTAC — crop fixe J0", "#1f77b4", "s"),
        "registered_j0": ("Planaire CTAC — J0 + profil d'activité", "#2ca02c", "^"),
        "silhouette_j0": (
            "Planaire CTAC — J0 + silhouette avec repli",
            "#9467bd",
            "P",
        ),
    }
    fig, axis = plt.subplots(figsize=(8.2, 5.4), facecolor="white", layout="constrained")
    qspect = (
        frame[frame["method"] == "fixed_j0"]
        .sort_values("day")
        .drop_duplicates(subset="day")
    )
    axis.plot(
        qspect["day"],
        qspect["qspect_activity_mbq"],
        color="black",
        marker="D",
        linewidth=2.4,
        linestyle="--",
        label="Q/SPECT",
    )
    for method, (label, color, marker) in styles.items():
        subset = frame[frame["method"] == method].sort_values("day")
        axis.plot(
            subset["day"],
            subset["ctac_activity_mbq"],
            color=color,
            marker=marker,
            linewidth=2.0,
            label=label,
        )
    axis.set_xlabel("Temps après la première acquisition (jours)")
    axis.set_ylabel("Activité estimée (MBq)")
    axis.set_title("Activité en fonction de la méthode de crop")
    axis.grid(True, linestyle="--", alpha=0.3)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, fontsize=8)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_qspect_ratios(
    scalar_rows: List[Dict[str, Any]],
    output_path: Path = RATIO_FIGURE_PATH,
) -> Path:
    """Plot each CTAC-planar activity divided by the paired Q/SPECT activity."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(scalar_rows)
    styles = {
        "variable_j": ("Crop variable J", "#d62728", "o"),
        "fixed_j0": ("Crop fixe J0", "#1f77b4", "s"),
        "registered_j0": ("J0 + profil d'activité", "#2ca02c", "^"),
        "silhouette_j0": ("J0 + silhouette avec repli", "#9467bd", "P"),
    }
    fig, axis = plt.subplots(figsize=(8.2, 5.4), facecolor="white", layout="constrained")
    axis.axhline(
        1.0,
        color="black",
        linewidth=1.4,
        linestyle="--",
        label="Accord parfait avec Q/SPECT",
    )
    for method, (label, color, marker) in styles.items():
        subset = frame[frame["method"] == method].sort_values("day")
        axis.plot(
            subset["day"],
            subset["ctac_over_qspect"],
            color=color,
            marker=marker,
            linewidth=2.0,
            label=label,
        )
    axis.set_xlabel("Temps après la première acquisition (jours)")
    axis.set_ylabel("Activité planaire CTAC / activité Q/SPECT")
    axis.set_title("Ratio avec Q/SPECT selon la méthode de crop")
    axis.grid(True, linestyle="--", alpha=0.3)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, fontsize=8)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def write_comparison_report(scalar_rows: List[Dict[str, Any]]) -> tuple[Path, Path]:
    """Save detailed values and a method summary."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(scalar_rows)
    frame.to_csv(VALUES_PATH, index=False)
    lines = [
        "Four-method planar crop comparison",
        "===================================",
        "",
        "Methods:",
        "  variable_j: existing crop recalculated independently at each time point.",
        "  fixed_j0: Day-0 crop coordinates reused without movement.",
        "  registered_j0: Day-0 height preserved; bounded vertical shift estimated only from planar multi-window profiles.",
        "  silhouette_j0: Day-0 height preserved; shift estimated from both body-envelope edges and rejected when the silhouette is incomplete.",
        "",
        "Registered-J0 processing:",
        "  AP/PA geometric mean for each available energy window; percentile normalization;",
        "  mean of normalized windows; longitudinal sum; log compression; Gaussian smoothing;",
        f"  normalized correlation over integer shifts limited to +/-{REGISTRATION_MAX_SHIFT_PX} pixels.",
        "  Q/SPECT is not used to estimate any follow-up shift.",
        "",
        "Silhouette-J0 acceptance:",
        f"  envelope length ratio {SILHOUETTE_MIN_LENGTH_RATIO:.2f}-{SILHOUETTE_MAX_LENGTH_RATIO:.2f};",
        f"  top/bottom shift disagreement <= {SILHOUETTE_MAX_EDGE_DISAGREEMENT_PX} pixels;",
        "  otherwise the applied shift is zero (fixed-J0 fallback).",
        "",
        " day | method | crop y | shift px | profile corr | COM diff cm | GM-Qmean corr | outside % | CTAC/QSP",
        "---- | ------ | ------ | -------- | ------------ | ----------- | ------------- | --------- | --------",
    ]
    for row in scalar_rows:
        profile_correlation = row["profile_registration_correlation"]
        profile_text = "n/a" if not np.isfinite(profile_correlation) else f"{profile_correlation:.3f}"
        lines.append(
            f"{row['day']:4.2f} | {row['method']:13s} | "
            f"{row['crop_top']}-{row['crop_bottom_exclusive'] - 1} | "
            f"{row['shift_from_day0_px']:+8d} | {profile_text:>12s} | "
            f"{row['com_difference_cm']:11.2f} | "
            f"{row['gm_qspect_mean_correlation']:13.3f} | "
            f"{row['outside_counts_percent']:9.3f} | {row['ctac_over_qspect']:8.3f}"
        )
    lines.extend(["", "Mean absolute COM difference and mean GM-Q/SPECT correlation:"])
    for method in ("variable_j", "fixed_j0", "registered_j0", "silhouette_j0"):
        subset = frame[frame["method"] == method]
        lines.append(
            f"  {method}: mean |COM|={subset['com_difference_cm'].abs().mean():.2f} cm, "
            f"mean correlation={subset['gm_qspect_mean_correlation'].mean():.3f}."
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return VALUES_PATH, REPORT_PATH


def main() -> None:
    methods, scalar_rows = calculate_comparison_outputs()
    bounds_path = plot_crop_bounds(methods)
    metrics_path = plot_crop_metrics(scalar_rows)
    activity_path = plot_activity_curves(scalar_rows)
    ratio_path = plot_qspect_ratios(scalar_rows)
    values_path, report_path = write_comparison_report(scalar_rows)
    print(
        "day | variable y | fixed y | profile y | profile shift | "
        "silhouette y | silhouette raw/applied/accepted"
    )
    for variable, fixed, registered, silhouette in zip(
        methods["variable_j"],
        methods["fixed_j0"],
        methods["registered_j0"],
        methods["silhouette_j0"],
    ):
        print(
            f"{registered['day_offset']:.2f} | "
            f"{variable['crop_top']}:{variable['crop_bottom']} | "
            f"{fixed['crop_top']}:{fixed['crop_bottom']} | "
            f"{registered['crop_top']}:{registered['crop_bottom']} | "
            f"{registered['crop_shift_y_px']:+d} px | "
            f"{silhouette['crop_top']}:{silhouette['crop_bottom']} | "
            f"{silhouette['silhouette_raw_shift_y_px']:+d}/"
            f"{silhouette['crop_shift_y_px']:+d}/"
            f"{silhouette['silhouette_shift_accepted']}"
        )
    print(f"Saved crop-bound figure: {bounds_path}")
    print(f"Saved metric figure: {metrics_path}")
    print(f"Saved activity figure: {activity_path}")
    print(f"Saved Q/SPECT-ratio figure: {ratio_path}")
    print(f"Saved values: {values_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
