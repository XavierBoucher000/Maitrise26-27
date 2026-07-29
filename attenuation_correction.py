"""
Experimental attenuation-correction helpers for planar Lu-177 whole-body images.

Important distinction:
- The 4 energy windows can support scatter correction very well.
- True attenuation correction normally needs an attenuation map, CT, body contour,
  transmission scan, or an explicit thickness model.

The functions below provide practical exploratory tools using only the planar
AP/PA images and the 4 energy windows:
1. geometric mean AP/PA, which reduces depth-dependent attenuation bias for a
   uniform slab model;
2. AP/PA asymmetry maps, which highlight likely depth/attenuation effects;
3. low-energy-scatter ratio maps, an empirical attenuation/scatter proxy;
4. optional heuristic correction from a proxy map. This is not a calibrated
   clinical correction; it is meant for method development and visualization.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

import correction_3DEW as c3
import dicom_loader
import planar_processing
import qspect_processing
from plots import view_patient_images


FIG_ROOT = Path(__file__).resolve().parent / "fig"
ATTENUATION_MODEL_DIR = FIG_ROOT / "attenuation_model"
ATTENUATION_MODEL_REPORT_PATH = ATTENUATION_MODEL_DIR / "one_patient_katt_leave_one_out.txt"
ATTENUATION_MODEL_FIGURE_PATH = ATTENUATION_MODEL_DIR / "one_patient_katt_leave_one_out.png"
CT_CORRECTION_DIR = FIG_ROOT / "ct_correction"
CT_CORRECTION_REPORT_PATH = CT_CORRECTION_DIR / "ct_attenuation_correction_report.txt"
CT_CORRECTION_FIGURE_PATH = CT_CORRECTION_DIR / "ct_attenuation_correction_activity.png"
CT_CORRECTION_MAP_FIGURE_PATH = CT_CORRECTION_DIR / "ct_attenuation_maps_day0.png"
MU_WATER_208_CM_INV = 0.154
CT_FACTOR_CLIP = (1.0, 20.0)
PLANAR_QSPECT_CROP_THRESHOLD = 0.1
CTAC_RELATIVE_METHOD_UNCERTAINTY = 0.25


def _energy_key(label: Any) -> str:
    return planar_processing.normalize_energy_label(label)


def group_by_energy_and_view(images: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    groups: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for item in images:
        view = item.get("view")
        if view not in {"AP", "PA"}:
            continue
        energy = _energy_key(item.get("energy_window") or item.get("energy_window_name"))
        groups.setdefault(energy, {})[view] = item
    return groups


def geometric_mean(ap: np.ndarray, pa: np.ndarray) -> np.ndarray:
    """AP/PA geometric mean: sqrt(AP * PA).

    For a simplified uniform attenuation model, AP and PA have opposite
    depth-dependence, so their geometric mean reduces source-depth bias.
    It does not recover absolute attenuation without body thickness and mu.
    """
    ap = np.asarray(ap, dtype=np.float64)
    pa = np.asarray(pa, dtype=np.float64)
    return np.sqrt(np.clip(ap, 0, None) * np.clip(pa, 0, None))


def ap_pa_asymmetry(ap: np.ndarray, pa: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Return 0.5 * log(AP / PA).

    Values near 0 mean AP and PA are balanced. Positive/negative values indicate
    stronger signal in one view, often reflecting depth, attenuation, positioning,
    or overlap effects.
    """
    ap = np.asarray(ap, dtype=np.float64)
    pa = np.asarray(pa, dtype=np.float64)
    return 0.5 * np.log((ap + eps) / (pa + eps))


def low_energy_scatter_ratio(groups: Dict[str, Dict[str, Dict[str, Any]]], eps: float = 1e-6) -> Optional[np.ndarray]:
    """Build a low-energy-scatter / photopeak proxy map from geometric means."""
    low = groups.get("Low Energy Scatter")
    photo = groups.get("Photopeak")
    if not low or not photo:
        return None
    if "AP" not in low or "PA" not in low or "AP" not in photo or "PA" not in photo:
        return None

    low_gm = geometric_mean(low["AP"]["image"], low["PA"]["image"])
    photo_gm = geometric_mean(photo["AP"]["image"], photo["PA"]["image"])
    return low_gm / (photo_gm + eps)


def heuristic_proxy_correction(
    image: np.ndarray,
    proxy: np.ndarray,
    strength: float = 0.25,
    clip_percentile: float = 99.0,
) -> np.ndarray:
    """Apply an empirical correction from an attenuation/scatter proxy map.

    `strength` controls how strongly high-proxy regions are boosted.
    This is intentionally conservative and normalized by the proxy median.
    """
    image = np.asarray(image, dtype=np.float64)
    proxy = np.asarray(proxy, dtype=np.float64)
    finite_proxy = proxy[np.isfinite(proxy)]
    if finite_proxy.size == 0:
        return image.copy()

    upper = np.percentile(finite_proxy, clip_percentile)
    clipped = np.clip(proxy, 0, upper)
    median = np.median(clipped[clipped > 0]) if np.any(clipped > 0) else 1.0
    normalized_proxy = clipped / max(median, 1e-6)
    correction_factor = 1.0 + strength * (normalized_proxy - 1.0)
    correction_factor = np.clip(correction_factor, 0.5, 2.0)
    return image * correction_factor


def tew_correct_view_image(groups: Dict[str, Dict[str, Dict[str, Any]]], view: str) -> np.ndarray:
    """Apply TEW correction to one detector view before AP/PA geometric mean."""
    required = ["Lower Scatter", "Photopeak", "Upper Scatter"]
    missing = [energy for energy in required if energy not in groups or view not in groups[energy]]
    if missing:
        raise ValueError(f"Missing {view} image(s) for TEW: {', '.join(missing)}")

    images_for_widths = [groups[energy][view] for energy in required]
    widths = planar_processing.window_widths(images_for_widths)
    missing_widths = [energy for energy in required if energy not in widths]
    if missing_widths:
        raise ValueError(f"Missing {view} energy width(s) for TEW: {', '.join(missing_widths)}")

    lower = np.asarray(groups["Lower Scatter"][view]["image"], dtype=np.float64)
    photopeak = np.asarray(groups["Photopeak"][view]["image"], dtype=np.float64)
    upper = np.asarray(groups["Upper Scatter"][view]["image"], dtype=np.float64)
    scatter = ((lower / widths["Lower Scatter"]) + (upper / widths["Upper Scatter"])) / 2.0
    scatter *= widths["Photopeak"]
    return np.clip(photopeak - scatter, 0.0, None)


def body_mask_from_image(image: np.ndarray, threshold_fraction: float = 0.01) -> np.ndarray:
    """Simple activity mask for the one-patient proof-of-concept model."""
    image = np.nan_to_num(np.asarray(image, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    image = np.clip(image, 0.0, None)
    maximum = float(image.max())
    if maximum <= 0.0:
        raise ValueError("Cannot build body mask from an empty image")
    return image >= threshold_fraction * maximum


def masked_tew_geometric_mean_for_scan(
    scan: Dict[str, Any],
    body_mask_threshold_fraction: float = 0.01,
) -> Dict[str, Any]:
    """Return body-masked AP/PA TEW geometric-mean counts for one planar scan.

    This follows the baseline model:
      TEW each view -> scalar dead-time correction -> sqrt(AP * PA) -> body mask sum.
    """
    images = scan["images"]
    groups = group_by_energy_and_view(images)
    ap_tew = tew_correct_view_image(groups, "AP")
    pa_tew = tew_correct_view_image(groups, "PA")

    geometric_images = planar_processing.geometric_mean_images(images)
    correction = planar_processing.apply_tew_correction(geometric_images)
    timing = planar_processing.planar_timing_from_dicom(images)
    correction = planar_processing.apply_dead_time_to_tew_correction(
        geometric_images,
        correction,
        timing.actual_frame_duration_s,
    )
    dead_time_info = correction["dead_time"] or {}
    dtcf = float(dead_time_info.get("dtcf", 1.0))

    ap_tew_dead_time = dtcf * ap_tew
    pa_tew_dead_time = dtcf * pa_tew
    geometric_tew_dead_time = geometric_mean(ap_tew_dead_time, pa_tew_dead_time)
    body_mask = body_mask_from_image(
        geometric_tew_dead_time,
        threshold_fraction=body_mask_threshold_fraction,
    )
    body_counts = float(np.sum(geometric_tew_dead_time[body_mask]))
    full_counts = float(np.sum(geometric_tew_dead_time))
    local_activity_mbq = planar_processing.counts_to_activity_mbq(
        body_counts,
        timing.local_dwell_time_s,
        planar_processing.CAMERA_SENSITIVITY_CPS_PER_MBQ,
    )

    scan_collection = {
        "patient_path": None,
        "patient_id": None,
        "scans": [scan],
    }
    day_offset = dicom_loader.scan_day_offsets(scan_collection, [scan["scan_name"]])[0]
    scan_datetime = dicom_loader.scan_datetime(scan)
    return {
        "label": scan["scan_name"],
        "datetime": scan_datetime,
        "day_offset": day_offset,
        "body_counts": body_counts,
        "full_counts": full_counts,
        "body_fraction_of_full_counts": body_counts / full_counts if full_counts > 0 else np.nan,
        "planar_body_local_activity_mbq": local_activity_mbq,
        "body_mask_pixels": int(np.count_nonzero(body_mask)),
        "body_mask_threshold_fraction": body_mask_threshold_fraction,
        "dead_time_dtcf": dtcf,
        "dead_time_loss_percent": float(dead_time_info.get("count_loss_percent", 0.0)),
        "local_dwell_time_seconds": timing.local_dwell_time_s,
        "sensitivity_cps_per_mbq": planar_processing.CAMERA_SENSITIVITY_CPS_PER_MBQ,
    }


def planar_masked_tew_decay_data(
    study_dir: Path,
    body_mask_threshold_fraction: float = 0.01,
) -> List[Dict[str, Any]]:
    patient_scans = dicom_loader.load_patient_scans(study_dir)
    rows = [
        masked_tew_geometric_mean_for_scan(scan, body_mask_threshold_fraction)
        for scan in patient_scans["scans"]
        if scan["images"]
    ]
    rows.sort(key=lambda row: (row["datetime"] is None, row["datetime"], row["label"]))
    first_datetime = next((row["datetime"] for row in rows if row["datetime"] is not None), None)
    if first_datetime is not None:
        for row in rows:
            if row["datetime"] is not None:
                row["day_offset"] = (row["datetime"] - first_datetime).total_seconds() / 86400.0
    return rows


def nearest_time_point(day_offset: float, rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    return min(rows, key=lambda row: abs(float(row["day_offset"]) - day_offset))


def pair_planar_qspect_activity(
    planar_rows: List[Dict[str, Any]],
    qspect_rows: List[Dict[str, Any]],
    max_day_difference: float = 0.6,
) -> List[Dict[str, Any]]:
    pairs = []
    for planar_row in planar_rows:
        qspect_row = nearest_time_point(float(planar_row["day_offset"]), qspect_rows)
        if qspect_row is None:
            continue
        day_difference = abs(float(qspect_row["day_offset"]) - float(planar_row["day_offset"]))
        if day_difference > max_day_difference:
            continue
        if qspect_row.get("activity_mbq") is None:
            continue
        planar_activity = float(planar_row["planar_body_local_activity_mbq"])
        qspect_activity = float(qspect_row["activity_mbq"])
        pairs.append(
            {
                **planar_row,
                "qspect_label": qspect_row["label"],
                "qspect_activity_mbq": qspect_activity,
                "day_difference": day_difference,
                "individual_k_att": qspect_activity / planar_activity if planar_activity > 0 else np.nan,
            }
        )
    return pairs


def fit_constant_k_att(planar_activity: np.ndarray, qspect_activity: np.ndarray) -> float:
    denominator = float(np.sum(planar_activity**2))
    if denominator <= 0:
        raise ValueError("Cannot fit k_att with zero planar activity")
    return float(np.sum(planar_activity * qspect_activity) / denominator)


def one_patient_katt_leave_one_out(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(pairs) < 3:
        raise ValueError("At least three paired time points are required for leave-one-out validation")

    planar = np.asarray([pair["planar_body_local_activity_mbq"] for pair in pairs], dtype=np.float64)
    qspect = np.asarray([pair["qspect_activity_mbq"] for pair in pairs], dtype=np.float64)
    days = np.asarray([pair["day_offset"] for pair in pairs], dtype=np.float64)
    labels = [str(pair["qspect_label"]) for pair in pairs]

    global_k_att = fit_constant_k_att(planar, qspect)
    global_prediction = global_k_att * planar
    loo_rows = []
    loo_predictions = []
    for index, pair in enumerate(pairs):
        train_mask = np.ones(len(pairs), dtype=bool)
        train_mask[index] = False
        k_train = fit_constant_k_att(planar[train_mask], qspect[train_mask])
        prediction = k_train * planar[index]
        error = prediction - qspect[index]
        relative_error = error / qspect[index] if qspect[index] != 0 else np.nan
        loo_predictions.append(prediction)
        loo_rows.append(
            {
                "excluded_label": labels[index],
                "day_offset": days[index],
                "k_train": k_train,
                "planar_activity_mbq": planar[index],
                "qspect_activity_mbq": qspect[index],
                "predicted_qspect_mbq": prediction,
                "error_mbq": error,
                "relative_error": relative_error,
                "relative_error_percent": 100.0 * relative_error,
            }
        )

    loo_predictions_array = np.asarray(loo_predictions, dtype=np.float64)
    errors = loo_predictions_array - qspect
    relative_errors = errors / qspect
    individual_k = np.asarray([pair["individual_k_att"] for pair in pairs], dtype=np.float64)
    return {
        "pairs": pairs,
        "days": days,
        "labels": labels,
        "planar_activity_mbq": planar,
        "qspect_activity_mbq": qspect,
        "global_k_att": global_k_att,
        "global_prediction_mbq": global_prediction,
        "leave_one_out_rows": loo_rows,
        "loo_prediction_mbq": loo_predictions_array,
        "bias_mbq": float(np.mean(errors)),
        "rmse_mbq": float(np.sqrt(np.mean(errors**2))),
        "mean_relative_error_percent": float(100.0 * np.mean(relative_errors)),
        "mean_absolute_relative_error_percent": float(100.0 * np.mean(np.abs(relative_errors))),
        "rmse_relative_percent": float(100.0 * np.sqrt(np.mean(relative_errors**2))),
        "individual_k_mean": float(np.mean(individual_k)),
        "individual_k_std": float(np.std(individual_k, ddof=1)) if len(individual_k) > 1 else 0.0,
        "individual_k_min": float(np.min(individual_k)),
        "individual_k_max": float(np.max(individual_k)),
        "individual_k_cv_percent": float(100.0 * np.std(individual_k, ddof=1) / np.mean(individual_k)) if len(individual_k) > 1 else 0.0,
    }


def format_table(headers: List[str], rows: List[List[str]]) -> List[str]:
    widths = [
        max(len(header), *(len(row[column]) for row in rows))
        for column, header in enumerate(headers)
    ]

    def format_row(values: List[str]) -> str:
        return " | ".join(value.rjust(widths[column]) for column, value in enumerate(values))

    return [format_row(headers), format_row(["-" * width for width in widths])] + [
        format_row(row) for row in rows
    ]


def write_one_patient_katt_report(result: Dict[str, Any], output_path: Path = ATTENUATION_MODEL_REPORT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pairs = result["pairs"]
    lines = [
        "One-patient constant k_att attenuation baseline",
        "================================================",
        "",
        "Model:",
        "  A_planar_corr(t) = k_att * sum_body sqrt(I_AP_TEW(t) * I_PA_TEW(t)) / (t_local * S)",
        "",
        "Implementation notes:",
        "  - AP and PA are TEW-corrected before geometric mean.",
        "  - Dead-time correction is applied as the scalar DTCF from the wide-spectrum rate.",
        "  - Body mask is a simple relative threshold on the TEW dead-time-corrected geometric-mean image.",
        "  - This is a one-patient physics baseline, not machine learning.",
        "",
        f"body_mask_threshold_fraction = {pairs[0]['body_mask_threshold_fraction']:.4f}",
        f"sensitivity_cps_per_mbq = {pairs[0]['sensitivity_cps_per_mbq']:.4f}",
        f"local_dwell_time_seconds = {pairs[0]['local_dwell_time_seconds']:.4f}",
        "",
        "Paired data:",
    ]
    pair_headers = [
        "day",
        "label",
        "planar MBq",
        "QSP MBq",
        "QSP/planar",
        "body ct",
        "full ct",
        "body/full",
        "DTCF",
    ]
    pair_rows = [
        [
            f"{pair['day_offset']:.2f}",
            pair["qspect_label"],
            f"{pair['planar_body_local_activity_mbq']:.1f}",
            f"{pair['qspect_activity_mbq']:.1f}",
            f"{pair['individual_k_att']:.3f}",
            f"{pair['body_counts']:.1f}",
            f"{pair['full_counts']:.1f}",
            f"{pair['body_fraction_of_full_counts']:.3f}",
            f"{pair['dead_time_dtcf']:.4f}",
        ]
        for pair in pairs
    ]
    lines.extend(format_table(pair_headers, pair_rows))
    lines.extend(
        [
            "",
            "Correction-factor stability:",
            f"  global least-squares k_att = {result['global_k_att']:.4f}",
            f"  individual k_att mean = {result['individual_k_mean']:.4f}",
            f"  individual k_att std = {result['individual_k_std']:.4f}",
            f"  individual k_att CV = {result['individual_k_cv_percent']:.1f}%",
            f"  individual k_att min/max = {result['individual_k_min']:.4f} / {result['individual_k_max']:.4f}",
            "",
            "Leave-one-time-point-out validation:",
        ]
    )
    loo_headers = ["excluded", "day", "k_train", "planar MBq", "QSP MBq", "pred MBq", "err MBq", "err %"]
    loo_rows = [
        [
            row["excluded_label"],
            f"{row['day_offset']:.2f}",
            f"{row['k_train']:.4f}",
            f"{row['planar_activity_mbq']:.1f}",
            f"{row['qspect_activity_mbq']:.1f}",
            f"{row['predicted_qspect_mbq']:.1f}",
            f"{row['error_mbq']:.1f}",
            f"{row['relative_error_percent']:.1f}",
        ]
        for row in result["leave_one_out_rows"]
    ]
    lines.extend(format_table(loo_headers, loo_rows))
    lines.extend(
        [
            "",
            "LOO metrics:",
            f"  bias_mbq = {result['bias_mbq']:.2f}",
            f"  rmse_mbq = {result['rmse_mbq']:.2f}",
            f"  mean_relative_error_percent = {result['mean_relative_error_percent']:.2f}",
            f"  mean_absolute_relative_error_percent = {result['mean_absolute_relative_error_percent']:.2f}",
            f"  rmse_relative_percent = {result['rmse_relative_percent']:.2f}",
            "",
            "Interpretation:",
            "  A stable attenuation-only factor would require individual QSP/planar ratios to be roughly constant.",
            "  Large time-dependent variation suggests other effects dominate: sensitivity convention, WB timing, coverage,",
            "  masking/background, low-count scatter behavior, or mismatch between whole-body planar and partial-volume Q/SPECT.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def plot_one_patient_katt_validation(
    result: Dict[str, Any],
    output_path: Path = ATTENUATION_MODEL_FIGURE_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    days = result["days"]
    qspect = result["qspect_activity_mbq"]
    planar = result["planar_activity_mbq"]
    global_prediction = result["global_prediction_mbq"]
    loo_prediction = result["loo_prediction_mbq"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="white")
    axes[0].plot(days, qspect, marker="s", linewidth=2, label="Q/SPECT")
    axes[0].plot(days, planar, marker="o", linewidth=2, label="Planar body local estimate")
    axes[0].plot(days, global_prediction, marker="^", linestyle="--", linewidth=2, label="Planar * global k_att")
    axes[0].plot(days, loo_prediction, marker="x", linestyle=":", linewidth=2, label="LOO prediction")
    axes[0].set_xlabel("Time after first acquisition (days)")
    axes[0].set_ylabel("Activity (MBq)")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)

    individual_k = [pair["individual_k_att"] for pair in result["pairs"]]
    axes[1].plot(days, individual_k, marker="o", linewidth=2, color="tab:red")
    axes[1].axhline(result["global_k_att"], linestyle="--", linewidth=1.5, color="0.25")
    axes[1].set_xlabel("Time after first acquisition (days)")
    axes[1].set_ylabel("Q/SPECT / planar local estimate")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    for spine in ["top", "right"]:
        axes[0].spines[spine].set_visible(False)
        axes[1].spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.show()
    return output_path


def run_one_patient_katt_baseline(
    planar_dir: Path = planar_processing.default_planar_study_dir(),
    qspect_dir: Path = qspect_processing.default_qspect_dir(),
    body_mask_threshold_fraction: float = 0.01,
) -> Dict[str, Any]:
    planar_rows = planar_masked_tew_decay_data(planar_dir, body_mask_threshold_fraction)
    qspect_rows = qspect_processing.qspect_decay_data(qspect_dir)
    pairs = pair_planar_qspect_activity(planar_rows, qspect_rows)
    result = one_patient_katt_leave_one_out(pairs)
    report_path = write_one_patient_katt_report(result)
    figure_path = plot_one_patient_katt_validation(result)
    print(f"Saved attenuation baseline report: {report_path}")
    print(f"Saved attenuation baseline figure: {figure_path}")
    print(f"Saved attenuation baseline figure: {figure_path.with_suffix('.svg')}")
    print(
        "One-patient k_att baseline: "
        f"k={result['global_k_att']:.4f}, "
        f"LOO RMSE={result['rmse_mbq']:.1f} MBq, "
        f"LOO MARE={result['mean_absolute_relative_error_percent']:.1f}%"
    )
    return result


def sorted_planar_scans(study_dir: Path) -> List[Dict[str, Any]]:
    patient_scans = dicom_loader.load_patient_scans(study_dir)
    scans = [scan for scan in patient_scans["scans"] if scan["images"]]
    scans.sort(key=lambda scan: (dicom_loader.scan_datetime(scan) is None, dicom_loader.scan_datetime(scan), scan["scan_name"]))
    return scans


def tew_geometric_mean_for_scan(scan: Dict[str, Any]) -> Dict[str, Any]:
    """Return AP/PA TEW geometric mean without dead-time correction."""
    images = scan["images"]
    groups = group_by_energy_and_view(images)
    ap_tew = tew_correct_view_image(groups, "AP")
    pa_tew = tew_correct_view_image(groups, "PA")
    timing = planar_processing.planar_timing_from_dicom(images)
    gm_tew = geometric_mean(ap_tew, pa_tew)
    return {
        "image": gm_tew,
        "timing": timing,
        "dead_time_applied": False,
    }


def planar_crop_matching_qspect(
    planar_image: np.ndarray,
    planar_record: Dict[str, Any],
    qspect: Dict[str, Any],
    threshold_fraction: float = PLANAR_QSPECT_CROP_THRESHOLD,
) -> Dict[str, Any]:
    """Use the current imaging crop rule from plots/view_patient_images.py.

    The crop is centered on the planar mean-profile center of mass, uses the
    Q/SPECT thresholded coronal signal length, then adds the Q/SPECT head-side
    gap to match the display logic used in the imaging test-match figures.
    """
    line = view_patient_images.qspect_length_line_on_planar(planar_image, planar_record, qspect)
    planar_measurement = view_patient_images.measure_signal_length_y(
        planar_image,
        line["pixel_spacing_mm"],
        threshold_fraction=threshold_fraction,
    )
    qspect_volume = qspect["volume"]
    coronal_index = qspect_volume.shape[1] // 2
    qspect_coronal = view_patient_images.orient_qspect_display(qspect_volume[:, coronal_index, :])
    qspect_spacing_y_mm = line["qspect_length_mm"] / qspect_coronal.shape[0]
    qspect_measurement = view_patient_images.measure_signal_length_y(
        qspect_coronal,
        qspect_spacing_y_mm,
        threshold_fraction=threshold_fraction,
    )

    matched_length_px = qspect_measurement["length_cm"] * 10.0 / line["pixel_spacing_mm"]
    matched_center_y = planar_measurement["center_of_mass_y"]
    matched_top = matched_center_y - matched_length_px / 2.0
    matched_bottom = matched_center_y + matched_length_px / 2.0
    qspect_gap_px = float(qspect_coronal.shape[0]) - qspect_measurement["length_pixels"]
    qspect_gap_cm = qspect_gap_px * qspect_spacing_y_mm / 10.0
    qspect_gap_planar_px = qspect_gap_cm * 10.0 / line["pixel_spacing_mm"]
    adjusted_top = max(0.0, matched_top - qspect_gap_planar_px)
    adjusted_bottom = min(float(planar_image.shape[0] - 1), matched_bottom)
    crop_top = int(np.floor(adjusted_top))
    crop_bottom = int(np.ceil(adjusted_bottom)) + 1
    crop = planar_image[crop_top:crop_bottom, :]

    return {
        "crop": crop,
        "crop_top": crop_top,
        "crop_bottom": crop_bottom,
        "crop_height_cm": crop.shape[0] * line["pixel_spacing_mm"] / 10.0,
        "line": line,
        "planar_measurement": planar_measurement,
        "qspect_measurement": qspect_measurement,
        "qspect_coronal": qspect_coronal,
    }


def hu_to_mu_208_cm_inv(
    ct_hu: np.ndarray,
    mu_water_208_cm_inv: float = MU_WATER_208_CM_INV,
) -> np.ndarray:
    """Approximate CT HU to linear attenuation coefficient at 208 keV.

    This is a simple water-scaled baseline for method development, not a
    scanner-specific bilinear calibration.
    """
    mu = mu_water_208_cm_inv * (1.0 + np.asarray(ct_hu, dtype=np.float64) / 1000.0)
    return np.clip(mu, 0.0, None)


def ct_attenuation_factor_map(
    ct: Dict[str, Any],
    mu_water_208_cm_inv: float = MU_WATER_208_CM_INV,
    clip_range: Tuple[float, float] = CT_FACTOR_CLIP,
) -> Dict[str, Any]:
    volume_hu = np.asarray(ct["volume"], dtype=np.float64)
    pixel_spacing = ct.get("pixel_spacing") or []
    if len(pixel_spacing) < 1:
        raise ValueError("CT PixelSpacing is required for attenuation-map integration")

    ap_spacing_cm = float(pixel_spacing[0]) / 10.0
    mu_208 = hu_to_mu_208_cm_inv(volume_hu, mu_water_208_cm_inv)
    mu_integral = np.sum(mu_208, axis=1) * ap_spacing_cm
    factor = np.exp(0.5 * mu_integral)
    factor = np.clip(factor, clip_range[0], clip_range[1])
    factor_display = view_patient_images.orient_qspect_display(factor)
    return {
        "factor_map": factor_display,
        "mu_integral": view_patient_images.orient_qspect_display(mu_integral),
        "ap_spacing_cm": ap_spacing_cm,
        "mu_water_208_cm_inv": mu_water_208_cm_inv,
        "clip_range": clip_range,
    }


def resize_map_to_image(image: np.ndarray, reference_shape: Tuple[int, int], order: int = 1) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("Only 2D maps can be resized to the planar crop")
    factors = (reference_shape[0] / image.shape[0], reference_shape[1] / image.shape[1])
    resized = zoom(image, factors, order=order)
    return resized[: reference_shape[0], : reference_shape[1]]


def ct_attenuation_correct_scan(
    scan: Dict[str, Any],
    qspect: Dict[str, Any],
    qspect_dir: Path,
    threshold_fraction: float = PLANAR_QSPECT_CROP_THRESHOLD,
) -> Dict[str, Any]:
    planar_result = tew_geometric_mean_for_scan(scan)
    planar_record = {
        "images": scan["images"],
        "correction": {"corrected_image": {"image": planar_result["image"]}},
    }
    crop_result = planar_crop_matching_qspect(
        planar_result["image"],
        planar_record,
        qspect,
        threshold_fraction=threshold_fraction,
    )
    ct = view_patient_images.load_ct_for_qspect_day(qspect_dir, qspect)
    ct_map = ct_attenuation_factor_map(ct)
    factor_resized = resize_map_to_image(ct_map["factor_map"], crop_result["crop"].shape)
    corrected_crop = crop_result["crop"] * factor_resized
    timing = planar_result["timing"]
    sensitivity = planar_processing.CAMERA_SENSITIVITY_CPS_PER_MBQ
    before_counts = float(np.sum(crop_result["crop"]))
    after_counts = float(np.sum(corrected_crop))
    qspect_activity = None if qspect.get("total_activity_bq") is None else float(qspect["total_activity_bq"]) / 1e6
    before_activity = planar_processing.counts_to_activity_mbq(before_counts, timing.local_dwell_time_s, sensitivity)
    after_activity = planar_processing.counts_to_activity_mbq(after_counts, timing.local_dwell_time_s, sensitivity)
    effective_factor = after_counts / before_counts if before_counts > 0 else np.nan
    scan_datetime = dicom_loader.scan_datetime(scan)

    return {
        "label": scan["scan_name"],
        "datetime": scan_datetime,
        "qspect_label": qspect["label"],
        "qspect_activity_mbq": qspect_activity,
        "planar_crop_counts": before_counts,
        "ctac_crop_counts": after_counts,
        "planar_crop_local_activity_mbq": before_activity,
        "ctac_local_activity_mbq": after_activity,
        "effective_ct_factor": effective_factor,
        "dead_time_applied": planar_result["dead_time_applied"],
        "frame_duration_s": timing.actual_frame_duration_s,
        "scan_time_s": timing.scan_time_s,
        "local_dwell_time_s": timing.local_dwell_time_s,
        "crop_top": crop_result["crop_top"],
        "crop_bottom": crop_result["crop_bottom"],
        "crop_height_cm": crop_result["crop_height_cm"],
        "ct_series_description": ct["series_description"],
        "ct_shape": tuple(ct["shape"]),
        "ct_pixel_spacing": ct["pixel_spacing"],
        "ct_slice_thickness": ct["slice_thickness"],
        "planar_crop": crop_result["crop"],
        "ct_factor_resized": factor_resized,
        "ctac_crop": corrected_crop,
        "qspect_coronal": crop_result["qspect_coronal"],
        "threshold_fraction": threshold_fraction,
    }


def ct_attenuation_correction_rows(
    planar_dir: Path = planar_processing.default_planar_study_dir(),
    qspect_dir: Path = qspect_processing.default_qspect_dir(),
    threshold_fraction: float = PLANAR_QSPECT_CROP_THRESHOLD,
) -> List[Dict[str, Any]]:
    planar_scans = sorted_planar_scans(planar_dir)
    qspect_series = qspect_processing.load_qspect_study(qspect_dir)
    count = min(len(planar_scans), len(qspect_series))
    if count == 0:
        raise ValueError("No paired planar/QSPECT acquisitions available")

    rows = [
        ct_attenuation_correct_scan(planar_scans[index], qspect_series[index], qspect_dir, threshold_fraction)
        for index in range(count)
    ]
    first_datetime = next((row["datetime"] for row in rows if row["datetime"] is not None), None)
    for index, row in enumerate(rows):
        if row["datetime"] is not None and first_datetime is not None:
            row["day_offset"] = (row["datetime"] - first_datetime).total_seconds() / 86400.0
        else:
            row["day_offset"] = float(index)
        qspect_activity = row["qspect_activity_mbq"]
        if qspect_activity and qspect_activity > 0:
            row["planar_over_qspect"] = row["planar_crop_local_activity_mbq"] / qspect_activity
            row["ctac_over_qspect"] = row["ctac_local_activity_mbq"] / qspect_activity
            row["error_before_percent"] = 100.0 * (row["planar_crop_local_activity_mbq"] - qspect_activity) / qspect_activity
            row["error_after_percent"] = 100.0 * (row["ctac_local_activity_mbq"] - qspect_activity) / qspect_activity
        else:
            row["planar_over_qspect"] = np.nan
            row["ctac_over_qspect"] = np.nan
            row["error_before_percent"] = np.nan
            row["error_after_percent"] = np.nan
    return rows


def write_ct_attenuation_correction_report(
    rows: List[Dict[str, Any]],
    output_path: Path = CT_CORRECTION_REPORT_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "CT-based planar attenuation-correction reference",
        "================================================",
        "",
        "Formula:",
        "  F_CT(x,y) = exp(0.5 * integral mu_208(x,y,z) dz)",
        "  GM_CTAC(x,y,t) = sqrt(AP_TEW(t) * PA_TEW(t)) * F_CT(x,y)",
        "",
        "Implementation:",
        "  - AP and PA are TEW-corrected before geometric mean.",
        "  - Dead-time correction is not applied in this CT-correction experiment.",
        "  - Planar crop uses the imaging test-match rule: threshold 0.1, planar center of mass, Q/SPECT head-side gap.",
        "  - CT series is selected with priority for ACCT [Transformed Object].",
        "  - HU to mu_208 uses a simple water-scaled approximation, not scanner-specific bilinear calibration.",
        f"  - mu_water_208_cm_inv = {MU_WATER_208_CM_INV:.4f}",
        f"  - F_CT clipped to [{CT_FACTOR_CLIP[0]:.1f}, {CT_FACTOR_CLIP[1]:.1f}]",
        f"  - Displayed CTAC uncertainty band = +/- {100.0 * CTAC_RELATIVE_METHOD_UNCERTAINTY:.0f}% method uncertainty.",
        "  - This band is exploratory; it is not a validated propagated uncertainty budget.",
        "",
    ]
    headers = [
        "day",
        "QSP label",
        "GM MBq",
        "CTAC MBq",
        "QSP MBq",
        "GM/QSP",
        "CTAC/QSP",
        "Ceff",
        "err GM %",
        "err CTAC %",
        "crop cm",
    ]
    table_rows = [
        [
            f"{row['day_offset']:.2f}",
            str(row["qspect_label"]),
            f"{row['planar_crop_local_activity_mbq']:.1f}",
            f"{row['ctac_local_activity_mbq']:.1f}",
            f"{row['qspect_activity_mbq']:.1f}",
            f"{row['planar_over_qspect']:.3f}",
            f"{row['ctac_over_qspect']:.3f}",
            f"{row['effective_ct_factor']:.3f}",
            f"{row['error_before_percent']:.1f}",
            f"{row['error_after_percent']:.1f}",
            f"{row['crop_height_cm']:.1f}",
        ]
        for row in rows
    ]
    lines.extend(format_table(headers, table_rows))
    lines.extend(
        [
            "",
            "Timing and CT metadata:",
        ]
    )
    metadata_headers = ["day", "frame s", "scan s", "local s", "crop y", "CT series", "CT shape"]
    metadata_rows = [
        [
            f"{row['day_offset']:.2f}",
            f"{row['frame_duration_s']:.3f}",
            f"{row['scan_time_s']:.3f}",
            f"{row['local_dwell_time_s']:.3f}",
            f"{row['crop_top']}-{row['crop_bottom'] - 1}",
            row["ct_series_description"],
            "x".join(str(value) for value in row["ct_shape"]),
        ]
        for row in rows
    ]
    lines.extend(format_table(metadata_headers, metadata_rows))
    lines.extend(
        [
            "",
            "Limits:",
            "  - This is a CT-oracle/reference correction for development, not a validated clinical planar activity.",
            "  - Registration is approximate: the CT attenuation map is resized onto the planar crop.",
            "  - The AP/PA integration axis is assumed from the transformed CT/QSPECT grid orientation.",
            "  - Q/SPECT is used only as a reference for evaluation, not to fit the correction factor.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def plot_ct_attenuation_correction(
    rows: List[Dict[str, Any]],
    output_path: Path = CT_CORRECTION_FIGURE_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    days = np.asarray([row["day_offset"] for row in rows], dtype=np.float64)
    gm = np.asarray([row["planar_crop_local_activity_mbq"] for row in rows], dtype=np.float64)
    ctac = np.asarray([row["ctac_local_activity_mbq"] for row in rows], dtype=np.float64)
    qspect = np.asarray([row["qspect_activity_mbq"] for row in rows], dtype=np.float64)
    ceff = np.asarray([row["effective_ct_factor"] for row in rows], dtype=np.float64)
    ctac_uncertainty = CTAC_RELATIVE_METHOD_UNCERTAINTY * ctac

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="white")
    axes[0].plot(days, gm, marker="o", linewidth=2, label="Planar GM TEW")
    axes[0].fill_between(
        days,
        ctac - ctac_uncertainty,
        ctac + ctac_uncertainty,
        color="tab:orange",
        alpha=0.18,
        linewidth=0,
        label=f"CTAC +/- {100.0 * CTAC_RELATIVE_METHOD_UNCERTAINTY:.0f}% method band",
    )
    axes[0].plot(days, ctac, marker="s", linewidth=2, color="tab:orange", label="Planar GM CTAC")
    axes[0].plot(days, qspect, marker="^", linewidth=2, label="Q/SPECT")
    axes[0].set_xlabel("Time after first acquisition (days)")
    axes[0].set_ylabel("Activity estimate (MBq)")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].plot(days, ceff, marker="o", linewidth=2, color="tab:red")
    axes[1].set_xlabel("Time after first acquisition (days)")
    axes[1].set_ylabel("Effective CT correction factor")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.show()
    return output_path


def plot_ct_attenuation_maps(
    row: Dict[str, Any],
    output_path: Path = CT_CORRECTION_MAP_FIGURE_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(13, 4), facecolor="white")
    panels = [
        ("Planar GM TEW crop", row["planar_crop"], "magma"),
        ("CT attenuation factor", row["ct_factor_resized"], "viridis"),
        ("Planar GM CTAC", row["ctac_crop"], "magma"),
        ("Q/SPECT coronal center", row["qspect_coronal"], "magma"),
    ]
    for ax, (title, image, cmap) in zip(axes, panels):
        if title == "CT attenuation factor":
            im = ax.imshow(image, cmap=cmap, aspect="auto")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        else:
            vmin, vmax = view_patient_images.display_limits(image)
            ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.show()
    return output_path


def run_ct_attenuation_correction(
    planar_dir: Path = planar_processing.default_planar_study_dir(),
    qspect_dir: Path = qspect_processing.default_qspect_dir(),
    threshold_fraction: float = PLANAR_QSPECT_CROP_THRESHOLD,
) -> List[Dict[str, Any]]:
    rows = ct_attenuation_correction_rows(planar_dir, qspect_dir, threshold_fraction)
    report_path = write_ct_attenuation_correction_report(rows)
    figure_path = plot_ct_attenuation_correction(rows)
    map_path = plot_ct_attenuation_maps(rows[0])
    print(f"Saved CT correction report: {report_path}")
    print(f"Saved CT correction figure: {figure_path}")
    print(f"Saved CT correction figure: {figure_path.with_suffix('.svg')}")
    print(f"Saved CT correction map figure: {map_path}")
    print(f"Saved CT correction map figure: {map_path.with_suffix('.svg')}")
    for row in rows:
        print(
            f"day={row['day_offset']:.2f} | "
            f"GM={row['planar_crop_local_activity_mbq']:.1f} MBq | "
            f"CTAC={row['ctac_local_activity_mbq']:.1f} MBq | "
            f"QSP={row['qspect_activity_mbq']:.1f} MBq | "
            f"Ceff={row['effective_ct_factor']:.3f}"
        )
    return rows


def build_attenuation_experiment(images: List[Dict[str, Any]], strength: float = 0.25) -> Dict[str, Any]:
    groups = group_by_energy_and_view(images)
    photo = groups.get("Photopeak")
    if not photo or "AP" not in photo or "PA" not in photo:
        raise ValueError("Photopeak AP and PA images are required")

    photo_gm = geometric_mean(photo["AP"]["image"], photo["PA"]["image"])
    asymmetry = ap_pa_asymmetry(photo["AP"]["image"], photo["PA"]["image"])
    proxy = low_energy_scatter_ratio(groups)
    proxy_corrected = None if proxy is None else heuristic_proxy_correction(photo_gm, proxy, strength=strength)

    counts = c3.extract_counts_from_images(planar_processing.geometric_mean_images(images))
    widths = planar_processing.window_widths(images)

    return {
        "groups": groups,
        "photopeak_geometric_mean": photo_gm,
        "ap_pa_asymmetry": asymmetry,
        "low_energy_proxy": proxy,
        "proxy_corrected_photopeak": proxy_corrected,
        "counts": counts,
        "widths": widths,
    }


def plot_attenuation_experiment(result: Dict[str, Any], title: str = "Attenuation experiment") -> None:
    images: List[Tuple[str, Optional[np.ndarray]]] = [
        ("Photopeak geometric mean", result["photopeak_geometric_mean"]),
        ("AP/PA asymmetry", result["ap_pa_asymmetry"]),
        ("Low-energy proxy", result["low_energy_proxy"]),
        ("Proxy-corrected photopeak", result["proxy_corrected_photopeak"]),
    ]

    fig, axes = plt.subplots(1, len(images), figsize=(4.2 * len(images), 4.2))
    for ax, (label, image) in zip(axes, images):
        if image is None:
            ax.text(0.5, 0.5, "Unavailable", ha="center", va="center")
            ax.set_title(label)
            ax.axis("off")
            continue

        finite = image[np.isfinite(image)]
        vmax = float(np.percentile(finite, 99.5)) if finite.size else 1.0
        cmap = "coolwarm" if "asymmetry" in label.lower() else "hot"
        vmin = -vmax if "asymmetry" in label.lower() else 0.0
        ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(label)
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()


def print_method_ideas() -> None:
    print("Potential attenuation-correction approaches with these data:")
    print("  1. Use AP/PA geometric mean as the baseline depth-bias reduction.")
    print("  2. Use TEW/3DEW first, then apply geometric mean on corrected photopeak images.")
    print("  3. Use AP/PA log-asymmetry maps to visualize likely attenuation/depth bias.")
    print("  4. Use Low Energy Scatter / Photopeak as an empirical attenuation/scatter proxy.")
    print("  5. Calibrate the proxy against CT/QSPECT activity if available; without calibration it is exploratory.")
    print("  6. Best option: derive a body contour/thickness or CT attenuation map, then apply a physics-based correction.")


def run_attenuation_experiment(scan_dir: Path, strength: float = 0.25) -> None:
    images = dicom_loader.load_scan_directory(scan_dir)
    result = build_attenuation_experiment(images, strength=strength)
    print_method_ideas()
    print("Geometric mean counts:")
    for label, value in result["counts"].items():
        print(f"  {label}: {value:.1f}")
    plot_attenuation_experiment(result, title=scan_dir.name)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() in {"baseline", "katt", "model"}:
        threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
        run_one_patient_katt_baseline(body_mask_threshold_fraction=threshold)
    elif len(sys.argv) > 1 and sys.argv[1].lower() in {"ct", "ctac", "correction_ct", "ct_correction"}:
        threshold = float(sys.argv[2]) if len(sys.argv) > 2 else PLANAR_QSPECT_CROP_THRESHOLD
        run_ct_attenuation_correction(threshold_fraction=threshold)
    else:
        default_scan = planar_processing.default_rapid_scan_dir()
        path = Path(default_scan)
        run_attenuation_experiment(path)
