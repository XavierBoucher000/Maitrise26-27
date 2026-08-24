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

import correction_3DEW as c3
import ct_attenuation_correction as ctac
import dicom_loader
import planar_qspect_crop
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
CT_INDIVIDUAL_CROP_FIGURE_PATH = CT_CORRECTION_DIR / "ct_attenuation_correction_activity_individual_crop.png"
CT_CORRECTION_MAP_FIGURE_PATH = CT_CORRECTION_DIR / "ct_attenuation_maps_day0.png"
CT_CROP_BY_DAY_DIR = CT_CORRECTION_DIR / "crops_by_day"
CT_FIXED_CROP_BY_DAY_DIR = CT_CORRECTION_DIR / "crops_by_day_fixed_day0"
CT_PROJECTION_QC_DIR = CT_CORRECTION_DIR / "ct_projection_qc"
CT_PROJECTION_QC_REPORT_PATH = CT_PROJECTION_QC_DIR / "ct_factor_map_statistics.txt"
CT_CROP_ALIGNMENT_REPORT_PATH = CT_CROP_BY_DAY_DIR / "crop_alignment_metrics.txt"
CT_FIXED_CROP_ALIGNMENT_REPORT_PATH = CT_FIXED_CROP_BY_DAY_DIR / "crop_alignment_metrics.txt"
CT_CROP_STRATEGY_REPORT_PATH = CT_CORRECTION_DIR / "ct_crop_strategy_comparison.txt"
CT_CROP_STRATEGY_FIGURE_PATH = CT_CORRECTION_DIR / "ct_crop_strategy_comparison.png"
CTAC_PLANAR_FIXED_CROP_FIGURE_PATH = CT_CORRECTION_DIR / "ctac_planar_fixed_crop_vs_qspect.png"
CTAC_PLANAR_FIXED_CROP_RATIO_FIGURE_PATH = CT_CORRECTION_DIR / "ctac_planar_fixed_crop_qspect_ratio.png"
CT_ACTIVITY_CROP_COMPARISON_FIGURE_PATH = CT_CORRECTION_DIR / "ct_attenuation_correction_activity_crop_comparison.png"
MU_WATER_208_CM_INV = ctac.MU_WATER_208_CM_INV
CT_FACTOR_CLIP = ctac.CT_FACTOR_CLIP
PLANAR_QSPECT_CROP_THRESHOLD = planar_qspect_crop.PLANAR_QSPECT_CROP_THRESHOLD
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


def normalized_positive_image(image: np.ndarray) -> np.ndarray:
    image = np.nan_to_num(np.asarray(image, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    image = np.clip(image, 0.0, None)
    maximum = float(image.max())
    if maximum <= 0.0:
        return image
    return image / maximum


def image_y_profile(image: np.ndarray) -> np.ndarray:
    return normalized_positive_image(image).mean(axis=1)


def weighted_center_of_mass_y(image: np.ndarray) -> float:
    image = normalized_positive_image(image)
    weights = image.sum(axis=1)
    if float(weights.sum()) <= 0.0:
        return np.nan
    indices = np.arange(image.shape[0], dtype=np.float64)
    return float(np.average(indices, weights=weights))


def resized_image_correlation(reference: np.ndarray, moving: np.ndarray) -> float:
    moving_resized = ctac.resize_map_to_image(moving, reference.shape)
    reference_norm = normalized_positive_image(reference).ravel()
    moving_norm = normalized_positive_image(moving_resized).ravel()
    if float(np.std(reference_norm)) == 0.0 or float(np.std(moving_norm)) == 0.0:
        return np.nan
    return float(np.corrcoef(reference_norm, moving_norm)[0, 1])


def qspect_coronal_projections(qspect: Dict[str, Any]) -> Dict[str, np.ndarray]:
    volume = np.asarray(qspect["volume"], dtype=np.float64)
    center_index = volume.shape[1] // 2
    return {
        "center_slice": view_patient_images.orient_qspect_display(volume[:, center_index, :]),
        "mean_projection": view_patient_images.orient_qspect_display(np.mean(volume, axis=1)),
        "mip_projection": view_patient_images.orient_qspect_display(np.max(volume, axis=1)),
    }


def ct_attenuation_correct_scan(
    scan: Dict[str, Any],
    qspect: Dict[str, Any],
    qspect_dir: Path,
    threshold_fraction: float = PLANAR_QSPECT_CROP_THRESHOLD,
    fixed_crop_bounds: Optional[Tuple[int, int]] = None,
    conversion_method: str = "water_scaled",
) -> Dict[str, Any]:
    planar_result = tew_geometric_mean_for_scan(scan)
    planar_record = {
        "images": scan["images"],
        "correction": {"corrected_image": {"image": planar_result["image"]}},
    }
    crop_result = planar_qspect_crop.compute_planar_crop_for_qspect(
        planar_result["image"],
        planar_record,
        qspect,
        threshold_fraction=threshold_fraction,
        fixed_crop_bounds=fixed_crop_bounds,
    )
    ct = view_patient_images.load_ct_for_qspect_day(qspect_dir, qspect)
    ct_correction = ctac.apply_ct_attenuation_correction_to_crop(
        crop_result["crop"], ct, conversion_method=conversion_method
    )
    ct_projection_qc = ctac.ct_planar_equivalent_images(ct)
    qspect_projections = qspect_coronal_projections(qspect)
    timing = planar_result["timing"]
    sensitivity = planar_processing.CAMERA_SENSITIVITY_CPS_PER_MBQ
    before_counts = ct_correction["planar_crop_counts"]
    after_counts = ct_correction["ctac_crop_counts"]
    qspect_activity = None if qspect.get("total_activity_bq") is None else float(qspect["total_activity_bq"]) / 1e6
    before_activity = planar_processing.counts_to_activity_mbq(before_counts, timing.local_dwell_time_s, sensitivity)
    after_activity = planar_processing.counts_to_activity_mbq(after_counts, timing.local_dwell_time_s, sensitivity)
    effective_factor = ct_correction["effective_ct_factor"]
    scan_datetime = dicom_loader.scan_datetime(scan)

    print(
        f"CTAC {qspect['label']} | "
        f"C_eff={effective_factor:.3f} | "
        f"before={before_activity:.1f} MBq | "
        f"after={after_activity:.1f} MBq"
    )

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
        "dynamic_crop_top": crop_result["dynamic_crop_top"],
        "dynamic_crop_bottom": crop_result["dynamic_crop_bottom"],
        "crop_strategy": crop_result["crop_strategy"],
        "crop_height_cm": crop_result["crop_height_cm"],
        "ct_series_description": ct["series_description"],
        "ct_shape": tuple(ct["shape"]),
        "ct_pixel_spacing": ct["pixel_spacing"],
        "ct_slice_thickness": ct["slice_thickness"],
        "planar_crop": crop_result["crop"],
        "ct_factor_resized": ct_correction["ct_factor_resized"],
        "ct_factor_map": ct_correction["ct_map"]["factor_map"],
        "ct_mu_integral": ct_correction["ct_map"]["mu_integral"],
        "ct_factor_stats": ct_correction["ct_factor_stats"],
        "ct_coronal_center_hu": ct_projection_qc["coronal_center_hu"],
        "ct_mean_projection_hu": ct_projection_qc["mean_projection_hu"],
        "ctac_crop": ct_correction["ctac_crop"],
        "qspect_coronal": crop_result["qspect_coronal"],
        "qspect_coronal_center": qspect_projections["center_slice"],
        "qspect_coronal_mean_projection": qspect_projections["mean_projection"],
        "qspect_coronal_mip_projection": qspect_projections["mip_projection"],
        "threshold_fraction": threshold_fraction,
        "conversion_method": conversion_method,
        "crop_excluded_counts": crop_result["excluded_counts"],
    }


def ct_attenuation_correction_rows(
    planar_dir: Path = planar_processing.default_planar_study_dir(),
    qspect_dir: Path = qspect_processing.default_qspect_dir(),
    threshold_fraction: float = PLANAR_QSPECT_CROP_THRESHOLD,
    crop_strategy: str = "individual",
    conversion_method: str = "water_scaled",
) -> List[Dict[str, Any]]:
    planar_scans = sorted_planar_scans(planar_dir)
    qspect_series = qspect_processing.load_qspect_study(qspect_dir)
    count = min(len(planar_scans), len(qspect_series))
    if count == 0:
        raise ValueError("No paired planar/QSPECT acquisitions available")

    fixed_crop_bounds = None
    if crop_strategy == "fixed_day0":
        day0_row = ct_attenuation_correct_scan(
            planar_scans[0],
            qspect_series[0],
            qspect_dir,
            threshold_fraction,
            conversion_method=conversion_method,
        )
        fixed_crop_bounds = (int(day0_row["crop_top"]), int(day0_row["crop_bottom"]))
    elif crop_strategy != "individual":
        raise ValueError(f"Unknown crop strategy: {crop_strategy}")

    rows = [
        ct_attenuation_correct_scan(
            planar_scans[index],
            qspect_series[index],
            qspect_dir,
            threshold_fraction,
            fixed_crop_bounds=fixed_crop_bounds,
            conversion_method=conversion_method,
        )
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
        f"  - Crop strategy = {rows[0].get('crop_strategy', 'unknown')}.",
        "  - Individual crop uses the imaging test-match rule: threshold 0.1, planar center of mass, Q/SPECT head-side gap.",
        "  - Fixed_day0 crop reuses Day0 crop coordinates for every time point.",
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
    metadata_headers = ["day", "frame s", "scan s", "local s", "crop y", "dynamic y", "CT series", "CT shape"]
    metadata_rows = [
        [
            f"{row['day_offset']:.2f}",
            f"{row['frame_duration_s']:.3f}",
            f"{row['scan_time_s']:.3f}",
            f"{row['local_dwell_time_s']:.3f}",
            f"{row['crop_top']}-{row['crop_bottom'] - 1}",
            f"{row['dynamic_crop_top']}-{row['dynamic_crop_bottom'] - 1}",
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


def add_ct_hu_panel(ax: plt.Axes, image: np.ndarray, title: str, vmin: float = -200.0, vmax: float = 300.0) -> None:
    ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def plot_ct_projection_qc_by_day(
    row: Dict[str, Any],
    output_dir: Path = CT_PROJECTION_QC_DIR,
) -> Path:
    """Save CT slice/projection/factor-map QC for one time point."""
    output_dir.mkdir(parents=True, exist_ok=True)
    label = str(row["qspect_label"]).lower()
    output_path = output_dir / f"{label}_ct_projection_qc.png"

    fig, axes = plt.subplots(1, 4, figsize=(13, 4), facecolor="white")
    add_ct_hu_panel(axes[0], row["ct_coronal_center_hu"], "CT coronal center")
    add_ct_hu_panel(axes[1], row["ct_mean_projection_hu"], "CT mean projection", vmin=-1000, vmax=100)

    im_mu = axes[2].imshow(row["ct_mu_integral"], cmap="viridis", aspect="auto")
    axes[2].set_title("Integral mu_208 dz", fontsize=10)
    axes[2].axis("off")
    fig.colorbar(im_mu, ax=axes[2], fraction=0.046, pad=0.02)

    im_factor = axes[3].imshow(row["ct_factor_map"], cmap="viridis", aspect="auto")
    axes[3].set_title("F_CT factor map", fontsize=10)
    axes[3].axis("off")
    fig.colorbar(im_factor, ax=axes[3], fraction=0.046, pad=0.02)

    fig.suptitle(f"{row['qspect_label']} CT projection QC", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.show()
    return output_path


def write_ct_factor_map_statistics_report(
    rows: List[Dict[str, Any]],
    output_path: Path = CT_PROJECTION_QC_REPORT_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["day", "label", "min", "p05", "median", "mean", "p95", "max", "clip low %", "clip high %"]
    table_rows = []
    for row in rows:
        stats = row["ct_factor_stats"]
        table_rows.append(
            [
                f"{row['day_offset']:.2f}",
                str(row["qspect_label"]),
                f"{stats['min']:.3f}",
                f"{stats['p05']:.3f}",
                f"{stats['median']:.3f}",
                f"{stats['mean']:.3f}",
                f"{stats['p95']:.3f}",
                f"{stats['max']:.3f}",
                f"{100.0 * stats['clip_low_fraction']:.2f}",
                f"{100.0 * stats['clip_high_fraction']:.2f}",
            ]
        )

    lines = [
        "CT attenuation factor-map QC",
        "============================",
        "",
        "Clipping meaning:",
        "  F_CT is clipped to the configured range to avoid extreme nonphysical factors.",
        "  clip high % reports the fraction of factor-map pixels equal to the upper clip value.",
        "  If clip high % is large, the CT factor map or projection axis should be questioned.",
        "",
    ]
    lines.extend(format_table(headers, table_rows))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def plot_all_ct_projection_qc(rows: List[Dict[str, Any]]) -> List[Path]:
    return [plot_ct_projection_qc_by_day(row) for row in rows]


def crop_alignment_metrics(row: Dict[str, Any]) -> Dict[str, float]:
    planar = row["planar_crop"]
    ctac_crop = row["ctac_crop"]
    qspect_mean = row["qspect_coronal_mean_projection"]
    qspect_mip = row["qspect_coronal_mip_projection"]
    qspect_mean_resized = ctac.resize_map_to_image(qspect_mean, planar.shape)
    pixel_spacing_cm = row["crop_height_cm"] / planar.shape[0]
    planar_com_y = weighted_center_of_mass_y(planar)
    qspect_mean_com_y = weighted_center_of_mass_y(qspect_mean_resized)
    return {
        "planar_com_y_px": planar_com_y,
        "qspect_mean_com_y_px": qspect_mean_com_y,
        "com_y_difference_cm": (planar_com_y - qspect_mean_com_y) * pixel_spacing_cm,
        "planar_vs_qspect_mean_corr": resized_image_correlation(planar, qspect_mean),
        "ctac_vs_qspect_mean_corr": resized_image_correlation(ctac_crop, qspect_mean),
        "planar_vs_qspect_mip_corr": resized_image_correlation(planar, qspect_mip),
    }


def plot_crop_comparison_by_day(
    row: Dict[str, Any],
    output_dir: Path = CT_CROP_BY_DAY_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    label = str(row["qspect_label"]).lower()
    output_path = output_dir / f"{label}_planar_ctac_qspect_crop.png"

    panels = [
        ("Planar GM TEW crop", row["planar_crop"], "magma"),
        ("CT attenuation factor", row["ct_factor_resized"], "viridis"),
        ("Planar GM CTAC", row["ctac_crop"], "magma"),
        ("Q/SPECT coronal mean projection", row["qspect_coronal_mean_projection"], "magma"),
        ("Q/SPECT coronal MIP", row["qspect_coronal_mip_projection"], "magma"),
        ("Q/SPECT center slice", row["qspect_coronal_center"], "magma"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), facecolor="white")
    for ax, (title, image, cmap) in zip(axes.ravel(), panels):
        if title == "CT attenuation factor":
            im = ax.imshow(image, cmap=cmap, aspect="auto")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        else:
            vmin, vmax = view_patient_images.display_limits(image)
            ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{row['qspect_label']} crop correspondence", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.show()
    return output_path


def write_crop_alignment_report(
    rows: List[Dict[str, Any]],
    output_path: Path = CT_CROP_ALIGNMENT_REPORT_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "day",
        "label",
        "crop y",
        "crop cm",
        "COM diff cm",
        "GM-Qmean corr",
        "CTAC-Qmean corr",
        "GM-Qmip corr",
    ]
    table_rows = []
    for row in rows:
        metrics = crop_alignment_metrics(row)
        row["crop_alignment_metrics"] = metrics
        table_rows.append(
            [
                f"{row['day_offset']:.2f}",
                str(row["qspect_label"]),
                f"{row['crop_top']}-{row['crop_bottom'] - 1}",
                f"{row['crop_height_cm']:.1f}",
                f"{metrics['com_y_difference_cm']:.2f}",
                f"{metrics['planar_vs_qspect_mean_corr']:.3f}",
                f"{metrics['ctac_vs_qspect_mean_corr']:.3f}",
                f"{metrics['planar_vs_qspect_mip_corr']:.3f}",
            ]
        )

    lines = [
        "Planar crop versus Q/SPECT projection correspondence",
        "====================================================",
        "",
        "Images saved in this folder compare the planar crop against Q/SPECT coronal projections.",
        "The coronal mean projection is preferred for correspondence with planar imaging because it uses the full Q/SPECT volume.",
        "The center slice is kept only as a visual anatomical reference.",
        "",
    ]
    lines.extend(format_table(headers, table_rows))
    lines.extend(
        [
            "",
            "Metric notes:",
            "  COM diff cm = planar crop Y center of mass minus resized Q/SPECT mean-projection Y center of mass.",
            "  Correlations are normalized image correlations after resizing Q/SPECT to the planar crop shape.",
            "  These are quick quality-control metrics, not a validated registration score.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def plot_all_crop_comparisons(rows: List[Dict[str, Any]]) -> List[Path]:
    return [plot_crop_comparison_by_day(row) for row in rows]


def paired_crop_strategy_rows(
    individual_rows: List[Dict[str, Any]],
    fixed_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    fixed_by_label = {str(row["qspect_label"]): row for row in fixed_rows}
    paired = []
    for individual in individual_rows:
        fixed = fixed_by_label.get(str(individual["qspect_label"]))
        if fixed is None:
            continue
        qspect_activity = float(individual["qspect_activity_mbq"])
        paired.append(
            {
                "day_offset": float(individual["day_offset"]),
                "label": str(individual["qspect_label"]),
                "qspect_activity_mbq": qspect_activity,
                "individual_gm_mbq": float(individual["planar_crop_local_activity_mbq"]),
                "individual_ctac_mbq": float(individual["ctac_local_activity_mbq"]),
                "individual_ceff": float(individual["effective_ct_factor"]),
                "individual_crop_y": f"{individual['crop_top']}-{individual['crop_bottom'] - 1}",
                "fixed_gm_mbq": float(fixed["planar_crop_local_activity_mbq"]),
                "fixed_ctac_mbq": float(fixed["ctac_local_activity_mbq"]),
                "fixed_ceff": float(fixed["effective_ct_factor"]),
                "fixed_crop_y": f"{fixed['crop_top']}-{fixed['crop_bottom'] - 1}",
                "individual_ctac_over_qspect": float(individual["ctac_local_activity_mbq"]) / qspect_activity,
                "fixed_ctac_over_qspect": float(fixed["ctac_local_activity_mbq"]) / qspect_activity,
                "fixed_vs_individual_ctac": float(fixed["ctac_local_activity_mbq"]) / float(individual["ctac_local_activity_mbq"]),
            }
        )
    return paired


def write_crop_strategy_comparison_report(
    individual_rows: List[Dict[str, Any]],
    fixed_rows: List[Dict[str, Any]],
    output_path: Path = CT_CROP_STRATEGY_REPORT_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    paired = paired_crop_strategy_rows(individual_rows, fixed_rows)
    headers = [
        "day",
        "label",
        "QSP MBq",
        "ind CTAC",
        "fix CTAC",
        "ind/QSP",
        "fix/QSP",
        "fix/ind",
        "ind crop y",
        "fix crop y",
    ]
    table_rows = [
        [
            f"{row['day_offset']:.2f}",
            row["label"],
            f"{row['qspect_activity_mbq']:.1f}",
            f"{row['individual_ctac_mbq']:.1f}",
            f"{row['fixed_ctac_mbq']:.1f}",
            f"{row['individual_ctac_over_qspect']:.3f}",
            f"{row['fixed_ctac_over_qspect']:.3f}",
            f"{row['fixed_vs_individual_ctac']:.3f}",
            row["individual_crop_y"],
            row["fixed_crop_y"],
        ]
        for row in paired
    ]
    lines = [
        "CT attenuation correction: crop strategy comparison",
        "===================================================",
        "",
        "Compared strategies:",
        "  individual = crop recalculated independently at each time point.",
        "  fixed_day0 = Day0 crop coordinates reused for all time points.",
        "",
        "Rationale:",
        "  Individual cropping can become unstable when activity decreases and the planar image becomes noisy.",
        "  Fixed Day0 cropping tests whether a stable anatomical field of view improves temporal consistency.",
        "",
    ]
    lines.extend(format_table(headers, table_rows))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def plot_crop_strategy_comparison(
    individual_rows: List[Dict[str, Any]],
    fixed_rows: List[Dict[str, Any]],
    output_path: Path = CT_CROP_STRATEGY_FIGURE_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    paired = paired_crop_strategy_rows(individual_rows, fixed_rows)
    days = np.asarray([row["day_offset"] for row in paired], dtype=np.float64)
    qspect = np.asarray([row["qspect_activity_mbq"] for row in paired], dtype=np.float64)
    individual_ctac = np.asarray([row["individual_ctac_mbq"] for row in paired], dtype=np.float64)
    fixed_ctac = np.asarray([row["fixed_ctac_mbq"] for row in paired], dtype=np.float64)
    individual_ratio = np.asarray([row["individual_ctac_over_qspect"] for row in paired], dtype=np.float64)
    fixed_ratio = np.asarray([row["fixed_ctac_over_qspect"] for row in paired], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="white")
    axes[0].plot(days, qspect, marker="^", linewidth=2, label="Q/SPECT")
    axes[0].plot(days, individual_ctac, marker="s", linewidth=2, label="CTAC individual crop")
    axes[0].plot(days, fixed_ctac, marker="o", linewidth=2, label="CTAC fixed Day0 crop")
    axes[0].set_xlabel("Time after first acquisition (days)")
    axes[0].set_ylabel("Activity estimate (MBq)")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].plot(days, individual_ratio, marker="s", linewidth=2, label="Individual crop / Q/SPECT")
    axes[1].plot(days, fixed_ratio, marker="o", linewidth=2, label="Fixed Day0 crop / Q/SPECT")
    axes[1].set_xlabel("Time after first acquisition (days)")
    axes[1].set_ylabel("CTAC / Q/SPECT")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.show()
    return output_path


def plot_ctac_planar_fixed_crop_vs_qspect(
    fixed_rows: List[Dict[str, Any]],
    output_path: Path = CTAC_PLANAR_FIXED_CROP_FIGURE_PATH,
) -> Path:
    """Plot Q/SPECT and CTAC Planar using only the fixed Day0 crop."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    days = np.asarray([row["day_offset"] for row in fixed_rows], dtype=np.float64)
    qspect = np.asarray([row["qspect_activity_mbq"] for row in fixed_rows], dtype=np.float64)
    fixed_ctac = np.asarray([row["ctac_local_activity_mbq"] for row in fixed_rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.5, 5.0), facecolor="white")
    ax.plot(
        days,
        qspect,
        marker="^",
        linewidth=2,
        color="tab:blue",
        label="Q/SPECT",
    )
    ax.plot(
        days,
        fixed_ctac,
        marker="o",
        linewidth=2,
        color="tab:green",
        label="CTAC Planar",
    )
    ax.set_title("CTAC Planar")
    ax.set_xlabel("Temps après la première acquisition (jours)")
    ax.set_ylabel("Activité estimée (MBq)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_ctac_planar_fixed_crop_qspect_ratio(
    fixed_rows: List[Dict[str, Any]],
    output_path: Path = CTAC_PLANAR_FIXED_CROP_RATIO_FIGURE_PATH,
) -> Path:
    """Plot the fixed-Day0-crop CTAC Planar activity divided by Q/SPECT."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    days = np.asarray([row["day_offset"] for row in fixed_rows], dtype=np.float64)
    ratio = np.asarray(
        [row["ctac_local_activity_mbq"] / row["qspect_activity_mbq"] for row in fixed_rows],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.0), facecolor="white")
    ax.axhline(1.0, color="black", linewidth=1.5, label="Accord avec Q/SPECT")
    ax.plot(
        days,
        ratio,
        marker="o",
        linewidth=2,
        color="tab:green",
        label="CTAC Planar / Q/SPECT",
    )
    ax.set_title("Ratio CTAC Planar / Q/SPECT")
    ax.set_xlabel("Temps après la première acquisition (jours)")
    ax.set_ylabel("Ratio d'activité CTAC Planar / Q/SPECT")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_ct_attenuation_activity_crop_comparison(
    individual_rows: List[Dict[str, Any]],
    fixed_rows: List[Dict[str, Any]],
    output_path: Path = CT_ACTIVITY_CROP_COMPARISON_FIGURE_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    paired = paired_crop_strategy_rows(individual_rows, fixed_rows)
    days = np.asarray([row["day_offset"] for row in paired], dtype=np.float64)
    qspect = np.asarray([row["qspect_activity_mbq"] for row in paired], dtype=np.float64)
    individual_gm = np.asarray([row["individual_gm_mbq"] for row in paired], dtype=np.float64)
    individual_ctac = np.asarray([row["individual_ctac_mbq"] for row in paired], dtype=np.float64)
    fixed_ctac = np.asarray([row["fixed_ctac_mbq"] for row in paired], dtype=np.float64)
    individual_ceff = np.asarray([row["individual_ceff"] for row in paired], dtype=np.float64)
    fixed_ceff = np.asarray([row["fixed_ceff"] for row in paired], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="white")
    axes[0].plot(days, individual_gm, marker="o", linewidth=2, color="tab:blue", label="Planaire non corrigé")
    axes[0].plot(days, individual_ctac, marker="s", linewidth=2, color="tab:orange", label="Planaire + CTAC — crop individuel")
    axes[0].plot(days, fixed_ctac, marker="o", linewidth=2, color="tab:green", label="Planaire + CTAC — crop fixe Day0")
    axes[0].plot(days, qspect, marker="^", linewidth=2, color="tab:red", label="Q/SPECT")
    axes[0].set_xlabel("Time after first acquisition (days)")
    axes[0].set_ylabel("Activity estimate (MBq)")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(days, individual_ceff, marker="s", linewidth=2, color="tab:orange", label="Individual crop")
    axes[1].plot(days, fixed_ceff, marker="o", linewidth=2, color="tab:green", label="Fixed Day0 crop")
    axes[1].set_xlabel("Time after first acquisition (days)")
    axes[1].set_ylabel("Effective CT correction factor")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

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
    rows = ct_attenuation_correction_rows(planar_dir, qspect_dir, threshold_fraction, crop_strategy="individual")
    fixed_rows = ct_attenuation_correction_rows(planar_dir, qspect_dir, threshold_fraction, crop_strategy="fixed_day0")
    report_path = write_ct_attenuation_correction_report(rows)
    fixed_report_path = write_ct_attenuation_correction_report(
        fixed_rows,
        CT_CORRECTION_DIR / "ct_attenuation_correction_report_fixed_day0.txt",
    )
    ct_qc_report_path = write_ct_factor_map_statistics_report(rows)
    crop_report_path = write_crop_alignment_report(rows)
    fixed_crop_report_path = write_crop_alignment_report(fixed_rows, CT_FIXED_CROP_ALIGNMENT_REPORT_PATH)
    strategy_report_path = write_crop_strategy_comparison_report(rows, fixed_rows)
    figure_path = plot_ct_attenuation_correction(rows, CT_INDIVIDUAL_CROP_FIGURE_PATH)
    strategy_figure_path = plot_crop_strategy_comparison(rows, fixed_rows)
    ctac_planar_fixed_crop_path = plot_ctac_planar_fixed_crop_vs_qspect(fixed_rows)
    ctac_planar_fixed_crop_ratio_path = plot_ctac_planar_fixed_crop_qspect_ratio(fixed_rows)
    activity_crop_comparison_path = plot_ct_attenuation_activity_crop_comparison(rows, fixed_rows)
    map_path = plot_ct_attenuation_maps(rows[0])
    ct_qc_paths = plot_all_ct_projection_qc(rows)
    crop_paths = plot_all_crop_comparisons(rows)
    fixed_crop_paths = [plot_crop_comparison_by_day(row, CT_FIXED_CROP_BY_DAY_DIR) for row in fixed_rows]
    print(f"Saved CT correction report: {report_path}")
    print(f"Saved fixed Day0 CT correction report: {fixed_report_path}")
    print(f"Saved CT factor-map QC report: {ct_qc_report_path}")
    print(f"Saved crop alignment report: {crop_report_path}")
    print(f"Saved fixed Day0 crop alignment report: {fixed_crop_report_path}")
    print(f"Saved crop strategy report: {strategy_report_path}")
    print(f"Saved individual-crop CT correction figure: {figure_path}")
    print(f"Saved individual-crop CT correction figure: {figure_path.with_suffix('.svg')}")
    print(f"Saved crop strategy figure: {strategy_figure_path}")
    print(f"Saved crop strategy figure: {strategy_figure_path.with_suffix('.svg')}")
    print(f"Saved CTAC Planar fixed-crop figure: {ctac_planar_fixed_crop_path}")
    print(f"Saved CTAC Planar/Q-SPECT ratio figure: {ctac_planar_fixed_crop_ratio_path}")
    print(f"Saved CT activity crop comparison figure: {activity_crop_comparison_path}")
    print(f"Saved CT activity crop comparison figure: {activity_crop_comparison_path.with_suffix('.svg')}")
    print(f"Saved CT correction map figure: {map_path}")
    print(f"Saved CT correction map figure: {map_path.with_suffix('.svg')}")
    print(f"Saved CT projection QC figures in: {CT_PROJECTION_QC_DIR}")
    print(f"Saved {len(ct_qc_paths)} CT projection QC PNG files")
    print(f"Saved crop comparison figures in: {CT_CROP_BY_DAY_DIR}")
    print(f"Saved {len(crop_paths)} crop comparison PNG files")
    print(f"Saved fixed Day0 crop comparison figures in: {CT_FIXED_CROP_BY_DAY_DIR}")
    print(f"Saved {len(fixed_crop_paths)} fixed Day0 crop comparison PNG files")
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
