from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

import correction_3DEW as c3
import dead_time
import dicom_loader


ENERGY_ORDER = ["Lower Scatter", "Photopeak", "Upper Scatter", "Low Energy Scatter"]
CAMERA_SENSITIVITY_CPS_PER_MBQ = 9.36
APPLY_DEAD_TIME_CORRECTION = True
DEAD_TIME_TAU_US = 0.632
DEAD_TIME_WIDE_WINDOW_ENERGIES = ["Low Energy Scatter", "Lower Scatter", "Photopeak", "Upper Scatter"]


def normalize_energy_label(label: Any) -> str:
    if not isinstance(label, str):
        return "Unknown Window"
    label_lower = label.lower()
    if "lower" in label_lower:
        return "Lower Scatter"
    if "upper" in label_lower:
        return "Upper Scatter"
    if "photo" in label_lower or "lutetium" in label_lower or "177" in label_lower or "main" in label_lower:
        return "Photopeak"
    if "low energy" in label_lower:
        return "Low Energy Scatter"
    return label


def group_ap_pa_by_energy(images: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    groups: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for item in images:
        view = item.get("view")
        if view not in {"AP", "PA"}:
            continue
        energy = normalize_energy_label(item.get("energy_window") or item.get("energy_window_name"))
        groups.setdefault(energy, {})[view] = item
    return groups


def geometric_mean_images(images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create one geometric-mean image per energy window from AP and PA views."""
    means = []
    for energy, views in group_ap_pa_by_energy(images).items():
        ap = views.get("AP")
        pa = views.get("PA")
        if ap is None or pa is None:
            continue

        ap_image = np.asarray(ap["image"], dtype=np.float64)
        pa_image = np.asarray(pa["image"], dtype=np.float64)
        mean_image = np.sqrt(np.clip(ap_image, 0, None) * np.clip(pa_image, 0, None))

        item = ap.copy()
        item["view"] = "GM"
        item["energy_window"] = energy
        item["energy_window_name"] = f"Geometric Mean {energy}"
        item["image"] = mean_image
        item["pixel_sum"] = float(mean_image.sum())
        item["pixel_mean"] = float(mean_image.mean())
        item["pixel_dtype"] = str(mean_image.dtype)
        item["source_views"] = ("AP", "PA")
        means.append(item)

    return sorted(means, key=lambda item: ENERGY_ORDER.index(item["energy_window"]) if item["energy_window"] in ENERGY_ORDER else len(ENERGY_ORDER))


def window_widths(images: List[Dict[str, Any]]) -> Dict[str, float]:
    widths: Dict[str, float] = {}
    for item in images:
        energy = normalize_energy_label(item.get("energy_window") or item.get("energy_window_name"))
        lower = item.get("energy_window_lower_limit")
        upper = item.get("energy_window_upper_limit")
        if lower is None or upper is None:
            continue
        width = float(upper) - float(lower)
        if width > 0:
            widths[energy] = width
    return widths


def apply_tew_correction(geometric_images: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply TEW correction to geometric-mean Photopeak using Lower/Upper Scatter."""
    by_energy = {item["energy_window"]: item for item in geometric_images}
    required = ["Lower Scatter", "Photopeak", "Upper Scatter"]
    missing = [energy for energy in required if energy not in by_energy]
    if missing:
        raise ValueError(f"Missing required energy window(s) for TEW: {', '.join(missing)}")

    widths = window_widths(geometric_images)
    missing_widths = [energy for energy in required if energy not in widths]
    if missing_widths:
        raise ValueError(f"Missing energy window width(s) for TEW: {', '.join(missing_widths)}")

    lower_image = np.asarray(by_energy["Lower Scatter"]["image"], dtype=np.float64)
    photopeak_image = np.asarray(by_energy["Photopeak"]["image"], dtype=np.float64)
    upper_image = np.asarray(by_energy["Upper Scatter"]["image"], dtype=np.float64)

    w_lower = widths["Lower Scatter"]
    w_photopeak = widths["Photopeak"]
    w_upper = widths["Upper Scatter"]
    scatter_image = ((lower_image / w_lower) + (upper_image / w_upper)) / 2.0 * w_photopeak
    corrected_image = photopeak_image - scatter_image
    corrected_clipped = np.clip(corrected_image, 0, None)

    counts = c3.extract_counts_from_images(geometric_images)
    scatter_counts, corrected_counts = c3.tew_scatter_estimate(
        counts.get("Lower Scatter", 0.0),
        counts.get("Photopeak", 0.0),
        counts.get("Upper Scatter", 0.0),
        w_lower,
        w_photopeak,
        w_upper,
    )

    corrected_item = by_energy["Photopeak"].copy()
    corrected_item["energy_window"] = "TEW Corrected Photopeak"
    corrected_item["energy_window_name"] = "TEW Corrected Photopeak"
    corrected_item["image"] = corrected_clipped
    corrected_item["pixel_sum"] = float(corrected_clipped.sum())
    corrected_item["pixel_mean"] = float(corrected_clipped.mean())
    corrected_item["pixel_dtype"] = str(corrected_clipped.dtype)

    scatter_item = by_energy["Photopeak"].copy()
    scatter_item["energy_window"] = "TEW Scatter Estimate"
    scatter_item["energy_window_name"] = "TEW Scatter Estimate"
    scatter_item["image"] = scatter_image
    scatter_item["pixel_sum"] = float(scatter_image.sum())
    scatter_item["pixel_mean"] = float(scatter_image.mean())
    scatter_item["pixel_dtype"] = str(scatter_image.dtype)

    return {
        "counts": counts,
        "widths": widths,
        "scatter_counts": scatter_counts,
        "corrected_counts": corrected_counts,
        "corrected_counts_before_dead_time": corrected_counts,
        "scatter_image": scatter_item,
        "corrected_image": corrected_item,
        "dead_time": None,
    }


def wide_spectrum_counts(geometric_images: List[Dict[str, Any]]) -> float:
    counts = c3.extract_counts_from_images(geometric_images)
    available = [
        counts[energy]
        for energy in DEAD_TIME_WIDE_WINDOW_ENERGIES
        if energy in counts
    ]
    if not available:
        raise ValueError("No energy windows available to estimate wide-spectrum count rate")
    return float(sum(available))


def apply_dead_time_to_tew_correction(
    geometric_images: List[Dict[str, Any]],
    correction: Dict[str, Any],
    duration_seconds: float,
    tau_us: float = DEAD_TIME_TAU_US,
) -> Dict[str, Any]:
    """Apply Frezza paralyzable dead-time correction to TEW-corrected primary counts."""
    if not APPLY_DEAD_TIME_CORRECTION:
        return correction

    rpo_cps = float(correction["corrected_counts_before_dead_time"]) / duration_seconds
    rwo_counts = wide_spectrum_counts(geometric_images)
    rwo_cps = rwo_counts / duration_seconds
    result = dead_time.correct_dead_time(
        rpo_cps=rpo_cps,
        rwo_cps=rwo_cps,
        tau_us=tau_us,
        calibration_factor_cps_per_mbq=CAMERA_SENSITIVITY_CPS_PER_MBQ,
        image_primary=correction["corrected_image"]["image"],
    )

    corrected_image = correction["corrected_image"].copy()
    corrected_image["image"] = result.corrected_image
    corrected_image["pixel_sum"] = float(np.sum(result.corrected_image))
    corrected_image["pixel_mean"] = float(np.mean(result.corrected_image))
    corrected_image["energy_window_name"] = "TEW + dead-time corrected photopeak"

    updated = correction.copy()
    updated["corrected_counts"] = result.rpt_corrected_cps * duration_seconds
    updated["corrected_image"] = corrected_image
    updated["dead_time"] = {
        "dtcf": result.dtcf,
        "rpo_observed_cps": result.rpo_observed_cps,
        "rpt_corrected_cps": result.rpt_corrected_cps,
        "rwo_counts": rwo_counts,
        "rwo_cps": rwo_cps,
        "tau_us": tau_us,
        "count_loss_fraction": result.count_loss_fraction,
        "count_loss_percent": result.count_loss_percent,
        "warnings": result.warnings,
    }
    return updated


def plot_counts_before_after(geometric_images: List[Dict[str, Any]], correction: Dict[str, Any]) -> None:
    counts = c3.extract_counts_from_images(geometric_images)
    labels = [energy for energy in ENERGY_ORDER if energy in counts]
    labels.extend(sorted(energy for energy in counts if energy not in labels))

    before_values = [counts[label] for label in labels]
    after_labels = ["Photopeak", "TEW Scatter", "TEW Corrected"]
    after_values = [
        counts.get("Photopeak", 0.0),
        correction["scatter_counts"],
        correction["corrected_counts"],
    ]

    fig, axes = plt.subplots(ncols=2, figsize=(12, 4.5))
    axes[0].bar(labels, before_values)
    axes[0].set_title("Moyenne géométrique par fenêtre")
    axes[0].set_ylabel("Counts")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(after_labels, after_values, color=["tab:blue", "tab:orange", "tab:green"])
    axes[1].set_title("Correction TEW sur Photopeak")
    axes[1].set_ylabel("Counts")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    plt.show()


def acquisition_duration_seconds(images: List[Dict[str, Any]]) -> float:
    for item in images:
        duration_ms = item.get("actual_frame_duration_ms")
        if duration_ms is None:
            continue
        try:
            return float(duration_ms) / 1000.0
        except (TypeError, ValueError):
            continue
    raise ValueError("ActualFrameDuration is required to convert planar counts to activity")


def counts_to_activity_mbq(
    counts: float,
    duration_seconds: float,
    sensitivity_cps_per_mbq: float = CAMERA_SENSITIVITY_CPS_PER_MBQ,
) -> float:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if sensitivity_cps_per_mbq <= 0:
        raise ValueError("sensitivity_cps_per_mbq must be positive")
    return float(counts) / (float(sensitivity_cps_per_mbq) * float(duration_seconds))


def run_planar_workflow(scan_dir: Path) -> None:
    images = dicom_loader.load_scan_directory(scan_dir)
    print(f"Loaded {len(images)} image frame(s) from {scan_dir.name}")

    dicom_loader.plot_scan_images(images, title=f"{scan_dir.name} - AP/PA")

    geometric_images = geometric_mean_images(images)
    print("Geometric mean counts:")
    for item in geometric_images:
        print(f"  {item['energy_window']}: {item['pixel_sum']:.1f}")
    dicom_loader.plot_scan_images(geometric_images, title="Moyenne géométrique AP/PA")

    correction = apply_tew_correction(geometric_images)
    duration_seconds = acquisition_duration_seconds(images)
    correction = apply_dead_time_to_tew_correction(geometric_images, correction, duration_seconds)
    print(f"TEW scatter estimate: {correction['scatter_counts']:.1f}")
    print(f"TEW corrected photopeak before dead time: {correction['corrected_counts_before_dead_time']:.1f}")
    print(f"TEW corrected photopeak after dead time: {correction['corrected_counts']:.1f}")
    if correction["dead_time"] is not None:
        dead_time_info = correction["dead_time"]
        print(
            f"Dead-time correction: DTCF={dead_time_info['dtcf']:.4f}, "
            f"loss={dead_time_info['count_loss_percent']:.2f}%, "
            f"RWo={dead_time_info['rwo_cps']:.1f} cps"
        )

    dicom_loader.plot_scan_images(
        [correction["scatter_image"], correction["corrected_image"]],
        title="Correction TEW",
    )
    plot_counts_before_after(geometric_images, correction)


def run_planar_study_workflow(study_dir: Path) -> None:
    scan_dirs = dicom_loader.find_scan_dirs(study_dir)
    if not scan_dirs:
        raise ValueError(f"No planar DICOM scan folders found in {study_dir}")

    patient_scans = dicom_loader.load_patient_scans(study_dir)
    print(f"Loaded {len(patient_scans['scans'])} planar acquisition(s) from {study_dir.name}")

    for scan in patient_scans["scans"]:
        images = scan["images"]
        if not images:
            continue
        scan_title = scan["scan_name"]
        print(f"\nPlanar acquisition: {scan_title}")
        print(f"  frames: {len(images)}")
        dicom_loader.plot_scan_images(images, title=f"{scan_title} - AP/PA")

        geometric_images = geometric_mean_images(images)
        print("  Geometric mean counts:")
        for item in geometric_images:
            print(f"    {item['energy_window']}: {item['pixel_sum']:.1f}")

        if geometric_images:
            dicom_loader.plot_scan_images(geometric_images, title=f"{scan_title} - moyenne géométrique")

        try:
            correction = apply_tew_correction(geometric_images)
            duration_seconds = acquisition_duration_seconds(images)
            correction = apply_dead_time_to_tew_correction(geometric_images, correction, duration_seconds)
        except ValueError as exc:
            print(f"  TEW correction skipped: {exc}")
            continue

        print(f"  TEW scatter estimate: {correction['scatter_counts']:.1f}")
        print(f"  TEW corrected photopeak before dead time: {correction['corrected_counts_before_dead_time']:.1f}")
        print(f"  TEW corrected photopeak after dead time: {correction['corrected_counts']:.1f}")
        if correction["dead_time"] is not None:
            dead_time_info = correction["dead_time"]
            print(
                f"  Dead-time correction: DTCF={dead_time_info['dtcf']:.4f}, "
                f"loss={dead_time_info['count_loss_percent']:.2f}%, "
                f"RWo={dead_time_info['rwo_cps']:.1f} cps"
            )
        dicom_loader.plot_scan_images(
            [correction["scatter_image"], correction["corrected_image"]],
            title=f"{scan_title} - correction TEW",
        )
        plot_counts_before_after(geometric_images, correction)

    if patient_scans["scans"]:
        dicom_loader.plot_summary_grid(patient_scans)
        dicom_loader.plot_decay_curve(patient_scans)


def planar_tew_decay_data(study_dir: Path) -> List[Dict[str, Any]]:
    """Return per-acquisition planar TEW data without plotting.

    This is the numeric API used by comparison scripts. It keeps planar loading,
    geometric mean, and TEW correction inside this module.
    """
    patient_scans = dicom_loader.load_patient_scans(study_dir)
    rows = []
    for scan in patient_scans["scans"]:
        images = scan["images"]
        if not images:
            continue

        geometric_images = geometric_mean_images(images)
        correction = apply_tew_correction(geometric_images)
        duration_seconds = acquisition_duration_seconds(images)
        correction = apply_dead_time_to_tew_correction(geometric_images, correction, duration_seconds)
        planar_activity_mbq = counts_to_activity_mbq(correction["corrected_counts"], duration_seconds)
        dead_time_info = correction["dead_time"] or {}
        scan_collection = {
            "patient_path": patient_scans["patient_path"],
            "patient_id": patient_scans["patient_id"],
            "scans": [scan],
        }
        day_offset = dicom_loader.scan_day_offsets(scan_collection, [scan["scan_name"]])[0]
        scan_datetime = dicom_loader.scan_datetime(scan)
        rows.append(
            {
                "label": scan["scan_name"],
                "datetime": scan_datetime,
                "day_offset": day_offset,
                "photopeak_counts": correction["counts"].get("Photopeak", 0.0),
                "scatter_counts": correction["scatter_counts"],
                "tew_corrected_counts": correction["corrected_counts"],
                "tew_corrected_counts_before_dead_time": correction["corrected_counts_before_dead_time"],
                "tew_corrected_cps": correction["corrected_counts"] / duration_seconds,
                "tew_corrected_cps_before_dead_time": correction["corrected_counts_before_dead_time"] / duration_seconds,
                "duration_seconds": duration_seconds,
                "planar_activity_mbq": planar_activity_mbq,
                "sensitivity_cps_per_mbq": CAMERA_SENSITIVITY_CPS_PER_MBQ,
                "dead_time_applied": correction["dead_time"] is not None,
                "dead_time_tau_us": dead_time_info.get("tau_us"),
                "dead_time_dtcf": dead_time_info.get("dtcf", 1.0),
                "dead_time_loss_percent": dead_time_info.get("count_loss_percent", 0.0),
                "wide_spectrum_counts": dead_time_info.get("rwo_counts"),
                "wide_spectrum_cps": dead_time_info.get("rwo_cps"),
            }
        )

    rows.sort(key=lambda row: (row["datetime"] is None, row["datetime"], row["label"]))
    if rows:
        first_datetime = next((row["datetime"] for row in rows if row["datetime"] is not None), None)
        if first_datetime is not None:
            for row in rows:
                if row["datetime"] is not None:
                    row["day_offset"] = (row["datetime"] - first_datetime).total_seconds() / 86400.0
    return rows


def default_planar_study_dir() -> Path:
    return (
        Path(__file__).resolve().parent
        / "data"
        / "2026-05_studies"
        / "2026-05__Studies_WBP"
    )


def default_rapid_scan_dir() -> Path:
    return (
        Path(__file__).resolve().parent
        / "data"
        / "2026-05_studies"
        / "2026-05__Studies_WBP"
        / "DOE^JOHN_ANON64926_NM_2026-05-21_082240_MN.LU177.POST.TRAITEMENT-EN_WB.RAPIDE_n8__00000"
    )
