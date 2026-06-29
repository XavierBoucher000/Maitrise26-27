from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

import correction_3DEW as c3
import dicom_loader


ENERGY_ORDER = ["Lower Scatter", "Photopeak", "Upper Scatter", "Low Energy Scatter"]


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
        "scatter_image": scatter_item,
        "corrected_image": corrected_item,
    }


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
    print(f"TEW scatter estimate: {correction['scatter_counts']:.1f}")
    print(f"TEW corrected photopeak: {correction['corrected_counts']:.1f}")

    dicom_loader.plot_scan_images(
        [correction["scatter_image"], correction["corrected_image"]],
        title="Correction TEW",
    )
    plot_counts_before_after(geometric_images, correction)


def default_rapid_scan_dir() -> Path:
    return (
        Path(__file__).resolve().parent
        / "2026-05__Studies"
        / "DOE^JOHN_ANON62096_NM_2026-05-21_082240_MN.LU177.POST.TRAITEMENT-EN_WB.RAPIDE_n"
    )
