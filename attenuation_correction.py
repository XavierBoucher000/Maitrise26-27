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
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

import correction_3DEW as c3
import dicom_loader
import planar_processing


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
    default_scan = planar_processing.default_rapid_scan_dir()
    path = Path(default_scan)
    run_attenuation_experiment(path)
