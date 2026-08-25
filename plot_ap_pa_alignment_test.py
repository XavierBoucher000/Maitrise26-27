"""Create a visual AP/PA orientation test without changing the main pipeline."""

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

import attenuation_correction
import planar_processing


OUTPUT_DIR = Path(__file__).resolve().parent / "fig" / "test"
OUTPUT_PATH = OUTPUT_DIR / "ap_pa_horizontal_alignment_test.png"


def robust_normalize(image: np.ndarray) -> np.ndarray:
    """Normalize a positive image to [0, 1] using its 99.5th percentile."""
    positive = np.clip(np.asarray(image, dtype=np.float64), 0.0, None)
    nonzero = positive[positive > 0.0]
    high = float(np.percentile(nonzero, 99.5)) if nonzero.size else 1.0
    return np.clip(positive / max(high, np.finfo(float).eps), 0.0, 1.0)


def normalized_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Return the Pearson correlation between two image arrays."""
    first = np.asarray(first, dtype=np.float64).ravel()
    second = np.asarray(second, dtype=np.float64).ravel()
    if float(np.std(first)) == 0.0 or float(np.std(second)) == 0.0:
        return np.nan
    return float(np.corrcoef(first, second)[0, 1])


def overlay_ap_pa(ap: np.ndarray, pa: np.ndarray) -> np.ndarray:
    """Show AP in red and PA in green; aligned signal therefore appears yellow."""
    ap_normalized = robust_normalize(ap)
    pa_normalized = robust_normalize(pa)
    return np.dstack(
        [ap_normalized, pa_normalized, np.zeros_like(ap_normalized)]
    )


def tew_ap_pa(scan: dict) -> Tuple[np.ndarray, np.ndarray]:
    groups = attenuation_correction.group_by_energy_and_view(scan["images"])
    ap = attenuation_correction.tew_correct_view_image(groups, "AP")
    pa = attenuation_correction.tew_correct_view_image(groups, "PA")
    return ap, pa


def plot_ap_pa_alignment_test(output_path: Path = OUTPUT_PATH) -> Path:
    """Compare raw and horizontally flipped PA alignment at Day 0 and Day 6."""
    scans = attenuation_correction.sorted_planar_scans(
        planar_processing.default_planar_study_dir()
    )
    if len(scans) < 2:
        raise ValueError("At least two planar acquisitions are required")

    selected = [(0, "Day 0"), (len(scans) - 1, "Day 6")]
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 11.5), facecolor="white")

    for row, (scan_index, label) in enumerate(selected):
        ap, pa_raw = tew_ap_pa(scans[scan_index])
        pa_aligned = np.fliplr(pa_raw)
        raw_correlation = normalized_correlation(ap, pa_raw)
        aligned_correlation = normalized_correlation(ap, pa_aligned)

        panels = [
            (robust_normalize(ap), f"{label} — AP", "magma"),
            (robust_normalize(pa_raw), f"{label} — PA brut", "magma"),
            (
                overlay_ap_pa(ap, pa_raw),
                f"Sans retournement\nr = {raw_correlation:.3f}",
                None,
            ),
            (
                overlay_ap_pa(ap, pa_aligned),
                f"PA retourné horizontalement\nr = {aligned_correlation:.3f}",
                None,
            ),
        ]
        for axis, (image, title, cmap) in zip(axes[row], panels):
            axis.imshow(image, cmap=cmap, aspect="auto", origin="upper")
            axis.set_title(title, fontsize=12)
            axis.axis("off")

    fig.suptitle(
        "Test d’orientation AP/PA — rouge = AP, vert = PA, jaune = superposition",
        fontsize=16,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    saved = plot_ap_pa_alignment_test()
    print(f"Saved AP/PA alignment test: {saved}")
    print(f"Saved AP/PA alignment test: {saved.with_suffix('.svg')}")
