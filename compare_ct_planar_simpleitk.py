"""Compare the current CT-map resize with physical-grid SimpleITK resampling.

Transformation sequence
-----------------------
1. Convert CT HU to mu at 208 keV.
2. Integrate mu along the assumed AP/PA axis to obtain a 2D optical-depth map.
3. Assign the projected CT map its physical SI/LR spacing.
4. Center the CT and fixed Day-0 planar crop in one 2D physical frame.
5. Optionally blur the optical-depth map in millimetres.
6. Resample optical depth onto the planar grid with SimpleITK linear interpolation.
7. Form F_CT = exp(optical_depth / 2), apply it to the planar crop, and compare
   with the historical shape-only scipy zoom.

The whole-body planar DICOM contains detector orientation but no validated
ImagePositionPatient.  Consequently this first implementation is center-aligned,
not an absolute DICOM registration.  Q/SPECT is used only after the transform as
an evaluation reference; it is never used to choose the transform.
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import attenuation_correction as ac
import ct_attenuation_correction as ctac
import planar_processing
import qspect_processing
from plots import view_patient_images


OUTPUT_DIR = Path(__file__).resolve().parent / "fig" / "ct_correction" / "simpleitk_resampling"
MAP_FIGURE_PATH = OUTPUT_DIR / "day0_ct_planar_resampling_comparison.png"
TEMPORAL_FIGURE_PATH = OUTPUT_DIR / "ctac_resampling_comparison_over_time.png"
VALUES_PATH = OUTPUT_DIR / "ctac_resampling_comparison_values.csv"
REPORT_PATH = OUTPUT_DIR / "ctac_resampling_comparison_report.txt"


def _positive_limits(image: np.ndarray, percentile: float = 99.5) -> tuple[float, float]:
    values = np.asarray(image, dtype=np.float64)
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        return 0.0, 1.0
    return 0.0, float(np.percentile(positive, percentile))


def compare_ct_planar_resampling(
    conversion_method: str = ctac.DEFAULT_CT_CONVERSION_METHOD,
    blur_fwhm_mm: float = 0.0,
) -> List[Dict[str, Any]]:
    """Return current-resize and SimpleITK CTAC results for all five time points."""
    planar_dir = planar_processing.default_planar_study_dir()
    qspect_dir = qspect_processing.default_qspect_dir()
    scans = ac.sorted_planar_scans(planar_dir)
    qspect_series = qspect_processing.load_qspect_study(qspect_dir)
    current_rows = ac.ct_attenuation_correction_rows(
        planar_dir=planar_dir,
        qspect_dir=qspect_dir,
        crop_strategy="fixed_day0",
        conversion_method=conversion_method,
        align_pa_to_ap=True,
    )
    count = min(len(scans), len(qspect_series), len(current_rows))
    results: List[Dict[str, Any]] = []
    for index in range(count):
        scan = scans[index]
        qspect = qspect_series[index]
        current = current_rows[index]
        planar_crop = np.asarray(current["planar_crop"], dtype=np.float64)
        planar_spacing = tuple(float(value) for value in scan["images"][0]["pixel_spacing"])
        ct = view_patient_images.load_ct_for_qspect_day(qspect_dir, qspect)
        physical = ctac.apply_ct_attenuation_correction_to_crop_simpleitk(
            planar_crop,
            planar_spacing_yx_mm=planar_spacing,
            ct=ct,
            conversion_method=conversion_method,
            blur_fwhm_mm=blur_fwhm_mm,
        )

        dwell_s = float(current["local_dwell_time_s"])
        sensitivity = float(planar_processing.CAMERA_SENSITIVITY_CPS_PER_MBQ)
        physical_activity = planar_processing.counts_to_activity_mbq(
            physical["ctac_crop_counts"], dwell_s, sensitivity
        )
        current_activity = float(current["ctac_local_activity_mbq"])
        qspect_activity = float(current["qspect_activity_mbq"])
        current_factor = np.asarray(current["ct_factor_resized"], dtype=np.float64)
        physical_factor = np.asarray(physical["factor_map"], dtype=np.float64)
        correlation = float(np.corrcoef(current_factor.ravel(), physical_factor.ravel())[0, 1])
        mean_absolute_factor_difference = float(np.mean(np.abs(physical_factor - current_factor)))

        results.append(
            {
                "day": float(current["day_offset"]),
                "label": str(current["qspect_label"]),
                "crop_top": int(current["crop_top"]),
                "crop_bottom": int(current["crop_bottom"]),
                "planar_spacing_y_mm": planar_spacing[0],
                "planar_spacing_x_mm": planar_spacing[1],
                "source_spacing_y_mm": physical["source_spacing_yx_mm"][0],
                "source_spacing_x_mm": physical["source_spacing_yx_mm"][1],
                "source_extent_y_mm": physical["source_extent_yx_mm"][0],
                "source_extent_x_mm": physical["source_extent_yx_mm"][1],
                "planar_extent_y_mm": physical["planar_extent_yx_mm"][0],
                "planar_extent_x_mm": physical["planar_extent_yx_mm"][1],
                "blur_fwhm_mm": float(blur_fwhm_mm),
                "current_effective_factor": float(current["effective_ct_factor"]),
                "simpleitk_effective_factor": float(physical["effective_ct_factor"]),
                "current_ctac_mbq": current_activity,
                "simpleitk_ctac_mbq": physical_activity,
                "qspect_mbq": qspect_activity,
                "current_over_qspect": current_activity / qspect_activity,
                "simpleitk_over_qspect": physical_activity / qspect_activity,
                "simpleitk_minus_current_percent": 100.0
                * (physical_activity / current_activity - 1.0),
                "factor_map_correlation": correlation,
                "mean_absolute_factor_difference": mean_absolute_factor_difference,
                "planar_crop": planar_crop,
                "current_factor_map": current_factor,
                "simpleitk_factor_map": physical_factor,
                "current_ctac_crop": np.asarray(current["ctac_crop"], dtype=np.float64),
                "simpleitk_ctac_crop": np.asarray(physical["ctac_crop"], dtype=np.float64),
                "alignment_assumption": physical["alignment_assumption"],
                "conversion_method": conversion_method,
            }
        )
    return results


def plot_day0_map_comparison(
    row: Dict[str, Any],
    output_path: Path = MAP_FIGURE_PATH,
) -> Path:
    """Show the map-level effect of replacing resize with physical resampling."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    planar = row["planar_crop"]
    current_factor = row["current_factor_map"]
    physical_factor = row["simpleitk_factor_map"]
    difference_percent = 100.0 * (physical_factor / np.maximum(current_factor, 1e-12) - 1.0)
    current_ctac = row["current_ctac_crop"]
    physical_ctac = row["simpleitk_ctac_crop"]

    fig, axes = plt.subplots(
        2, 3, figsize=(12.0, 9.0), facecolor="white", layout="constrained"
    )
    _, planar_max = _positive_limits(planar)
    axes[0, 0].imshow(planar, cmap="magma", vmin=0.0, vmax=planar_max, aspect="auto")
    axes[0, 0].set_title("Planar TEW GM — crop fixe J0")

    factor_min = float(min(current_factor.min(), physical_factor.min()))
    factor_max = float(np.percentile(np.concatenate([current_factor.ravel(), physical_factor.ravel()]), 99.5))
    factor_image = axes[0, 1].imshow(
        current_factor, cmap="viridis", vmin=factor_min, vmax=factor_max, aspect="auto"
    )
    axes[0, 1].set_title("Carte CT — resize actuel")
    axes[0, 2].imshow(
        physical_factor, cmap="viridis", vmin=factor_min, vmax=factor_max, aspect="auto"
    )
    axes[0, 2].set_title("Carte CT — SimpleITK physique")
    fig.colorbar(
        factor_image,
        ax=[axes[0, 1], axes[0, 2]],
        fraction=0.035,
        pad=0.02,
        label="Facteur CT",
    )

    diff_limit = max(float(np.percentile(np.abs(difference_percent), 99.0)), 1.0)
    difference_image = axes[1, 0].imshow(
        difference_percent,
        cmap="coolwarm",
        vmin=-diff_limit,
        vmax=diff_limit,
        aspect="auto",
    )
    axes[1, 0].set_title("SimpleITK − resize actuel (%)")
    fig.colorbar(difference_image, ax=axes[1, 0], fraction=0.046, pad=0.04)

    _, ctac_max = _positive_limits(np.concatenate([current_ctac, physical_ctac], axis=1))
    axes[1, 1].imshow(current_ctac, cmap="magma", vmin=0.0, vmax=ctac_max, aspect="auto")
    axes[1, 1].set_title(f"CTAC resize — Ceff={row['current_effective_factor']:.3f}")
    axes[1, 2].imshow(physical_ctac, cmap="magma", vmin=0.0, vmax=ctac_max, aspect="auto")
    axes[1, 2].set_title(f"CTAC SimpleITK — Ceff={row['simpleitk_effective_factor']:.3f}")

    for axis in axes.ravel():
        axis.axis("off")
    fig.suptitle(
        "Comparaison CT→planaire : redimensionnement versus grille physique",
        fontsize=15,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_temporal_comparison(
    rows: List[Dict[str, Any]],
    output_path: Path = TEMPORAL_FIGURE_PATH,
) -> Path:
    """Compare activity and Q/SPECT ratios without fitting the transform to Q/SPECT."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    days = np.asarray([row["day"] for row in rows])
    current = np.asarray([row["current_ctac_mbq"] for row in rows])
    physical = np.asarray([row["simpleitk_ctac_mbq"] for row in rows])
    qspect = np.asarray([row["qspect_mbq"] for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.7), facecolor="white")
    axes[0].plot(days, qspect, "^-", linewidth=2.0, label="Q/SPECT — évaluation")
    axes[0].plot(days, current, "o-", linewidth=2.0, label="CTAC — resize actuel")
    axes[0].plot(days, physical, "s-", linewidth=2.0, label="CTAC — SimpleITK")
    axes[0].set_ylabel("Activité (MBq)")

    axes[1].axhline(1.0, color="black", linewidth=1.0, label="Accord Q/SPECT")
    axes[1].plot(days, current / qspect, "o-", linewidth=2.0, label="Resize/Q/SPECT")
    axes[1].plot(days, physical / qspect, "s-", linewidth=2.0, label="SimpleITK/Q/SPECT")
    axes[1].set_ylabel("Ratio activité planaire / Q/SPECT")

    for axis in axes:
        axis.set_xlabel("Temps après la première acquisition (jours)")
        axis.grid(True, linestyle="--", alpha=0.3)
        axis.legend(frameon=False, fontsize=9)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle("Effet du rééchantillonnage CT→planaire", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def write_outputs(rows: List[Dict[str, Any]]) -> tuple[Path, Path]:
    """Write numeric values and a concise method report."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scalar_rows = [
        {key: value for key, value in row.items() if not isinstance(value, np.ndarray)}
        for row in rows
    ]
    pd.DataFrame(scalar_rows).to_csv(VALUES_PATH, index=False)
    first = rows[0]
    lines = [
        "CT-to-planar SimpleITK physical-grid comparison",
        "================================================",
        "",
        "Transformations:",
        "  1. CT HU -> mu_208.",
        "  2. AP/PA integration -> 2D optical depth.",
        "  3. CT physical spacing assigned to the projection.",
        "  4. CT and fixed Day0 crop centered in a common 2D frame.",
        "  5. Optional physical Gaussian blur.",
        "  6. SimpleITK linear resampling onto the planar physical grid.",
        "  7. F_CT = exp(optical_depth / 2), then pixel-wise CTAC.",
        "",
        "Important limitations:",
        "  - Planar ImagePositionPatient is empty, so absolute DICOM-origin registration is unavailable.",
        "  - This version uses center alignment and identity direction after display orientation.",
        "  - Q/SPECT is used only for evaluation, never to tune translation, scale, or blur.",
        f"  - HU conversion = {first['conversion_method']}.",
        f"  - Blur FWHM = {first['blur_fwhm_mm']:.2f} mm.",
        "",
        "Physical grids (Day0):",
        f"  CT projected spacing y,x = {first['source_spacing_y_mm']:.4f}, {first['source_spacing_x_mm']:.4f} mm.",
        f"  Planar spacing y,x = {first['planar_spacing_y_mm']:.4f}, {first['planar_spacing_x_mm']:.4f} mm.",
        f"  CT projected extent y,x = {first['source_extent_y_mm']:.1f}, {first['source_extent_x_mm']:.1f} mm.",
        f"  Planar crop extent y,x = {first['planar_extent_y_mm']:.1f}, {first['planar_extent_x_mm']:.1f} mm.",
        "",
        " day | resize MBq | SimpleITK MBq | Q/SPECT MBq | resize/QSP | SimpleITK/QSP | SimpleITK-resize % | factor corr",
        "---- | ---------- | ------------- | ----------- | ---------- | ------------- | ----------------- | -----------",
    ]
    for row in rows:
        lines.append(
            f"{row['day']:4.2f} | {row['current_ctac_mbq']:10.1f} | "
            f"{row['simpleitk_ctac_mbq']:13.1f} | {row['qspect_mbq']:11.1f} | "
            f"{row['current_over_qspect']:10.3f} | {row['simpleitk_over_qspect']:13.3f} | "
            f"{row['simpleitk_minus_current_percent']:17.2f} | {row['factor_map_correlation']:11.3f}"
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return VALUES_PATH, REPORT_PATH


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conversion-method",
        choices=("water_scaled", "raystation_materials", "catphan_t2_110kvp"),
        default=ctac.DEFAULT_CT_CONVERSION_METHOD,
    )
    parser.add_argument(
        "--blur-fwhm-mm",
        type=float,
        default=0.0,
        help="Exploratory planar-resolution blur; zero keeps geometry-only comparison.",
    )
    args = parser.parse_args()
    rows = compare_ct_planar_resampling(
        conversion_method=args.conversion_method,
        blur_fwhm_mm=args.blur_fwhm_mm,
    )
    map_path = plot_day0_map_comparison(rows[0])
    temporal_path = plot_temporal_comparison(rows)
    values_path, report_path = write_outputs(rows)
    print("day | resize/QSP | SimpleITK/QSP | SimpleITK-resize %")
    for row in rows:
        print(
            f"{row['day']:.2f} | {row['current_over_qspect']:.3f} | "
            f"{row['simpleitk_over_qspect']:.3f} | "
            f"{row['simpleitk_minus_current_percent']:+.2f}%"
        )
    print(f"Saved map comparison: {map_path}")
    print(f"Saved temporal comparison: {temporal_path}")
    print(f"Saved values: {values_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
