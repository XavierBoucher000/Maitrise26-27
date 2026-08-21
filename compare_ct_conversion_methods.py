"""Compare the original and RayStation/material CT-to-mu conversions.

Outputs are written to ``fig/ct_conversion_comparison``. The activity
comparison deliberately reuses a fixed Day-0 planar crop for every time point,
so the CT conversion is the only changed correction setting.
"""

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

import attenuation_correction
import ct_attenuation_correction as ctac


OUTPUT_DIR = Path(__file__).resolve().parent / "fig" / "ct_conversion_comparison"
CALIBRATION_FIGURE = OUTPUT_DIR / "raystation_hu_to_mass_density.png"
ACTIVITY_FIGURE = OUTPUT_DIR / "ctac_fixed_crop_conversion_comparison.png"
TEMPORAL_DIFFERENCE_FIGURE = OUTPUT_DIR / "ctac_conversion_difference_over_time.png"
VALUES_FILE = OUTPUT_DIR / "ctac_conversion_comparison_values.csv"


def _save_png_svg(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_hu_to_density_calibration(output_path: Path = CALIBRATION_FIGURE) -> Path:
    """Plot the supplied RayStation piecewise-linear HU-to-density curve."""
    hu = np.linspace(ctac.RAYSTATION_HU_NODES[0], ctac.RAYSTATION_HU_NODES[-1], 2500)
    density = ctac.raystation_hu_to_mass_density_g_cm3(hu)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), facecolor="white")
    for ax in axes:
        ax.plot(hu, density, color="tab:blue", linewidth=2.2, label="Interpolation RayStation")
        ax.scatter(
            ctac.RAYSTATION_HU_NODES,
            ctac.RAYSTATION_MASS_DENSITY_G_CM3,
            color="tab:red",
            s=38,
            zorder=3,
            label="Points de calibration",
        )
        ax.set_xlabel("Unité Hounsfield (HU)")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_title("Plage complète")
    axes[0].set_ylabel("Densité massique (g/cm³)")
    axes[0].legend(frameon=False)
    axes[1].set_title("Zoom : air, poumon et tissus")
    axes[1].set_xlim(-1050, 1000)
    axes[1].set_ylim(0.0, 1.7)
    fig.suptitle("Calibration CT : HU vers densité massique", fontsize=16)
    _save_png_svg(fig, output_path)
    return output_path


def _paired_arrays(
    old_rows: List[Dict[str, Any]], new_rows: List[Dict[str, Any]]
) -> Dict[str, np.ndarray]:
    if len(old_rows) != len(new_rows):
        raise ValueError("The two CT conversion methods returned different numbers of acquisitions")
    return {
        "days": np.asarray([row["day_offset"] for row in old_rows], dtype=float),
        "qspect": np.asarray([row["qspect_activity_mbq"] for row in old_rows], dtype=float),
        "planar": np.asarray([row["planar_crop_local_activity_mbq"] for row in old_rows], dtype=float),
        "old_ctac": np.asarray([row["ctac_local_activity_mbq"] for row in old_rows], dtype=float),
        "new_ctac": np.asarray([row["ctac_local_activity_mbq"] for row in new_rows], dtype=float),
        "old_factor": np.asarray([row["effective_ct_factor"] for row in old_rows], dtype=float),
        "new_factor": np.asarray([row["effective_ct_factor"] for row in new_rows], dtype=float),
    }


def plot_fixed_crop_activity_comparison(
    values: Dict[str, np.ndarray], output_path: Path = ACTIVITY_FIGURE
) -> Path:
    """Plot fixed-crop planar CTAC while varying only HU-to-mu conversion."""
    fig, ax = plt.subplots(figsize=(8.2, 5.2), facecolor="white")
    ax.plot(values["days"], values["qspect"], "^-", linewidth=2, label="Q/SPECT")
    ax.plot(
        values["days"], values["old_ctac"], "o-", linewidth=2,
        label="CTAC — ancienne conversion HU",
    )
    ax.plot(
        values["days"], values["new_ctac"], "s-", linewidth=2,
        label="CTAC — densité RayStation + matériaux",
    )
    ax.set_title("CTAC planaire — crop fixe Day 0")
    ax.set_xlabel("Temps après la première acquisition (jours)")
    ax.set_ylabel("Activité estimée (MBq)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_png_svg(fig, output_path)
    return output_path


def plot_temporal_conversion_difference(
    values: Dict[str, np.ndarray], output_path: Path = TEMPORAL_DIFFERENCE_FIGURE
) -> Path:
    """Compare each fixed-crop CTAC estimate with Q/SPECT over time."""
    old_over_qspect = values["old_ctac"] / values["qspect"]
    new_over_qspect = values["new_ctac"] / values["qspect"]

    fig, ax = plt.subplots(figsize=(8.2, 5.2), facecolor="white")
    ax.axhline(1.0, color="black", linewidth=1.4, linestyle="--", label="Accord avec Q/SPECT")
    ax.plot(
        values["days"], old_over_qspect, "o-", color="tab:orange", linewidth=2,
        label="Ancienne conversion / Q/SPECT",
    )
    ax.plot(
        values["days"], new_over_qspect, "s-", color="tab:green", linewidth=2,
        label="RayStation + matériaux / Q/SPECT",
    )
    ax.set_title("Ratio CTAC planaire / Q/SPECT — crop fixe Day 0")
    ax.set_xlabel("Temps après la première acquisition (jours)")
    ax.set_ylabel("Ratio d'activité CTAC planaire / Q/SPECT")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_png_svg(fig, output_path)
    return output_path


def write_values(values: Dict[str, np.ndarray], output_path: Path = VALUES_FILE) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.column_stack(
        [
            values["days"], values["planar"], values["qspect"],
            values["old_ctac"], values["new_ctac"],
            values["old_factor"], values["new_factor"],
            values["old_ctac"] / values["qspect"],
            values["new_ctac"] / values["qspect"],
            100.0 * (values["new_ctac"] - values["old_ctac"]) / values["old_ctac"],
        ]
    )
    np.savetxt(
        output_path,
        matrix,
        delimiter=",",
        header=(
            "day,planar_fixed_crop_mbq,qspect_mbq,ctac_old_mbq,ctac_raystation_materials_mbq,"
            "effective_factor_old,effective_factor_raystation_materials,"
            "ctac_old_over_qspect,ctac_raystation_materials_over_qspect,new_minus_old_percent"
        ),
        comments="",
        fmt="%.8g",
    )
    return output_path


def run_comparison() -> Dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    calibration_path = plot_hu_to_density_calibration()
    old_rows = attenuation_correction.ct_attenuation_correction_rows(
        crop_strategy="fixed_day0", conversion_method="water_scaled"
    )
    new_rows = attenuation_correction.ct_attenuation_correction_rows(
        crop_strategy="fixed_day0", conversion_method="raystation_materials"
    )
    values = _paired_arrays(old_rows, new_rows)
    activity_path = plot_fixed_crop_activity_comparison(values)
    difference_path = plot_temporal_conversion_difference(values)
    values_path = write_values(values)

    print(f"Saved HU-density calibration: {calibration_path}")
    print(f"Saved fixed-crop CTAC comparison: {activity_path}")
    print(f"Saved CTAC/Q-SPECT ratios over time: {difference_path}")
    print(f"Saved numerical values: {values_path}")
    return {
        "old_rows": old_rows,
        "new_rows": new_rows,
        "values": values,
        "calibration_path": calibration_path,
        "activity_path": activity_path,
        "difference_path": difference_path,
        "values_path": values_path,
    }


if __name__ == "__main__":
    run_comparison()
