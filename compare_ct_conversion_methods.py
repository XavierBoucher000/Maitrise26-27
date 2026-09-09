"""Compare three development CT HU-to-mu conversions.

Outputs are written to ``fig/HU_conversion_comparison``. The activity
comparison deliberately reuses the profile-matched planar crop for every conversion,
so the CT conversion is the only changed correction setting. The direct T2
Catphan curve is provisional because its QC reconstruction (B41s, 8 mm) does
not exactly match the patient ACCT reconstruction (B08s, 5 mm).
"""

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

import attenuation_correction
import ct_attenuation_correction as ctac


OUTPUT_DIR = Path(__file__).resolve().parent / "fig" / "HU_conversion_comparison"
ALIGN_PA_TO_AP = True
CALIBRATION_FIGURE = OUTPUT_DIR / "raystation_hu_to_mass_density.png"
RAYSTATION_SINGLE_CURVE_FIGURE = (
    OUTPUT_DIR / "raystation_hu_to_mass_density_single_curve.png"
)
PROJECT_CALIBRATION_FIGURE = OUTPUT_DIR / "hu_to_mass_density_calibration_used.png"
CATPHAN_CALIBRATION_FIGURE = OUTPUT_DIR / "catphan_t2_110kvp_hu_to_mu2084.png"
CATPHAN_VALUES_FILE = OUTPUT_DIR / "catphan_t2_110kvp_hu_to_mu2084_values.csv"
ACTIVITY_FIGURE = OUTPUT_DIR / "ctac_profile_crop_conversion_comparison.png"
TEMPORAL_DIFFERENCE_FIGURE = OUTPUT_DIR / "ctac_conversion_difference_over_time.png"
VALUES_FILE = OUTPUT_DIR / "ctac_conversion_comparison_values.csv"


def _save_png_svg(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
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


def plot_raystation_hu_to_density_single_curve(
    output_path: Path = RAYSTATION_SINGLE_CURVE_FIGURE,
) -> Path:
    """Plot the RayStation HU-density calibration in one full-range panel."""
    hu_nodes = np.asarray(ctac.RAYSTATION_HU_NODES, dtype=float)
    density_nodes = np.asarray(ctac.RAYSTATION_MASS_DENSITY_G_CM3, dtype=float)
    hu_curve = np.linspace(hu_nodes[0], hu_nodes[-1], 3000)
    density_curve = ctac.raystation_hu_to_mass_density_g_cm3(hu_curve)

    fig, ax = plt.subplots(figsize=(8.6, 5.6), facecolor="white")
    ax.plot(
        hu_curve,
        density_curve,
        color="#1f77b4",
        linewidth=2.4,
        label="Interpolation linéaire par segments",
        zorder=2,
    )
    ax.scatter(
        hu_nodes,
        density_nodes,
        color="#d62728",
        edgecolor="white",
        linewidth=0.7,
        s=58,
        label="Points de calibration RayStation",
        zorder=3,
    )
    ax.set_title("Calibration RayStation : HU vers densité massique")
    ax.set_xlabel("Unité Hounsfield (HU)")
    ax.set_ylabel("Densité massique (g/cm³)")
    ax.set_xlim(hu_nodes[0] - 120.0, hu_nodes[-1] + 120.0)
    ax.set_ylim(0.0, float(density_nodes.max()) * 1.08)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.98,
        0.05,
        "11 points — interpolation entre les valeurs de la table",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )
    _save_png_svg(fig, output_path)
    return output_path


def plot_project_hu_to_density_calibration(
    output_path: Path = PROJECT_CALIBRATION_FIGURE,
) -> Path:
    """Plot the original simple water-scaled HU conversion as density."""
    hu_curve = np.linspace(-1000.0, 4000.0, 3000)
    mu_curve = ctac.hu_to_mu_208_cm_inv(hu_curve)
    density_equivalent = mu_curve / ctac.MU_WATER_208_CM_INV
    reference_hu = np.asarray([-1000.0, 0.0, 1000.0, 4000.0])
    reference_density = (
        ctac.hu_to_mu_208_cm_inv(reference_hu) / ctac.MU_WATER_208_CM_INV
    )

    fig, ax = plt.subplots(figsize=(8.6, 5.6), facecolor="white")
    ax.plot(
        hu_curve,
        density_equivalent,
        color="#1f77b4",
        linewidth=2.4,
        label="Conversion simple utilisée",
        zorder=2,
    )
    ax.scatter(
        reference_hu,
        reference_density,
        color="#d62728",
        edgecolor="white",
        linewidth=0.7,
        s=58,
        label="Valeurs de référence",
        zorder=3,
    )
    ax.set_title("Conversion simple : HU vers densité équivalente à l'eau")
    ax.set_xlabel("Unité Hounsfield (HU)")
    ax.set_ylabel("Densité équivalente à l'eau (g/cm³)")
    ax.set_xlim(-1100.0, 4120.0)
    ax.set_ylim(0.0, float(reference_density.max()) * 1.08)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.97,
        0.08,
        (
            r"$\rho_{eq}(HU)=\max\left[0,\,1+\frac{HU}{1000}\right]$"
            "\n"
            r"$\mu_{208}=0{,}135\,\rho_{eq}$  (cm$^{-1}$)"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        color="#333333",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9, "edgecolor": "#bbbbbb"},
    )
    _save_png_svg(fig, output_path)
    return output_path


def plot_catphan_t2_hu_to_mu_calibration(
    output_path: Path = CATPHAN_CALIBRATION_FIGURE,
) -> Path:
    """Plot the provisional direct T2 Catphan HU-to-mu208.4 calibration."""
    hu_nodes = np.asarray(ctac.CATPHAN_T2_110KVP_HU_NODES, dtype=float)
    mu_nodes = np.asarray(ctac.CATPHAN_T2_110KVP_MU_2084_CM_INV, dtype=float)
    hu_curve = np.linspace(hu_nodes[0], hu_nodes[-1], 3000)
    mu_curve = ctac.catphan_t2_hu_to_mu_2084_cm_inv(hu_curve)

    fig, ax = plt.subplots(figsize=(9.2, 5.8), facecolor="white")
    ax.plot(
        hu_curve,
        mu_curve,
        color="#1f77b4",
        linewidth=2.4,
        label="Interpolation linéaire par segments",
        zorder=2,
    )
    ax.scatter(
        hu_nodes,
        mu_nodes,
        color="#d62728",
        edgecolor="white",
        linewidth=0.7,
        s=60,
        label="Points Catphan T2 à 110 kVp",
        zorder=3,
    )
    annotation_offsets = {
        "air": (5, 7),
        "PMP": (5, 7),
        "LDPE": (5, 8),
        "polystyrene": (-10, 18),
        "water": (12, -20),
        "acrylic": (5, 8),
        "Delrin": (5, 8),
        "Teflon": (5, 8),
    }
    for material, hu_value, mu_value in zip(
        ctac.CATPHAN_T2_110KVP_MATERIAL_NAMES, hu_nodes, mu_nodes
    ):
        ax.annotate(
            material,
            (hu_value, mu_value),
            xytext=annotation_offsets[material],
            textcoords="offset points",
            ha="right" if material == "polystyrene" else "left",
            fontsize=8,
            color="#444444",
        )
    ax.set_title("Courbe provisoire T2 : HU vers μ à 208,4 keV")
    ax.set_xlabel("Unité Hounsfield (HU)")
    ax.set_ylabel(r"Coefficient linéaire μ$_{208,4}$ (cm$^{-1}$)")
    ax.set_xlim(hu_nodes[0] - 70.0, hu_nodes[-1] + 70.0)
    ax.set_ylim(0.0, float(mu_nodes.max()) * 1.13)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.98,
        0.05,
        "HU mesurés : T2, 110 kVp, B41s, 8 mm\n"
        "Patient : T2, 110 kVp, B08s, 5 mm — méthode exploratoire",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )
    _save_png_svg(fig, output_path)
    return output_path


def write_catphan_calibration_values(
    output_path: Path = CATPHAN_VALUES_FILE,
) -> Path:
    """Export every direct Catphan calibration node and its units."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        stream.write(
            "material,hu_t2_110kvp_b41s_8mm,density_g_cm3,"
            "mass_attenuation_2084_cm2_g,linear_attenuation_2084_cm_inv\n"
        )
        for material, hu, density, mu_over_rho, mu in zip(
            ctac.CATPHAN_T2_110KVP_MATERIAL_NAMES,
            ctac.CATPHAN_T2_110KVP_HU_NODES,
            ctac.CATPHAN_SPECIFIC_GRAVITY_G_CM3,
            ctac.CATPHAN_MASS_ATTENUATION_2084_CM2_G,
            ctac.CATPHAN_T2_110KVP_MU_2084_CM_INV,
        ):
            stream.write(
                f"{material},{hu:.8g},{density:.8g},{mu_over_rho:.9g},{mu:.9g}\n"
            )
    return output_path


def _paired_arrays(
    old_rows: List[Dict[str, Any]],
    new_rows: List[Dict[str, Any]],
    catphan_rows: List[Dict[str, Any]],
) -> Dict[str, np.ndarray]:
    if len({len(old_rows), len(new_rows), len(catphan_rows)}) != 1:
        raise ValueError("The CT conversion methods returned different numbers of acquisitions")
    return {
        "days": np.asarray([row["day_offset"] for row in old_rows], dtype=float),
        "qspect": np.asarray([row["qspect_activity_mbq"] for row in old_rows], dtype=float),
        "planar": np.asarray([row["planar_crop_local_activity_mbq"] for row in old_rows], dtype=float),
        "old_ctac": np.asarray([row["ctac_local_activity_mbq"] for row in old_rows], dtype=float),
        "new_ctac": np.asarray([row["ctac_local_activity_mbq"] for row in new_rows], dtype=float),
        "catphan_ctac": np.asarray(
            [row["ctac_local_activity_mbq"] for row in catphan_rows], dtype=float
        ),
        "old_factor": np.asarray([row["effective_ct_factor"] for row in old_rows], dtype=float),
        "new_factor": np.asarray([row["effective_ct_factor"] for row in new_rows], dtype=float),
        "catphan_factor": np.asarray(
            [row["effective_ct_factor"] for row in catphan_rows], dtype=float
        ),
        "catphan_outside_fraction": np.asarray(
            [row["ct_conversion_outside_calibration_fraction"] for row in catphan_rows],
            dtype=float,
        ),
        "catphan_below_fraction": np.asarray(
            [row["ct_conversion_below_calibration_fraction"] for row in catphan_rows],
            dtype=float,
        ),
        "catphan_above_fraction": np.asarray(
            [row["ct_conversion_above_calibration_fraction"] for row in catphan_rows],
            dtype=float,
        ),
    }


def plot_profile_crop_activity_comparison(
    values: Dict[str, np.ndarray], output_path: Path = ACTIVITY_FIGURE
) -> Path:
    """Plot profile-crop planar CTAC while varying only HU-to-mu conversion."""
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
    ax.plot(
        values["days"], values["catphan_ctac"], "D-", linewidth=2,
        label="CTAC — courbe Catphan T2 (provisoire)",
    )
    ax.set_title("CTAC planaire — crop par profils")
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
    """Compare each profile-crop CTAC estimate with Q/SPECT over time."""
    old_over_qspect = values["old_ctac"] / values["qspect"]
    new_over_qspect = values["new_ctac"] / values["qspect"]
    catphan_over_qspect = values["catphan_ctac"] / values["qspect"]

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
    ax.plot(
        values["days"], catphan_over_qspect, "D-", color="tab:blue", linewidth=2,
        label="Catphan T2 / Q/SPECT (provisoire)",
    )
    ax.set_title("Ratio CTAC planaire / Q/SPECT — crop par profils")
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
            values["catphan_ctac"],
            values["old_factor"], values["new_factor"], values["catphan_factor"],
            values["old_ctac"] / values["qspect"],
            values["new_ctac"] / values["qspect"],
            values["catphan_ctac"] / values["qspect"],
            100.0 * (values["new_ctac"] - values["old_ctac"]) / values["old_ctac"],
            100.0 * (values["catphan_ctac"] - values["old_ctac"]) / values["old_ctac"],
            values["catphan_outside_fraction"],
            values["catphan_below_fraction"],
            values["catphan_above_fraction"],
        ]
    )
    np.savetxt(
        output_path,
        matrix,
        delimiter=",",
        header=(
            "day,planar_profile_crop_mbq,qspect_mbq,ctac_old_mbq,ctac_raystation_materials_mbq,"
            "ctac_catphan_t2_mbq,effective_factor_old,effective_factor_raystation_materials,"
            "effective_factor_catphan_t2,ctac_old_over_qspect,"
            "ctac_raystation_materials_over_qspect,ctac_catphan_t2_over_qspect,"
            "raystation_minus_old_percent,catphan_minus_old_percent,"
            "catphan_voxels_outside_calibration_fraction,"
            "catphan_voxels_below_air_hu_fraction,catphan_voxels_above_teflon_hu_fraction"
        ),
        comments="",
        fmt="%.8g",
    )
    return output_path


def run_comparison() -> Dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    calibration_path = plot_hu_to_density_calibration()
    raystation_single_curve_path = plot_raystation_hu_to_density_single_curve()
    project_calibration_path = plot_project_hu_to_density_calibration()
    catphan_calibration_path = plot_catphan_t2_hu_to_mu_calibration()
    catphan_values_path = write_catphan_calibration_values()
    old_rows = attenuation_correction.ct_attenuation_correction_rows(
        crop_strategy="profile_qspect", conversion_method="water_scaled",
        align_pa_to_ap=ALIGN_PA_TO_AP,
    )
    new_rows = attenuation_correction.ct_attenuation_correction_rows(
        crop_strategy="profile_qspect", conversion_method="raystation_materials",
        align_pa_to_ap=ALIGN_PA_TO_AP,
    )
    catphan_rows = attenuation_correction.ct_attenuation_correction_rows(
        crop_strategy="profile_qspect", conversion_method="catphan_t2_110kvp",
        align_pa_to_ap=ALIGN_PA_TO_AP,
    )
    values = _paired_arrays(old_rows, new_rows, catphan_rows)
    activity_path = plot_profile_crop_activity_comparison(values)
    difference_path = plot_temporal_conversion_difference(values)
    values_path = write_values(values)

    print(f"Saved HU-density calibration: {calibration_path}")
    print(f"Saved single-curve RayStation calibration: {raystation_single_curve_path}")
    print(f"Saved project HU-density calibration: {project_calibration_path}")
    print(f"Saved provisional Catphan HU-mu calibration: {catphan_calibration_path}")
    print(f"Saved provisional Catphan calibration values: {catphan_values_path}")
    print(f"Saved profile-crop CTAC comparison: {activity_path}")
    print(f"Saved CTAC/Q-SPECT ratios over time: {difference_path}")
    print(f"Saved numerical values: {values_path}")
    return {
        "old_rows": old_rows,
        "new_rows": new_rows,
        "catphan_rows": catphan_rows,
        "values": values,
        "calibration_path": calibration_path,
        "raystation_single_curve_path": raystation_single_curve_path,
        "project_calibration_path": project_calibration_path,
        "catphan_calibration_path": catphan_calibration_path,
        "catphan_values_path": catphan_values_path,
        "activity_path": activity_path,
        "difference_path": difference_path,
        "values_path": values_path,
    }


if __name__ == "__main__":
    run_comparison()
