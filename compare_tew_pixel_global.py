"""Compare pixel-wise and global TEW in the CTAC activity pipeline.

The comparison deliberately keeps the following elements identical:

* fixed Day0 planar crop;
* CT attenuation-factor map;
* local dwell time and camera sensitivity;
* paired Q/SPECT activity used only for evaluation.

Definitions
-----------
Pixel TEW (current CTAC pipeline):
    TEW is applied independently to every AP and PA pixel, negative corrected
    pixels are clipped to zero, and the AP/PA geometric mean is then formed.

Global TEW (comparison baseline):
    Lower-scatter, photopeak and upper-scatter geometric-mean images are summed
    inside the same crop. These integrated counts give one scalar TEW retention
    factor. That factor is applied to the CT-corrected raw-photopeak image sum.

The global method is a diagnostic baseline, not a spatial scatter-correction
map. It cannot preserve the spatial relationship between scatter and the CT
attenuation factor.
"""

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

import attenuation_correction as ac
import planar_processing
import qspect_processing


OUTPUT_DIR = Path(__file__).resolve().parent / "fig" / "ct_correction" / "tew_pixel_vs_global"
REPORT_PATH = OUTPUT_DIR / "tew_pixel_vs_global_report.txt"
ACTIVITY_FIGURE_PATH = OUTPUT_DIR / "tew_activity_comparison.png"
TEW_EFFECT_FIGURE_PATH = OUTPUT_DIR / "tew_effect_vs_no_tew.png"
QSPECT_RATIO_FIGURE_PATH = OUTPUT_DIR / "tew_planar_qspect_ratio.png"


def _window_geometric_mean(
    groups: Dict[str, Dict[str, Dict[str, Any]]],
    energy: str,
) -> np.ndarray:
    """Return the raw AP/PA geometric-mean image for one energy window."""
    window = groups.get(energy)
    if not window or "AP" not in window or "PA" not in window:
        raise ValueError(f"Missing AP/PA data for {energy}")
    return ac.geometric_mean(window["AP"]["image"], window["PA"]["image"])


def _tew_widths(groups: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, float]:
    """Read the TEW widths; AP and PA are expected to use the same windows."""
    required = ["Lower Scatter", "Photopeak", "Upper Scatter"]
    ap_items = []
    for energy in required:
        if energy not in groups or "AP" not in groups[energy]:
            raise ValueError(f"Missing AP window metadata for {energy}")
        ap_items.append(groups[energy]["AP"])
    widths = planar_processing.window_widths(ap_items)
    missing = [energy for energy in required if energy not in widths]
    if missing:
        raise ValueError(f"Missing TEW width(s): {', '.join(missing)}")
    return widths


def compare_tew_methods(
    planar_dir: Path = planar_processing.default_planar_study_dir(),
    qspect_dir: Path = qspect_processing.default_qspect_dir(),
) -> List[Dict[str, float]]:
    """Calculate final CTAC activity with pixel-wise and global TEW."""
    scans = ac.sorted_planar_scans(planar_dir)
    reference_rows = ac.ct_attenuation_correction_rows(
        planar_dir,
        qspect_dir,
        crop_strategy="fixed_day0",
    )

    if len(scans) != len(reference_rows):
        raise ValueError("Planar scan and CTAC reference-row counts do not match")

    results: List[Dict[str, float]] = []
    for scan, reference in zip(scans, reference_rows):
        groups = ac.group_by_energy_and_view(scan["images"])
        widths = _tew_widths(groups)

        top = int(reference["crop_top"])
        bottom = int(reference["crop_bottom"])
        factor_map = np.asarray(reference["ct_factor_resized"], dtype=np.float64)

        # Current method: TEW each detector view pixel by pixel, then AP/PA GM.
        ap_tew = ac.tew_correct_view_image(groups, "AP")
        pa_tew = ac.tew_correct_view_image(groups, "PA")
        pixel_tew_crop = ac.geometric_mean(ap_tew, pa_tew)[top:bottom, :]
        if pixel_tew_crop.shape != factor_map.shape:
            raise ValueError("Pixel-TEW crop and CT factor-map shapes do not match")
        pixel_ctac_counts = float(np.sum(pixel_tew_crop * factor_map))

        # Global baseline: one TEW factor from integrated GM-window crop counts.
        lower_crop = _window_geometric_mean(groups, "Lower Scatter")[top:bottom, :]
        photo_crop = _window_geometric_mean(groups, "Photopeak")[top:bottom, :]
        upper_crop = _window_geometric_mean(groups, "Upper Scatter")[top:bottom, :]

        lower_counts = float(np.sum(lower_crop))
        photo_counts = float(np.sum(photo_crop))
        upper_counts = float(np.sum(upper_crop))
        scatter_counts = (
            (
                lower_counts / widths["Lower Scatter"]
                + upper_counts / widths["Upper Scatter"]
            )
            / 2.0
            * widths["Photopeak"]
        )
        global_tew_counts = max(photo_counts - scatter_counts, 0.0)
        global_retention_factor = global_tew_counts / photo_counts if photo_counts > 0 else np.nan

        raw_photo_ctac_counts = float(np.sum(photo_crop * factor_map))
        global_ctac_counts = raw_photo_ctac_counts * global_retention_factor

        dwell_s = float(reference["local_dwell_time_s"])
        sensitivity = float(planar_processing.CAMERA_SENSITIVITY_CPS_PER_MBQ)
        no_tew_activity = planar_processing.counts_to_activity_mbq(
            raw_photo_ctac_counts,
            dwell_s,
            sensitivity,
        )
        pixel_activity = planar_processing.counts_to_activity_mbq(
            pixel_ctac_counts,
            dwell_s,
            sensitivity,
        )
        global_activity = planar_processing.counts_to_activity_mbq(
            global_ctac_counts,
            dwell_s,
            sensitivity,
        )
        qspect_activity = float(reference["qspect_activity_mbq"])
        difference_mbq = pixel_activity - global_activity
        difference_percent = (
            100.0 * difference_mbq / global_activity if global_activity > 0 else np.nan
        )
        pixel_reduction_percent = (
            100.0 * (pixel_activity / no_tew_activity - 1.0)
            if no_tew_activity > 0
            else np.nan
        )
        global_reduction_percent = (
            100.0 * (global_activity / no_tew_activity - 1.0)
            if no_tew_activity > 0
            else np.nan
        )

        results.append(
            {
                "day": float(reference["day_offset"]),
                "no_tew_activity_mbq": float(no_tew_activity),
                "pixel_activity_mbq": float(pixel_activity),
                "global_activity_mbq": float(global_activity),
                "difference_mbq": float(difference_mbq),
                "difference_percent": float(difference_percent),
                "pixel_reduction_percent": float(pixel_reduction_percent),
                "global_reduction_percent": float(global_reduction_percent),
                "global_tew_retention_factor": float(global_retention_factor),
                "global_scatter_fraction": float(scatter_counts / photo_counts),
                "no_tew_over_qspect": float(no_tew_activity / qspect_activity),
                "pixel_over_qspect": float(pixel_activity / qspect_activity),
                "global_over_qspect": float(global_activity / qspect_activity),
                "qspect_activity_mbq": qspect_activity,
            }
        )

    return results


def write_report(rows: List[Dict[str, float]]) -> Path:
    """Write a compact text report containing definitions and results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "Pixel-wise TEW versus global TEW in the CTAC pipeline",
        "====================================================",
        "",
        "Common settings:",
        "  - fixed Day0 crop",
        "  - identical CT attenuation-factor map",
        "  - identical local dwell time and camera sensitivity",
        "  - no dead-time correction in this CTAC comparison",
        "",
        "Pixel TEW:",
        "  TEW(AP pixels), TEW(PA pixels), then pixel-wise geometric mean and CTAC.",
        "",
        "Global TEW:",
        "  One TEW retention factor from crop-integrated geometric-mean window counts,",
        "  applied to the CT-corrected raw-photopeak sum.",
        "",
        "No TEW:",
        "  Raw photopeak geometric mean followed by the same pixel-wise CT correction.",
        "",
        "Positive difference means pixel TEW gives a larger activity than global TEW.",
        "",
        " day | no TEW MBq | pixel MBq | global MBq | pixel vs noTEW % | global vs noTEW % | pixel-global % | noTEW/QSP | pixel/QSP | global/QSP",
        "---- | ---------- | --------- | ---------- | ---------------- | ----------------- | -------------- | --------- | --------- | ----------",
    ]
    for row in rows:
        lines.append(
            f"{row['day']:4.2f} | "
            f"{row['no_tew_activity_mbq']:10.1f} | "
            f"{row['pixel_activity_mbq']:9.1f} | "
            f"{row['global_activity_mbq']:10.1f} | "
            f"{row['pixel_reduction_percent']:16.2f} | "
            f"{row['global_reduction_percent']:17.2f} | "
            f"{row['difference_percent']:14.2f} | "
            f"{row['no_tew_over_qspect']:9.3f} | "
            f"{row['pixel_over_qspect']:9.3f} | "
            f"{row['global_over_qspect']:10.3f}"
        )

    lines.extend(
        [
            "",
            "Interpretation:",
            "  TEW subtraction is linear before clipping, geometric mean and CT weighting.",
            "  Differences arise from the order of nonlinear operations, spatial CT weighting,",
            "  and clipping negative pixel-wise TEW values to zero.",
            "  The global method is a robustness check, not a replacement for a spatial CTAC map.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return REPORT_PATH


def _plot_arrays(rows: List[Dict[str, float]]) -> Dict[str, np.ndarray]:
    """Convert result rows into arrays shared by the three figures."""
    return {
        "day": np.asarray([row["day"] for row in rows]),
        "no_tew": np.asarray([row["no_tew_activity_mbq"] for row in rows]),
        "pixel": np.asarray([row["pixel_activity_mbq"] for row in rows]),
        "global": np.asarray([row["global_activity_mbq"] for row in rows]),
        "qspect": np.asarray([row["qspect_activity_mbq"] for row in rows]),
        "pixel_reduction": np.asarray([row["pixel_reduction_percent"] for row in rows]),
        "global_reduction": np.asarray([row["global_reduction_percent"] for row in rows]),
    }


def plot_activity_comparison(rows: List[Dict[str, float]]) -> Path:
    """Plot the three planar CTAC estimates and the Q/SPECT activity."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    values = _plot_arrays(rows)
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.plot(values["day"], values["no_tew"], "D-", color="tab:blue", label="CTAC sans TEW")
    ax.plot(values["day"], values["pixel"], "o-", color="tab:orange", label="CTAC + TEW pixelisé")
    ax.plot(values["day"], values["global"], "s-", color="tab:green", label="CTAC + TEW global")
    ax.plot(values["day"], values["qspect"], "^-", color="tab:red", label="Q/SPECT")
    ax.set_xlabel("Temps après la première acquisition (jours)", labelpad=10)
    ax.set_ylabel("Activité estimée (MBq)")
    ax.set_title("Activité finale — crop fixe Day0")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout(pad=1.2)
    fig.savefig(ACTIVITY_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return ACTIVITY_FIGURE_PATH


def plot_tew_effect(rows: List[Dict[str, float]]) -> Path:
    """Plot the change caused by each TEW method relative to no TEW."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    values = _plot_arrays(rows)
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.axhline(0.0, color="black", linewidth=1)
    ax.plot(
        values["day"],
        values["pixel_reduction"],
        "o-",
        color="tab:orange",
        label="TEW pixelisé",
    )
    ax.plot(
        values["day"],
        values["global_reduction"],
        "s-",
        color="tab:green",
        label="TEW global",
    )
    ax.set_xlabel("Temps après la première acquisition (jours)", labelpad=10)
    ax.set_ylabel("Variation par rapport à sans TEW (%)")
    ax.set_title("Effet de la correction TEW — crop fixe Day0")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout(pad=1.2)
    fig.savefig(TEW_EFFECT_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return TEW_EFFECT_FIGURE_PATH


def plot_qspect_ratio(rows: List[Dict[str, float]]) -> Path:
    """Plot each planar activity divided by its paired Q/SPECT activity."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    values = _plot_arrays(rows)
    qspect = values["qspect"]
    no_tew_ratio = values["no_tew"] / qspect
    pixel_ratio = values["pixel"] / qspect
    global_ratio = values["global"] / qspect

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.axhline(1.0, color="black", linewidth=1, label="Accord avec Q/SPECT")
    ax.plot(values["day"], no_tew_ratio, "D-", color="tab:blue", label="CTAC sans TEW")
    ax.plot(values["day"], pixel_ratio, "o-", color="tab:orange", label="CTAC + TEW pixelisé")
    ax.plot(values["day"], global_ratio, "s-", color="tab:green", label="CTAC + TEW global")
    ax.set_xlabel("Temps après la première acquisition (jours)", labelpad=10)
    ax.set_ylabel("Ratio activité planaire / Q/SPECT")
    ax.set_title("Ratio des activités planaires au Q/SPECT")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout(pad=1.2)
    fig.savefig(QSPECT_RATIO_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return QSPECT_RATIO_FIGURE_PATH


def main() -> None:
    rows = compare_tew_methods()
    report_path = write_report(rows)
    activity_figure_path = plot_activity_comparison(rows)
    tew_effect_figure_path = plot_tew_effect(rows)
    qspect_ratio_figure_path = plot_qspect_ratio(rows)

    print("day | no TEW MBq | pixel CTAC MBq | global CTAC MBq | pixel-global %")
    for row in rows:
        print(
            f"{row['day']:.2f} | "
            f"{row['no_tew_activity_mbq']:.1f} | "
            f"{row['pixel_activity_mbq']:.1f} | "
            f"{row['global_activity_mbq']:.1f} | "
            f"{row['difference_percent']:+.2f}%"
        )
    print(f"Saved report: {report_path}")
    print(f"Saved activity figure: {activity_figure_path}")
    print(f"Saved TEW-effect figure: {tew_effect_figure_path}")
    print(f"Saved planar/Q/SPECT-ratio figure: {qspect_ratio_figure_path}")


if __name__ == "__main__":
    main()
