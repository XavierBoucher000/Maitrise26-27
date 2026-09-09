"""Compare CTAC results with historical versus DICOM-aligned AP/PA geometry."""

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

import attenuation_correction
import ct_attenuation_correction as ctac


OUTPUT_DIR = Path(__file__).resolve().parent / "fig" / "test"
FIGURE_PATH = OUTPUT_DIR / "ctac_ap_pa_alignment_option_comparison.png"
VALUES_PATH = OUTPUT_DIR / "ctac_ap_pa_alignment_option_values.csv"


def profile_crop_rows(align_pa_to_ap: bool) -> List[Dict[str, Any]]:
    return attenuation_correction.ct_attenuation_correction_rows(
        crop_strategy="profile_qspect",
        conversion_method=ctac.DEFAULT_CT_CONVERSION_METHOD,
        align_pa_to_ap=align_pa_to_ap,
    )


def plot_comparison(
    historical_rows: List[Dict[str, Any]],
    aligned_rows: List[Dict[str, Any]],
    output_path: Path = FIGURE_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    days = np.asarray([row["day_offset"] for row in historical_rows], dtype=float)
    qspect = np.asarray([row["qspect_activity_mbq"] for row in historical_rows], dtype=float)
    historical = np.asarray(
        [row["ctac_local_activity_mbq"] for row in historical_rows], dtype=float
    )
    aligned = np.asarray(
        [row["ctac_local_activity_mbq"] for row in aligned_rows], dtype=float
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), facecolor="white")
    axes[0].plot(days, qspect, "^-", color="tab:blue", linewidth=2.2, label="Q/SPECT")
    axes[0].plot(
        days, historical, "o-", color="tab:orange", linewidth=2,
        label="align_pa_to_ap=False",
    )
    axes[0].plot(
        days, aligned, "s-", color="tab:green", linewidth=2,
        label="align_pa_to_ap=True",
    )
    axes[0].set_ylabel("Activité estimée (MBq)")
    axes[0].set_title("CTAC planaire — crop par profils")

    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.4, label="Accord Q/SPECT")
    axes[1].plot(
        days, historical / qspect, "o-", color="tab:orange", linewidth=2,
        label="False / Q/SPECT",
    )
    axes[1].plot(
        days, aligned / qspect, "s-", color="tab:green", linewidth=2,
        label="True / Q/SPECT",
    )
    axes[1].set_ylabel("Ratio CTAC planaire / Q/SPECT")
    axes[1].set_title("Effet de l’alignement AP/PA")

    for axis in axes:
        axis.set_xlabel("Temps après la première acquisition (jours)")
        axis.grid(True, linestyle="--", alpha=0.3)
        axis.legend(frameon=False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def write_values(
    historical_rows: List[Dict[str, Any]],
    aligned_rows: List[Dict[str, Any]],
    output_path: Path = VALUES_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = []
    for historical, aligned in zip(historical_rows, aligned_rows):
        qspect = float(historical["qspect_activity_mbq"])
        false_activity = float(historical["ctac_local_activity_mbq"])
        true_activity = float(aligned["ctac_local_activity_mbq"])
        matrix.append(
            [
                historical["day_offset"], qspect, false_activity, true_activity,
                false_activity / qspect, true_activity / qspect,
                100.0 * (true_activity / false_activity - 1.0),
            ]
        )
    np.savetxt(
        output_path,
        np.asarray(matrix),
        delimiter=",",
        header=(
            "day,qspect_mbq,ctac_align_false_mbq,ctac_align_true_mbq,"
            "ctac_align_false_over_qspect,ctac_align_true_over_qspect,"
            "true_minus_false_percent"
        ),
        comments="",
        fmt="%.8g",
    )
    return output_path


def run_comparison() -> Dict[str, Any]:
    historical_rows = profile_crop_rows(False)
    aligned_rows = profile_crop_rows(True)
    figure_path = plot_comparison(historical_rows, aligned_rows)
    values_path = write_values(historical_rows, aligned_rows)
    print(f"Saved AP/PA option comparison: {figure_path}")
    print(f"Saved AP/PA option values: {values_path}")
    return {
        "historical_rows": historical_rows,
        "aligned_rows": aligned_rows,
        "figure_path": figure_path,
        "values_path": values_path,
    }


if __name__ == "__main__":
    run_comparison()
