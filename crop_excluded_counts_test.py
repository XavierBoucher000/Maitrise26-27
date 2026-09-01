"""Test whether the fixed Day-0 crop excludes a negligible planar fraction."""

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

import attenuation_correction
import planar_processing


OUTPUT_DIR = Path(__file__).resolve().parent / "fig" / "crop_test"
ALIGN_PA_TO_AP = True
FIGURE_PATH = OUTPUT_DIR / "fixed_crop_excluded_counts_over_time.png"
VALUES_PATH = OUTPUT_DIR / "fixed_crop_excluded_counts_values.csv"
RECOVERED_ACTIVITY_FIGURE_PATH = (
    OUTPUT_DIR / "fixed_crop_recovered_activity_without_attenuation.png"
)
RECOVERED_ACTIVITY_VALUES_PATH = (
    OUTPUT_DIR / "fixed_crop_recovered_activity_without_attenuation_values.csv"
)


def collect_fixed_crop_excluded_counts() -> List[Dict[str, Any]]:
    """Run the existing fixed-Day0 pipeline and return its crop diagnostics."""
    return attenuation_correction.ct_attenuation_correction_rows(
        crop_strategy="fixed_day0",
        conversion_method="water_scaled",
        align_pa_to_ap=ALIGN_PA_TO_AP,
    )


def plot_excluded_counts(
    rows: List[Dict[str, Any]], output_path: Path = FIGURE_PATH
) -> Path:
    """Plot retained and excluded body-masked planar count fractions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    days = np.asarray([row["day_offset"] for row in rows], dtype=float)
    diagnostics = [row["crop_excluded_counts"] for row in rows]
    above = 100.0 * np.asarray([item["above_fraction"] for item in diagnostics])
    below = 100.0 * np.asarray([item["below_fraction"] for item in diagnostics])
    outside = 100.0 * np.asarray([item["outside_fraction"] for item in diagnostics])
    retained = 100.0 * np.asarray([item["inside_fraction"] for item in diagnostics])
    outside_unmasked = 100.0 * np.asarray(
        [item["unmasked_outside_fraction"] for item in diagnostics]
    )

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True, facecolor="white")
    axes[0].plot(days, above, "o-", linewidth=2, label="Retiré au-dessus")
    axes[0].plot(days, below, "s-", linewidth=2, label="Retiré en-dessous")
    axes[0].plot(days, outside, "^-", linewidth=2.2, label="Total retiré")
    axes[0].axhline(5.0, color="0.35", linestyle="--", linewidth=1.2, label="Repère exploratoire 5 %")
    axes[0].set_ylabel("Comptes corporels retirés (%)")
    axes[0].set_title("Effet du crop fixe Day 0 sur les comptes planaires")
    axes[0].legend(frameon=False)

    axes[1].plot(days, retained, "o-", linewidth=2.2, label="Conservé — masque 1 %")
    axes[1].plot(
        days, 100.0 - outside_unmasked, "s--", linewidth=1.8,
        label="Conservé — tous les comptes positifs",
    )
    axes[1].set_xlabel("Temps après la première acquisition (jours)")
    axes[1].set_ylabel("Comptes conservés dans le crop (%)")
    axes[1].legend(frameon=False)

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def write_values(
    rows: List[Dict[str, Any]], output_path: Path = VALUES_PATH
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = []
    for row in rows:
        item = row["crop_excluded_counts"]
        matrix.append(
            [
                row["day_offset"], row["crop_top"], row["crop_bottom"],
                100.0 * item["above_fraction"],
                100.0 * item["below_fraction"],
                100.0 * item["outside_fraction"],
                100.0 * item["inside_fraction"],
                100.0 * item["unmasked_outside_fraction"],
                item["masked_counts"]["inside"],
                item["masked_counts"]["outside"],
            ]
        )
    np.savetxt(
        output_path,
        np.asarray(matrix),
        delimiter=",",
        header=(
            "day,crop_top,crop_bottom,above_percent,below_percent,outside_percent,"
            "retained_percent,unmasked_outside_percent,masked_inside_counts,masked_outside_counts"
        ),
        comments="",
        fmt="%.8g",
    )
    return output_path


def calculate_recovered_activity(rows: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """Add excluded TEW/GM activity without applying CT attenuation outside crop.

    The crop keeps its existing CT attenuation correction. Positive counts
    outside the crop are converted with the same local dwell time and camera
    sensitivity, but with an attenuation factor of exactly one. The masked
    estimate is the primary conservative result; the unmasked estimate is a
    sensitivity analysis for positive background outside the crop.
    """
    results: List[Dict[str, float]] = []
    sensitivity = planar_processing.CAMERA_SENSITIVITY_CPS_PER_MBQ
    for row in rows:
        diagnostics = row["crop_excluded_counts"]
        dwell_s = float(row["local_dwell_time_s"])
        denominator = dwell_s * sensitivity
        masked_outside_counts = float(diagnostics["masked_counts"]["outside"])
        positive_outside_counts = float(diagnostics["positive_counts"]["outside"])
        recovered_masked_mbq = masked_outside_counts / denominator
        recovered_positive_mbq = positive_outside_counts / denominator
        ctac_crop_mbq = float(row["ctac_local_activity_mbq"])
        qspect_mbq = float(row["qspect_activity_mbq"])
        hybrid_masked_mbq = ctac_crop_mbq + recovered_masked_mbq
        hybrid_positive_mbq = ctac_crop_mbq + recovered_positive_mbq
        results.append(
            {
                "day": float(row["day_offset"]),
                "ctac_crop_mbq": ctac_crop_mbq,
                "outside_masked_counts": masked_outside_counts,
                "outside_positive_counts": positive_outside_counts,
                "recovered_outside_no_attenuation_mbq": recovered_masked_mbq,
                "recovered_outside_positive_no_attenuation_mbq": recovered_positive_mbq,
                "hybrid_activity_mbq": hybrid_masked_mbq,
                "hybrid_positive_activity_mbq": hybrid_positive_mbq,
                "qspect_activity_mbq": qspect_mbq,
                "ctac_crop_over_qspect": ctac_crop_mbq / qspect_mbq,
                "hybrid_over_qspect": hybrid_masked_mbq / qspect_mbq,
                "hybrid_positive_over_qspect": hybrid_positive_mbq / qspect_mbq,
            }
        )
    return results


def plot_recovered_activity(
    values: List[Dict[str, float]],
    output_path: Path = RECOVERED_ACTIVITY_FIGURE_PATH,
) -> Path:
    """Plot the activity recovered outside the crop without attenuation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    days = np.asarray([row["day"] for row in values])
    ctac = np.asarray([row["ctac_crop_mbq"] for row in values])
    hybrid = np.asarray([row["hybrid_activity_mbq"] for row in values])
    qspect = np.asarray([row["qspect_activity_mbq"] for row in values])
    ctac_ratio = np.asarray([row["ctac_crop_over_qspect"] for row in values])
    hybrid_ratio = np.asarray([row["hybrid_over_qspect"] for row in values])

    fig, axes = plt.subplots(
        2, 1, figsize=(8.4, 7.0), sharex=True, facecolor="white", layout="constrained"
    )
    axes[0].plot(days, qspect, "D-", color="black", linewidth=2.0, label="Q/SPECT")
    axes[0].plot(
        days, ctac, "s-", color="#1f77b4", linewidth=2.0,
        label="Planaire CTAC — crop fixe J0",
    )
    axes[0].plot(
        days, hybrid, "o-", color="#ff7f0e", linewidth=2.2,
        label="CTAC crop + hors crop sans atténuation",
    )
    axes[0].set_ylabel("Activité (MBq)")
    axes[0].set_title("Réintégration des counts hors crop sans correction d'atténuation")
    axes[0].legend(frameon=False)

    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.3)
    axes[1].plot(
        days, ctac_ratio, "s-", color="#1f77b4", linewidth=2.0,
        label="CTAC crop / Q/SPECT",
    )
    axes[1].plot(
        days, hybrid_ratio, "o-", color="#ff7f0e", linewidth=2.2,
        label="Avec hors crop sans atténuation / Q/SPECT",
    )
    axes[1].set_xlabel("Temps après la première acquisition (jours)")
    axes[1].set_ylabel("Rapport relatif au Q/SPECT")
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.3)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def write_recovered_activity_values(
    values: List[Dict[str, float]],
    output_path: Path = RECOVERED_ACTIVITY_VALUES_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(values[0])
    matrix = np.asarray([[row[column] for column in columns] for row in values])
    np.savetxt(
        output_path,
        matrix,
        delimiter=",",
        header=",".join(columns),
        comments="",
        fmt="%.8g",
    )
    return output_path


def run_test() -> Dict[str, Any]:
    rows = collect_fixed_crop_excluded_counts()
    figure_path = plot_excluded_counts(rows)
    values_path = write_values(rows)
    recovered_values = calculate_recovered_activity(rows)
    recovered_figure_path = plot_recovered_activity(recovered_values)
    recovered_values_path = write_recovered_activity_values(recovered_values)
    print(f"Saved crop excluded-count figure: {figure_path}")
    print(f"Saved crop excluded-count values: {values_path}")
    print(f"Saved recovered-activity figure: {recovered_figure_path}")
    print(f"Saved recovered-activity values: {recovered_values_path}")
    for row in rows:
        item = row["crop_excluded_counts"]
        print(
            f"day={row['day_offset']:.2f} | retained={100.0 * item['inside_fraction']:.2f}% | "
            f"excluded={100.0 * item['outside_fraction']:.2f}% "
            f"(above={100.0 * item['above_fraction']:.2f}%, "
            f"below={100.0 * item['below_fraction']:.2f}%)"
        )
    for item in recovered_values:
        print(
            f"day={item['day']:.2f} | recovered={item['recovered_outside_no_attenuation_mbq']:.2f} MBq | "
            f"hybrid={item['hybrid_activity_mbq']:.2f} MBq | "
            f"hybrid/QSPECT={item['hybrid_over_qspect']:.3f}"
        )
    return {
        "rows": rows,
        "figure_path": figure_path,
        "values_path": values_path,
        "recovered_values": recovered_values,
        "recovered_figure_path": recovered_figure_path,
        "recovered_values_path": recovered_values_path,
    }


if __name__ == "__main__":
    run_test()
