"""Test whether the fixed Day-0 crop excludes a negligible planar fraction."""

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

import attenuation_correction


OUTPUT_DIR = Path(__file__).resolve().parent / "fig" / "crop_test"
ALIGN_PA_TO_AP = True
FIGURE_PATH = OUTPUT_DIR / "fixed_crop_excluded_counts_over_time.png"
VALUES_PATH = OUTPUT_DIR / "fixed_crop_excluded_counts_values.csv"


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


def run_test() -> Dict[str, Any]:
    rows = collect_fixed_crop_excluded_counts()
    figure_path = plot_excluded_counts(rows)
    values_path = write_values(rows)
    print(f"Saved crop excluded-count figure: {figure_path}")
    print(f"Saved crop excluded-count values: {values_path}")
    for row in rows:
        item = row["crop_excluded_counts"]
        print(
            f"day={row['day_offset']:.2f} | retained={100.0 * item['inside_fraction']:.2f}% | "
            f"excluded={100.0 * item['outside_fraction']:.2f}% "
            f"(above={100.0 * item['above_fraction']:.2f}%, "
            f"below={100.0 * item['below_fraction']:.2f}%)"
        )
    return {"rows": rows, "figure_path": figure_path, "values_path": values_path}


if __name__ == "__main__":
    run_test()
