from pathlib import Path
import sys
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import correction_3DEW as c3
import dicom_loader
import planar_processing


ROOT = PROJECT_ROOT
OUT_DIR = ROOT / "fig" / "attenuation"
COUNTS_FIGURE_PATH = OUT_DIR / "raw_window_counts_by_day.png"
COUNTS_ZOOM_FIGURE_PATH = OUT_DIR / "raw_window_counts_by_day_zoom_no_low_energy.png"
RATIOS_FIGURE_PATH = OUT_DIR / "window_ratios_by_day.png"
RATIOS_ZOOM_FIGURE_PATH = OUT_DIR / "window_ratios_by_day_zoom_no_low_energy.png"
TABLE_PATH = OUT_DIR / "window_counts_and_ratios_by_day.csv"


def extract_window_count_rows(study_dir: Path) -> List[Dict[str, Any]]:
    patient_scans = dicom_loader.load_patient_scans(study_dir)
    rows = []
    scan_labels = [scan["scan_name"] for scan in patient_scans["scans"]]
    day_offsets = dicom_loader.scan_day_offsets(patient_scans, scan_labels)

    for scan, day_offset in zip(patient_scans["scans"], day_offsets):
        geometric_images = planar_processing.geometric_mean_images(scan["images"])
        counts = c3.extract_counts_from_images(geometric_images)
        widths = planar_processing.window_widths(geometric_images)
        photopeak = counts.get("Photopeak")
        lower = counts.get("Lower Scatter")
        upper = counts.get("Upper Scatter")
        low_energy = counts.get("Low Energy Scatter")
        scatter_estimate = None

        if all(value is not None for value in [photopeak, lower, upper]):
            scatter_estimate, _corrected = c3.tew_scatter_estimate(
                lower,
                photopeak,
                upper,
                widths.get("Lower Scatter"),
                widths.get("Photopeak"),
                widths.get("Upper Scatter"),
            )

        row = {
            "day": day_offset,
            "scan": scan["scan_name"],
            "photopeak_counts": photopeak,
            "lower_scatter_counts": lower,
            "upper_scatter_counts": upper,
            "low_energy_scatter_counts": low_energy,
            "tew_scatter_estimate_counts": scatter_estimate,
        }

        if photopeak and photopeak > 0:
            row.update(
                {
                    "lower_to_photopeak": None if lower is None else lower / photopeak,
                    "upper_to_photopeak": None if upper is None else upper / photopeak,
                    "low_energy_to_photopeak": None if low_energy is None else low_energy / photopeak,
                    "tew_scatter_to_photopeak": None if scatter_estimate is None else scatter_estimate / photopeak,
                }
            )
        rows.append(row)

    return rows


def plot_counts(df: pd.DataFrame, include_low_energy: bool, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor="white")
    series = [
        ("Photopeak", "photopeak_counts", "o"),
        ("Lower scatter", "lower_scatter_counts", "s"),
        ("Upper scatter", "upper_scatter_counts", "^"),
        ("Low-energy scatter", "low_energy_scatter_counts", "D"),
        ("TEW scatter estimate", "tew_scatter_estimate_counts", "x"),
    ]

    for label, column, marker in series:
        if not include_low_energy and column == "low_energy_scatter_counts":
            continue
        if column not in df or df[column].isna().all():
            continue
        ax.plot(df["day"], df[column], marker=marker, linewidth=2.0, label=label)

    ax.set_xlabel("Day")
    ax.set_ylabel("Raw geometric-mean counts")
    title = "Raw planar counts by energy window"
    if not include_low_energy:
        title += " (zoom without low-energy window)"
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_ratios(df: pd.DataFrame, include_low_energy: bool, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor="white")
    series = [
        ("Lower / photopeak", "lower_to_photopeak", "s"),
        ("Upper / photopeak", "upper_to_photopeak", "^"),
        ("Low-energy / photopeak", "low_energy_to_photopeak", "D"),
        ("TEW scatter / photopeak", "tew_scatter_to_photopeak", "o"),
    ]

    for label, column, marker in series:
        if not include_low_energy and column == "low_energy_to_photopeak":
            continue
        if column not in df or df[column].isna().all():
            continue
        ax.plot(df["day"], df[column], marker=marker, linewidth=2.0, label=label)

    ax.set_xlabel("Day")
    ax.set_ylabel("Window ratio")
    title = "Energy-window ratios over time"
    if not include_low_energy:
        title += " (zoom without low-energy ratio)"
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = extract_window_count_rows(planar_processing.default_planar_study_dir())
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_PATH, index=False)
    plot_counts(df, include_low_energy=True, output_path=COUNTS_FIGURE_PATH)
    plot_counts(df, include_low_energy=False, output_path=COUNTS_ZOOM_FIGURE_PATH)
    plot_ratios(df, include_low_energy=True, output_path=RATIOS_FIGURE_PATH)
    plot_ratios(df, include_low_energy=False, output_path=RATIOS_ZOOM_FIGURE_PATH)
    print(f"Saved: {COUNTS_FIGURE_PATH}")
    print(f"Saved: {COUNTS_FIGURE_PATH.with_suffix('.svg')}")
    print(f"Saved: {COUNTS_ZOOM_FIGURE_PATH}")
    print(f"Saved: {COUNTS_ZOOM_FIGURE_PATH.with_suffix('.svg')}")
    print(f"Saved: {RATIOS_FIGURE_PATH}")
    print(f"Saved: {RATIOS_FIGURE_PATH.with_suffix('.svg')}")
    print(f"Saved: {RATIOS_ZOOM_FIGURE_PATH}")
    print(f"Saved: {RATIOS_ZOOM_FIGURE_PATH.with_suffix('.svg')}")
    print(f"Saved: {TABLE_PATH}")


if __name__ == "__main__":
    main()
