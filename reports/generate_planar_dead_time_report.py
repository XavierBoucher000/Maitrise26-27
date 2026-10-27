from pathlib import Path
import csv
import os
import sys
from typing import Any, Dict, Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import figure_layout
import dead_time
import dicom_loader
import planar_processing
import study_layout


ROOT = PROJECT_ROOT
OUT_DIR = figure_layout.patient_cycle_dir(__file__)
OUT_PATH = OUT_DIR / "planar_dead_time_correction_report.txt"
DTCF_FIGURE_PATH = OUT_DIR / "planar_dead_time_dtcf_summary.png"
GLOBAL_OUT_DIR = figure_layout.shared_dir(__file__)
GLOBAL_RATE_FIGURE_PATH = GLOBAL_OUT_DIR / "global_dead_time_count_rates_all_patients.png"
GLOBAL_LOSS_FIGURE_PATH = GLOBAL_OUT_DIR / "global_dead_time_count_loss_all_patients.png"
GLOBAL_RATE_CSV_PATH = GLOBAL_OUT_DIR / "global_dead_time_rates_all_patients.csv"


MONTH_NAMES_FR = {
    "01": "janvier",
    "02": "février",
    "03": "mars",
    "04": "avril",
    "05": "mai",
    "06": "juin",
    "07": "juillet",
    "08": "août",
    "09": "septembre",
    "10": "octobre",
    "11": "novembre",
    "12": "décembre",
}

CYCLE_COLORS = {
    ("p11", "2026-05__Studies"): "#1F77B4",
    ("p11", "2026-07__Studies"): "#6F4C9B",
    ("p8", "2026-06__Studies"): "#D62728",
    ("p17", "2026-09__Studies"): "#2CA02C",
}
DEFAULT_CYCLE_COLOR = "#777777"


def fmt(value: Any, digits: int = 2, default: str = "n/a") -> str:
    if value is None:
        return default
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def report_rows(title: str, rows: Iterable[Dict[str, Any]]) -> List[str]:
    lines = [
        title,
        "-" * len(title),
        (
            "day | DTCF global | DTCF AP | DTCF PA | delta global/per-view % | "
            "label | TEW counts before DT | TEW counts after DT | loss % | "
            "RWo AP cps | RWo PA cps | RWo mean cps | duration s | corrected cps | activity MBq"
        ),
    ]
    for row in rows:
        corrected_counts = row.get("tew_corrected_counts")
        duration = row.get("duration_seconds")
        corrected_cps = None if corrected_counts is None or duration in (None, 0) else corrected_counts / duration
        rates_by_view = row.get("wide_spectrum_cps_by_view") or {}
        dtcf_by_view = row.get("dead_time_per_view_dtcf") or {}
        lines.append(
            " | ".join(
                [
                    fmt(row.get("day_offset"), 2),
                    fmt(row.get("dead_time_dtcf"), 4),
                    fmt(dtcf_by_view.get("AP"), 4),
                    fmt(dtcf_by_view.get("PA"), 4),
                    fmt(row.get("dead_time_global_vs_per_view_dtcf_percent"), 3),
                    str(row.get("label", "")),
                    fmt(row.get("tew_corrected_counts_before_dead_time"), 1),
                    fmt(corrected_counts, 1),
                    fmt(row.get("dead_time_loss_percent"), 2),
                    fmt(rates_by_view.get("AP"), 1),
                    fmt(rates_by_view.get("PA"), 1),
                    fmt(row.get("wide_spectrum_cps"), 1),
                    fmt(duration, 1),
                    fmt(corrected_cps, 1),
                    fmt(row.get("planar_activity_mbq"), 1),
                ]
            )
        )
    lines.append("")
    return lines


def plot_dtcf_summary(series: Dict[str, List[Dict[str, Any]]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor="white")
    markers = ["o", "s", "^"]

    for marker, (label, rows) in zip(markers, series.items()):
        day = [row["day_offset"] for row in rows]
        dtcf = [row.get("dead_time_dtcf", 1.0) for row in rows]
        rwo = [float(row.get("wide_spectrum_cps")) / 1000.0 for row in rows]
        axes[0].plot(day, dtcf, marker=marker, linewidth=2.0, label=label)
        axes[1].scatter(rwo, dtcf, marker=marker, s=42, label=label, zorder=3)

    axes[0].set_xlabel("Jour depuis la première acquisition")
    axes[0].set_ylabel("DTCF")
    axes[0].set_title("Facteur de correction du temps mort")
    axes[0].grid(True, linestyle="--", alpha=0.35)

    maximum_rate = planar_processing.DEAD_TIME_MAX_CALIBRATED_RWO_CPS
    curve_rates = np.linspace(0.0, maximum_rate, 500)
    curve_dtcf = [
        dead_time.dead_time_correction_factor(
            rate,
            planar_processing.DEAD_TIME_TAU_US,
            max_calibrated_rwo_cps=maximum_rate,
        )[0]
        for rate in curve_rates
    ]
    axes[1].plot(
        curve_rates / 1000.0,
        curve_dtcf,
        color="#333333",
        linewidth=1.8,
        label=f"Modèle paralysable, τ={planar_processing.DEAD_TIME_TAU_US:.3f} µs",
        zorder=2,
    )
    axes[1].axvline(
        maximum_rate / 1000.0,
        color="#C41230",
        linestyle="--",
        linewidth=1.2,
        label="Limite expérimentale ~355 kcps",
    )
    axes[1].set_xlabel("Taux large spectre moyen $R_{Wo}$ (kcps/détecteur)")
    axes[1].set_ylabel("DTCF")
    axes[1].set_title("Courbe du facteur de correction")
    axes[1].grid(True, linestyle="--", alpha=0.35)

    for ax in axes:
        ax.legend(frameon=False, fontsize=9)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.tight_layout()
    DTCF_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DTCF_FIGURE_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _cycle_display_label(dataset: study_layout.CycleDataset) -> str:
    """Return a concise patient/cycle label for shared figures."""
    year, month = dataset.cycle[:7].split("-")
    return f"{dataset.patient.upper()} — {MONTH_NAMES_FR.get(month, month)} {year}"


def _full_body_dead_time_rows(
    dataset: study_layout.CycleDataset,
) -> List[Dict[str, Any]]:
    """Calculate dead-time monitoring values for genuine whole-body scans.

    Some cycle folders contain an additional short local acquisition with the
    same ``WB RAPIDE`` series description.  The dead-time comparison must use
    comparable whole-body scans, so acquisitions shorter than 80% of the
    longest scan in that cycle are excluded.  For the current Siemens data,
    this retains the 2000-mm scans and rejects the 390-mm P11 July acquisition.
    """
    patient_scans = dicom_loader.load_patient_scans(
        dataset.planar_dir,
        series_description_filter="WB RAPIDE",
    )
    candidates: List[Dict[str, Any]] = []
    for scan in patient_scans["scans"]:
        images = scan.get("images") or []
        if not images:
            continue
        scan_lengths = [
            float(item["scan_length"])
            for item in images
            if item.get("scan_length") not in (None, "")
        ]
        if not scan_lengths:
            continue
        candidates.append(
            {
                "scan": scan,
                "images": images,
                "scan_length_mm": float(np.median(scan_lengths)),
            }
        )
    if not candidates:
        return []

    maximum_scan_length = max(row["scan_length_mm"] for row in candidates)
    minimum_full_body_length = 0.8 * maximum_scan_length
    rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        if candidate["scan_length_mm"] < minimum_full_body_length:
            continue
        images = candidate["images"]
        duration_seconds = planar_processing.acquisition_duration_seconds(images)
        rwo_cps, _counts_by_view, rates_by_view = (
            planar_processing.average_wide_spectrum_rate(images, duration_seconds)
        )
        dtcf, _warnings = dead_time.dead_time_correction_factor(
            rwo_cps,
            planar_processing.DEAD_TIME_TAU_US,
            max_calibrated_rwo_cps=(
                planar_processing.DEAD_TIME_MAX_CALIBRATED_RWO_CPS
            ),
        )
        rows.append(
            {
                "patient": dataset.patient,
                "cycle": dataset.cycle,
                "dataset_label": _cycle_display_label(dataset),
                "datetime": dicom_loader.scan_datetime(candidate["scan"]),
                "scan_length_mm": candidate["scan_length_mm"],
                "duration_seconds": duration_seconds,
                "rwo_ap_cps": rates_by_view["AP"],
                "rwo_pa_cps": rates_by_view["PA"],
                "rwo_mean_cps": rwo_cps,
                "dtcf": dtcf,
                "loss_percent": 100.0 * (1.0 - 1.0 / dtcf),
            }
        )

    rows.sort(key=lambda row: (row["datetime"] is None, row["datetime"]))
    first_datetime = next(
        (row["datetime"] for row in rows if row["datetime"] is not None),
        None,
    )
    for index, row in enumerate(rows):
        if first_datetime is not None and row["datetime"] is not None:
            day_offset = (row["datetime"] - first_datetime).total_seconds() / 86400.0
            row["day_offset"] = day_offset
            row["timepoint"] = f"J{int(round(day_offset))}"
        else:
            row["day_offset"] = float(index)
            row["timepoint"] = f"J{index}"
    return rows


def collect_global_dead_time_rows() -> List[Dict[str, Any]]:
    """Collect full-body planar dead-time values from every patient/cycle."""
    groups: List[List[Dict[str, Any]]] = []
    for dataset in study_layout.discover_cycles():
        rows = _full_body_dead_time_rows(dataset)
        if rows:
            groups.append(rows)
    groups.sort(
        key=lambda rows: figure_layout.patient_cycle_sort_key(
            rows[0]["patient"], rows[0]["cycle"]
        )
    )
    return [row for rows in groups for row in rows]


def plot_global_rate_summary(rows: List[Dict[str, Any]]) -> None:
    """Plot rates and count losses as two separate presentation figures."""
    if not rows:
        raise ValueError("No whole-body planar acquisition found")

    groups: List[tuple[str, List[Dict[str, Any]]]] = []
    for row in rows:
        if not groups or groups[-1][0] != row["dataset_label"]:
            groups.append((row["dataset_label"], []))
        groups[-1][1].append(row)

    positions: List[float] = []
    colors: List[str] = []
    timepoint_labels: List[str] = []
    group_centers: List[tuple[float, str]] = []
    separators: List[float] = []
    cursor = 0.0
    for group_index, (label, group_rows) in enumerate(groups):
        group_positions = []
        group_color = CYCLE_COLORS.get(
            (group_rows[0]["patient"], group_rows[0]["cycle"]),
            DEFAULT_CYCLE_COLOR,
        )
        for row in group_rows:
            positions.append(cursor)
            group_positions.append(cursor)
            colors.append(group_color)
            timepoint_labels.append(row["timepoint"])
            cursor += 1.0
        group_centers.append((float(np.mean(group_positions)), label))
        cursor += 0.8
        if group_index < len(groups) - 1:
            separators.append(cursor - 0.4)

    def decorate_axis(ax: plt.Axes) -> None:
        ax.set_xlabel("Timepoint de chaque cycle", fontsize=16, labelpad=48)
        ax.set_xticks(positions, timepoint_labels, fontsize=13)
        ax.tick_params(axis="y", labelsize=13)
        ax.grid(axis="y", linestyle="--", alpha=0.30)
        for center, label in group_centers:
            ax.text(
                center,
                -0.20,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=13,
                weight="bold",
            )
        for separator in separators:
            ax.axvline(separator, color="#B7B7B7", linewidth=1.0, linestyle=":")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    GLOBAL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    rates_kcps = [row["rwo_mean_cps"] / 1000.0 for row in rows]
    rate_fig, rate_ax = plt.subplots(figsize=(15.5, 5.8), facecolor="white")
    rate_ax.bar(positions, rates_kcps, color=colors, width=0.72)
    rate_ax.set_ylabel("Taux moyen 4W $R_{Wo}$ (kcps/détecteur)", fontsize=16)
    rate_ax.set_title(
        "Taux de comptage large spectre avant correction du temps mort",
        fontsize=22,
        weight="bold",
        pad=18,
    )
    for x, value in zip(positions, rates_kcps):
        rate_ax.text(
            x,
            value + max(rates_kcps) * 0.025,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )
    decorate_axis(rate_ax)
    rate_fig.subplots_adjust(left=0.09, right=0.985, top=0.86, bottom=0.25)
    rate_fig.savefig(
        GLOBAL_RATE_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(rate_fig)

    losses = [row["loss_percent"] for row in rows]
    loss_fig, loss_ax = plt.subplots(figsize=(15.5, 5.8), facecolor="white")
    loss_ax.bar(positions, losses, color=colors, width=0.72)
    loss_ax.set_ylabel("Perte de comptes estimée (%)", fontsize=16)
    loss_ax.set_title(
        "Fraction estimée de comptes perdus par temps mort",
        fontsize=22,
        weight="bold",
        pad=18,
    )
    for x, value in zip(positions, losses):
        loss_ax.text(
            x,
            value + max(losses) * 0.035,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )
    decorate_axis(loss_ax)
    loss_fig.subplots_adjust(left=0.09, right=0.985, top=0.86, bottom=0.25)
    loss_fig.savefig(
        GLOBAL_LOSS_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(loss_fig)

    fieldnames = [
        "patient",
        "cycle",
        "timepoint",
        "day_offset",
        "datetime",
        "scan_length_mm",
        "duration_seconds",
        "rwo_ap_cps",
        "rwo_pa_cps",
        "rwo_mean_cps",
        "dtcf",
        "loss_percent",
    ]
    with GLOBAL_RATE_CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_report() -> str:
    cycle_label = os.environ.get("FIG_CYCLE", figure_layout.DEFAULT_CYCLE)
    lines = [
        "Planar Lu-177 Dead-Time Correction Report",
        "==========================================",
        "",
        "Method:",
        "  Scatter correction: TEW on geometric mean AP/PA photopeak.",
        "  Dead-time correction: Frezza/Desy paralyzable model.",
        "  DTCF solves: 1 = DTCF * exp(-DTCF * RWo * tau).",
        "  Corrected primary rate: RPt = DTCF * RPo.",
        "  One global DTCF is applied uniformly to the full planar image.",
        "",
        "Parameters:",
        f"  tau = {planar_processing.DEAD_TIME_TAU_US:g} us for the 4W 55-250 keV scheme",
        f"  camera sensitivity = {planar_processing.CAMERA_SENSITIVITY_CPS_PER_MBQ:g} cps/MBq",
        "  RPo = TEW-corrected photopeak count rate.",
        "  RWo = arithmetic mean of raw AP and PA detector count rates before TEW or GM.",
        f"  Wide-window components used: {', '.join(planar_processing.DEAD_TIME_WIDE_WINDOW_ENERGIES)}",
        f"  Experimentally calibrated range ends near {planar_processing.DEAD_TIME_MAX_CALIBRATED_RWO_CPS / 1000:.0f} kcps/detector.",
        "",
        "Interpretation:",
        "  DTCF = multiplicative correction applied to primary counts/images.",
        "  loss % = 100 * (1 - 1/DTCF).",
        "  The correction is spatially global: every pixel receives the same DTCF.",
        "  AP/PA-specific DTCFs are reported only to quantify the approximation made by the global factor.",
        "  activity MBq is only shown when duration and camera sensitivity are available.",
        "",
    ]

    rows_2026 = planar_processing.planar_tew_decay_data(planar_processing.default_planar_study_dir())
    plot_series = {cycle_label.replace("__Studies", ""): rows_2026}
    lines.extend(report_rows(f"Dataset {cycle_label} planar WB", rows_2026))

    plot_dtcf_summary(plot_series)

    lines.extend(
        [
            "Notes:",
            f"  The dead-time correction has a small effect in the {cycle_label} planar data.",
            "  RWo must match the wide-window definition used when tau was calibrated.",
            "  The 0.55 us value in Desy et al. (2022) belongs to the 6W 18-680 keV scheme;",
            "  the present 4W 55-250 keV data require tau = 0.632 us (Frezza et al., 2020, Table 6).",
            "  Reference: Desy et al., EJNMMI Physics 2022;9:54, DOI 10.1186/s40658-022-00484-w.",
            "  Calibration: Frezza et al., EJNMMI Physics 2020;7:10, DOI 10.1186/s40658-020-0275-6.",
            "  These planar values are still not fully attenuation-corrected.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(build_report(), encoding="utf-8")
    global_rows = collect_global_dead_time_rows()
    plot_global_rate_summary(global_rows)
    print(f"Saved: {OUT_PATH}")
    print(f"Saved: {DTCF_FIGURE_PATH}")
    print(f"Saved: {GLOBAL_RATE_FIGURE_PATH}")
    print(f"Saved: {GLOBAL_LOSS_FIGURE_PATH}")
    print(f"Saved: {GLOBAL_RATE_CSV_PATH}")


if __name__ == "__main__":
    main()
