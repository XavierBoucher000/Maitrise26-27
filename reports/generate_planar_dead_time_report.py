from pathlib import Path
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
import planar_processing


ROOT = PROJECT_ROOT
OUT_DIR = figure_layout.patient_cycle_dir(__file__)
OUT_PATH = OUT_DIR / "planar_dead_time_correction_report.txt"
DTCF_FIGURE_PATH = OUT_DIR / "planar_dead_time_dtcf_summary.png"


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
    print(f"Saved: {OUT_PATH}")
    print(f"Saved: {DTCF_FIGURE_PATH}")


if __name__ == "__main__":
    main()
