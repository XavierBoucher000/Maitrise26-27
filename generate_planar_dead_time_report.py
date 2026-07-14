from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import planar_processing


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "fig" / "dead_time"
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
            "day | DTCF | label | TEW counts before DT | TEW counts after DT | "
            "loss % | wide cps | duration s | corrected cps | activity MBq"
        ),
    ]
    for row in rows:
        corrected_counts = row.get("tew_corrected_counts")
        duration = row.get("duration_seconds")
        corrected_cps = None if corrected_counts is None or duration in (None, 0) else corrected_counts / duration
        lines.append(
            " | ".join(
                [
                    fmt(row.get("day_offset"), 2),
                    fmt(row.get("dead_time_dtcf"), 4),
                    str(row.get("label", "")),
                    fmt(row.get("tew_corrected_counts_before_dead_time"), 1),
                    fmt(corrected_counts, 1),
                    fmt(row.get("dead_time_loss_percent"), 2),
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
        rwo = [row.get("wide_spectrum_cps") for row in rows]
        axes[0].plot(day, dtcf, marker=marker, linewidth=2.0, label=label)
        axes[1].plot(rwo, dtcf, marker=marker, linewidth=0, markersize=7, label=label)

    axes[0].set_xlabel("Day")
    axes[0].set_ylabel("DTCF")
    axes[0].set_title("Dead-time correction factor over time")
    axes[0].grid(True, linestyle="--", alpha=0.35)

    axes[1].set_xlabel("Wide-spectrum rate RWo (cps)")
    axes[1].set_ylabel("DTCF")
    axes[1].set_title("DTCF increases with wide-window count rate")
    axes[1].grid(True, linestyle="--", alpha=0.35)

    for ax in axes:
        ax.legend(frameon=False, fontsize=9)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.tight_layout()
    DTCF_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DTCF_FIGURE_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(DTCF_FIGURE_PATH.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_report() -> str:
    lines = [
        "Planar Lu-177 Dead-Time Correction Report",
        "==========================================",
        "",
        "Method:",
        "  Scatter correction: TEW on geometric mean AP/PA photopeak.",
        "  Dead-time correction: Frezza paralyzable model.",
        "  DTCF solves: 1 = DTCF * exp(-DTCF * RWo * tau).",
        "  Corrected primary rate: RPt = DTCF * RPo.",
        "",
        "Parameters:",
        f"  tau = {planar_processing.DEAD_TIME_TAU_US:g} us",
        f"  camera sensitivity = {planar_processing.CAMERA_SENSITIVITY_CPS_PER_MBQ:g} cps/MBq",
        "  RPo = TEW-corrected photopeak count rate.",
        "  RWo = observed wide-spectrum count rate estimated from available energy windows.",
        f"  Wide-window components used: {', '.join(planar_processing.DEAD_TIME_WIDE_WINDOW_ENERGIES)}",
        "",
        "Interpretation:",
        "  DTCF = multiplicative correction applied to primary counts/images.",
        "  loss % = 100 * (1 - 1/DTCF).",
        "  activity MBq is only shown when duration and camera sensitivity are available.",
        "",
    ]

    rows_2026 = planar_processing.planar_tew_decay_data(planar_processing.default_planar_study_dir())
    plot_series = {"2026-05": rows_2026}
    lines.extend(report_rows("Dataset 2026-05 planar WBP", rows_2026))

    plot_dtcf_summary(plot_series)

    lines.extend(
        [
            "Notes:",
            "  The dead-time correction has a small effect in the 2026-05 planar data.",
            "  RWo must match the wide-window definition used when tau was calibrated.",
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
    print(f"Saved: {DTCF_FIGURE_PATH.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
