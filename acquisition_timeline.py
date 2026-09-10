"""Audit acquisition dates and start times for paired planar and Q/SPECT studies.

The quantitative PT object stores an AcquisitionTime that coincides with the
start of the last raw TOMO bed in this dataset.  Consequently, this script
reports both:

* the actual whole-body SPECT start, obtained from the earliest raw
  ``TOMO WB-Scan`` series;
* the reference time declared in the reconstructed PT/QSPECT DICOM object.

No activity values are modified by this audit.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pydicom

import planar_processing
import qspect_processing


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "fig" / "acquisition_timeline"
LU177_PHYSICAL_HALF_LIFE_H = 159.5


def _clean_time(value: Any) -> str:
    return str(value or "").strip()


def parse_dicom_date_time(date_value: Any, time_value: Any) -> Optional[datetime]:
    """Parse separate DICOM DA/TM values, preserving fractional seconds."""
    date_text = str(date_value or "").strip()
    time_text = _clean_time(time_value)
    if not date_text or not time_text:
        return None
    main, dot, fraction = time_text.partition(".")
    main = main.ljust(6, "0")[:6]
    text = date_text + main
    fmt = "%Y%m%d%H%M%S"
    if dot and fraction:
        text += "." + fraction[:6].ljust(6, "0")
        fmt += ".%f"
    try:
        return datetime.strptime(text, fmt)
    except ValueError:
        return None


def acquisition_datetime(ds: pydicom.dataset.Dataset) -> Optional[datetime]:
    """Return the DICOM acquisition start using explicit acquisition fields."""
    acquisition_dt = str(getattr(ds, "AcquisitionDateTime", "") or "").strip()
    if acquisition_dt:
        main, dot, fraction = acquisition_dt.partition(".")
        main = main[:14]
        text = main
        fmt = "%Y%m%d%H%M%S"
        if dot and fraction:
            text += "." + fraction[:6].ljust(6, "0")
            fmt += ".%f"
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass

    date_value = (
        getattr(ds, "AcquisitionDate", None)
        or getattr(ds, "SeriesDate", None)
        or getattr(ds, "StudyDate", None)
    )
    return parse_dicom_date_time(date_value, getattr(ds, "AcquisitionTime", None))


def _series_headers(root_dir: Path) -> Iterable[tuple[Path, pydicom.dataset.Dataset]]:
    for series_dir in qspect_processing.find_dicom_series_dirs(root_dir):
        first_file = next(iter(sorted(series_dir.glob("*.dcm"))), None)
        if first_file is None:
            continue
        yield series_dir, pydicom.dcmread(
            first_file, stop_before_pixels=True, force=True
        )


def _minimum_series_acquisition_time(series_dir: Path) -> Optional[datetime]:
    times: List[datetime] = []
    for path in series_dir.glob("*.dcm"):
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        value = acquisition_datetime(ds)
        if value is not None:
            times.append(value)
    return min(times) if times else None


def _bed_number(description: str) -> Optional[int]:
    match = re.search(r"TOMO\s+WB-Scan\s*(\d+)", description, flags=re.IGNORECASE)
    return None if match is None else int(match.group(1))


def _format_datetime(value: Optional[datetime]) -> str:
    return "" if value is None else value.isoformat(sep=" ", timespec="seconds")


def _format_clock(value: Optional[datetime]) -> str:
    return "—" if value is None else value.strftime("%H:%M:%S")


def _minutes_after(reference: Optional[datetime], value: Optional[datetime]) -> float:
    if reference is None or value is None:
        return float("nan")
    return (value - reference).total_seconds() / 60.0


def collect_acquisition_timeline(
    planar_dir: Path = planar_processing.default_planar_study_dir(),
    qspect_dir: Path = qspect_processing.default_qspect_dir(),
) -> List[Dict[str, Any]]:
    """Collect one timing record per reconstructed Q/SPECT timepoint."""
    planar_by_date: Dict[str, List[Dict[str, Any]]] = {}
    for series_dir, ds in _series_headers(Path(planar_dir)):
        description = str(getattr(ds, "SeriesDescription", ""))
        if str(getattr(ds, "Modality", "")) != "NM" or "WB RAPIDE" not in description.upper():
            continue
        start = _minimum_series_acquisition_time(series_dir)
        if start is not None:
            planar_by_date.setdefault(start.strftime("%Y%m%d"), []).append(
                {"start": start, "series_dir": series_dir, "description": description}
            )

    beds_by_date: Dict[str, List[Dict[str, Any]]] = {}
    ct_by_date: Dict[str, List[Dict[str, Any]]] = {}
    pt_headers: Dict[Path, pydicom.dataset.Dataset] = {}
    for series_dir, ds in _series_headers(Path(qspect_dir)):
        modality = str(getattr(ds, "Modality", ""))
        description = str(getattr(ds, "SeriesDescription", ""))
        start = _minimum_series_acquisition_time(series_dir)
        date_key = str(
            getattr(ds, "AcquisitionDate", None)
            or getattr(ds, "SeriesDate", None)
            or getattr(ds, "StudyDate", "")
        )
        bed = _bed_number(description)
        if modality == "NM" and bed is not None and start is not None:
            beds_by_date.setdefault(date_key, []).append(
                {"bed": bed, "start": start, "description": description}
            )
        elif modality == "CT" and description.strip().upper() == "ACCT" and start is not None:
            ct_by_date.setdefault(date_key, []).append(
                {"start": start, "description": description, "series_dir": series_dir}
            )
        elif modality == "PT" and "QSPECT" in description.upper():
            pt_headers[series_dir.resolve()] = ds

    qspect_series = qspect_processing.load_qspect_study(Path(qspect_dir))
    rows: List[Dict[str, Any]] = []
    for index, qspect in enumerate(qspect_series):
        pt_start = qspect.get("acquisition_datetime")
        date_key = (
            pt_start.strftime("%Y%m%d")
            if isinstance(pt_start, datetime)
            else str(qspect.get("study_date", ""))
        )
        planars = sorted(planar_by_date.get(date_key, []), key=lambda item: item["start"])
        beds = sorted(beds_by_date.get(date_key, []), key=lambda item: item["bed"])
        cts = sorted(ct_by_date.get(date_key, []), key=lambda item: item["start"])
        planar_start = planars[0]["start"] if planars else None
        qspect_wb_start = min((item["start"] for item in beds), default=None)
        ct_start = cts[0]["start"] if cts else None
        pt_ds = pt_headers.get(Path(qspect["series_dir"]).resolve())
        pt_content = None
        if pt_ds is not None:
            content_date = getattr(pt_ds, "ContentDate", None) or getattr(pt_ds, "StudyDate", None)
            pt_content = parse_dicom_date_time(content_date, getattr(pt_ds, "ContentTime", None))

        bed_times = {item["bed"]: item["start"] for item in beds}
        delay_h = _minutes_after(planar_start, qspect_wb_start) / 60.0
        physical_adjustment_percent = (
            float("nan")
            if not np.isfinite(delay_h)
            else 100.0 * (2.0 ** (delay_h / LU177_PHYSICAL_HALF_LIFE_H) - 1.0)
        )
        rows.append(
            {
                "label": str(qspect.get("label", f"Timepoint {index + 1}")),
                "date": "" if pt_start is None else pt_start.strftime("%Y-%m-%d"),
                "planar_start": planar_start,
                "qspect_wb_start": qspect_wb_start,
                "qspect_bed1_start": bed_times.get(1),
                "qspect_bed2_start": bed_times.get(2),
                "qspect_bed3_start": bed_times.get(3),
                "qspect_pt_reference_time": pt_start,
                "qspect_pt_content_time": pt_content,
                "ct_start": ct_start,
                "qspect_start_minus_planar_min": _minutes_after(planar_start, qspect_wb_start),
                "pt_reference_minus_planar_min": _minutes_after(planar_start, pt_start),
                "ct_minus_planar_min": _minutes_after(planar_start, ct_start),
                "physical_adjustment_qspect_start_to_planar_percent": physical_adjustment_percent,
                "pt_decay_correction": qspect.get("decay_correction"),
                "pt_corrected_image": "\\".join(str(v) for v in qspect.get("corrected_image", [])),
            }
        )
    return rows


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    fields = list(rows[0].keys()) if rows else []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _format_datetime(value) if isinstance(value, datetime) else value
                    for key, value in row.items()
                }
            )


def write_report(rows: List[Dict[str, Any]], output_path: Path) -> None:
    lines = [
        "Audit des heures d'acquisition planaire et Q/SPECT",
        "=================================================",
        "",
        "Définition des heures:",
        "  - Planaire: premier AcquisitionTime de la série NM WB RAPIDE.",
        "  - Début Q/SPECT WB: premier AcquisitionTime des séries NM TOMO WB-Scan.",
        "  - Référence PT: AcquisitionTime du volume PT/QSPECT reconstruit.",
        "  - CT: premier AcquisitionTime de la série CT ACCT originale.",
        "",
        "Le temps PT reconstruit coïncide ici avec le début du troisième lit TOMO;",
        "il ne doit donc pas être présenté comme le début de tout le Q/SPECT WB.",
        "",
        "Jour | Date       | Planaire | Q/SPECT WB | Lit 2    | Lit 3    | Réf. PT  | CT       | Q/SPECT-plan. | Ajust. phys.",
        "-----|------------|----------|------------|----------|----------|----------|----------|---------------|-------------",
    ]
    for row in rows:
        lines.append(
            f"{row['label']:4s} | {row['date']:10s} | "
            f"{_format_clock(row['planar_start']):8s} | "
            f"{_format_clock(row['qspect_wb_start']):10s} | "
            f"{_format_clock(row['qspect_bed2_start']):8s} | "
            f"{_format_clock(row['qspect_bed3_start']):8s} | "
            f"{_format_clock(row['qspect_pt_reference_time']):8s} | "
            f"{_format_clock(row['ct_start']):8s} | "
            f"{row['qspect_start_minus_planar_min']:10.1f} min | "
            f"{row['physical_adjustment_qspect_start_to_planar_percent']:+8.3f} %"
        )
    lines.extend(
        [
            "",
            "L'ajustement physique est le pourcentage nécessaire pour ramener une",
            "activité mesurée au début du Q/SPECT vers l'heure antérieure du planaire.",
            "Il n'inclut aucune clairance biologique.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_timeline(rows: List[Dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    y_values = np.arange(len(rows))[::-1]
    styles = [
        ("planar_start", "Planaire WB", "o", "tab:blue"),
        ("qspect_wb_start", "Début Q/SPECT WB (lit 1 brut)", "D", "tab:green"),
        ("qspect_pt_reference_time", "Référence du PT reconstruit", "s", "tab:orange"),
        ("ct_start", "Début CT ACCT", "^", "tab:red"),
    ]
    for y, row in zip(y_values, rows):
        planar = row["planar_start"]
        available = []
        for key, label, marker, color in styles:
            value = row[key]
            delay = _minutes_after(planar, value)
            if not np.isfinite(delay):
                continue
            available.append(delay)
            ax.scatter(
                delay,
                y,
                s=90,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
                label=label if y == y_values[0] else None,
            )
            vertical_offset = 10 if key in {"planar_start", "qspect_pt_reference_time"} else -17
            ax.annotate(
                _format_clock(value),
                (delay, y),
                xytext=(0, vertical_offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if vertical_offset > 0 else "top",
                fontsize=8,
                color=color,
            )
        if available:
            ax.hlines(y, min(available), max(available), color="0.78", linewidth=1.2, zorder=1)

    ax.axvline(0.0, color="tab:blue", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_yticks(y_values)
    ax.set_yticklabels([f"{row['label']} — {row['date']}" for row in rows])
    ax.set_xlabel("Minutes après le début de l’acquisition planaire")
    ax.set_title("Chronologie DICOM des acquisitions planaires, Q/SPECT et CT")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", frameon=False)
    ax.set_ylim(-0.7, len(rows) - 0.15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(
    planar_dir: Path,
    qspect_dir: Path,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    rows = collect_acquisition_timeline(planar_dir, qspect_dir)
    if not rows:
        raise ValueError("Aucun timepoint Q/SPECT trouvé")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_dir / "acquisition_start_times.csv")
    write_report(rows, output_dir / "acquisition_timeline_report.txt")
    plot_timeline(rows, output_dir / "acquisition_start_timeline.png")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planar-dir", type=Path, default=planar_processing.default_planar_study_dir())
    parser.add_argument("--qspect-dir", type=Path, default=qspect_processing.default_qspect_dir())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    rows = run(args.planar_dir, args.qspect_dir, args.output_dir)
    for row in rows:
        print(
            f"{row['label']}: planaire {_format_clock(row['planar_start'])}, "
            f"Q/SPECT WB {_format_clock(row['qspect_wb_start'])}, "
            f"référence PT {_format_clock(row['qspect_pt_reference_time'])}, "
            f"écart {row['qspect_start_minus_planar_min']:.1f} min"
        )


if __name__ == "__main__":
    main()
