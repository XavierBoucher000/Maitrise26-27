"""Discover patient/cycle datasets from the canonical ``Data`` tree."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import pydicom

import dicom_series_dedup
import figure_layout
import qspect_processing


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "Data"
PATIENT_RE = re.compile(r"^p\d+$", re.IGNORECASE)
CYCLE_RE = re.compile(r"^\d{4}-\d{2}__Studies(?:-\d+)?$")


class CycleDataset(NamedTuple):
    patient: str
    cycle: str
    qspect_dir: Path
    planar_dir: Path


def discover_cycles(data_dir: Path = DATA_DIR) -> List[CycleDataset]:
    """Return every cycle folder with its matching ``_WBP`` planar folder."""
    cycles: List[CycleDataset] = []
    for patient_dir in sorted(Path(data_dir).iterdir()):
        if not patient_dir.is_dir() or PATIENT_RE.fullmatch(patient_dir.name) is None:
            continue
        for qspect_dir in sorted(patient_dir.iterdir()):
            if not qspect_dir.is_dir() or CYCLE_RE.fullmatch(qspect_dir.name) is None:
                continue
            separate_planar_dir = patient_dir / f"{qspect_dir.name}_WBP"
            cycles.append(
                CycleDataset(
                    patient=patient_dir.name,
                    cycle=qspect_dir.name,
                    qspect_dir=qspect_dir,
                    # New exports keep WB RAPIDE in the cycle folder itself.
                    # Retain compatibility with the older separate _WBP tree.
                    planar_dir=(
                        separate_planar_dir
                        if separate_planar_dir.exists()
                        else qspect_dir
                    ),
                )
            )
    return sorted(
        cycles,
        key=lambda item: figure_layout.patient_cycle_sort_key(
            item.patient, item.cycle
        ),
    )


def _first_dicom(series_dir: Path) -> Optional[Path]:
    return next(iter(sorted(series_dir.glob("*.dcm"))), None)


def cycle_inventory(dataset: CycleDataset) -> Dict[str, object]:
    """Summarize primary planar and quantitative Q/SPECT availability."""
    planar_count = 0
    qspect_complete = 0
    qspect_incomplete = 0
    qspect_details = []
    if dataset.planar_dir.exists():
        planar_candidates = []
        for series_dir in dataset.planar_dir.iterdir():
            dicom_path = _first_dicom(series_dir) if series_dir.is_dir() else None
            if dicom_path is None:
                continue
            ds = pydicom.dcmread(dicom_path, stop_before_pixels=True, force=True)
            if str(getattr(ds, "SeriesDescription", "")).strip().upper() == "WB RAPIDE":
                scan_length = getattr(ds, "ScanLength", None)
                try:
                    scan_length = float(scan_length)
                except (TypeError, ValueError):
                    scan_length = None
                planar_candidates.append((series_dir, scan_length))
        planar_unique, _planar_duplicates = (
            dicom_series_dedup.deduplicate_series_dirs(
                [series_dir for series_dir, _scan_length in planar_candidates]
            )
        )
        length_by_dir = dict(planar_candidates)
        unique_lengths = [length_by_dir.get(series_dir) for series_dir in planar_unique]
        known_lengths = [value for value in unique_lengths if value is not None]
        if planar_unique and len(known_lengths) == len(planar_unique):
            minimum_wb_length = 0.8 * max(known_lengths)
            planar_unique = [
                series_dir
                for series_dir in planar_unique
                if length_by_dir[series_dir] >= minimum_wb_length
            ]
        planar_count = len(planar_unique)

    if dataset.qspect_dir.exists():
        qspect_unique = qspect_processing.find_qspect_series_dirs(
            dataset.qspect_dir
        )
        for series_dir in qspect_unique:
            dicom_path = _first_dicom(series_dir)
            ds = pydicom.dcmread(dicom_path, stop_before_pixels=True, force=True)
            description = str(getattr(ds, "SeriesDescription", "")).strip()
            actual = len(list(series_dir.glob("*.dcm")))
            match = re.search(r"_n(\d+)__", series_dir.name)
            expected = None if match is None else int(match.group(1))
            complete = expected is None or actual == expected
            qspect_complete += int(complete)
            qspect_incomplete += int(not complete)
            qspect_details.append(
                {
                    "label": description,
                    "actual_files": actual,
                    "expected_files": expected,
                    "complete": complete,
                }
            )

    return {
        "patient": dataset.patient,
        "cycle": dataset.cycle,
        "planar_dir_exists": dataset.planar_dir.exists(),
        "planar_timepoints": planar_count,
        "qspect_complete_timepoints": qspect_complete,
        "qspect_incomplete_timepoints": qspect_incomplete,
        "qspect_details": qspect_details,
        "complete_for_full_pipeline": (
            planar_count >= 2
            and qspect_incomplete == 0
            and planar_count == qspect_complete
        ),
    }
