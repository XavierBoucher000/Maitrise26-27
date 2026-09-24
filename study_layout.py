"""Discover patient/cycle datasets from the canonical ``Data`` tree."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import pydicom


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
            cycles.append(
                CycleDataset(
                    patient=patient_dir.name,
                    cycle=qspect_dir.name,
                    qspect_dir=qspect_dir,
                    planar_dir=patient_dir / f"{qspect_dir.name}_WBP",
                )
            )
    return cycles


def _first_dicom(series_dir: Path) -> Optional[Path]:
    return next(iter(sorted(series_dir.glob("*.dcm"))), None)


def cycle_inventory(dataset: CycleDataset) -> Dict[str, object]:
    """Summarize primary planar and quantitative Q/SPECT availability."""
    planar_count = 0
    qspect_complete = 0
    qspect_incomplete = 0
    qspect_details = []
    if dataset.planar_dir.exists():
        for series_dir in dataset.planar_dir.iterdir():
            dicom_path = _first_dicom(series_dir) if series_dir.is_dir() else None
            if dicom_path is None:
                continue
            ds = pydicom.dcmread(dicom_path, stop_before_pixels=True, force=True)
            if str(getattr(ds, "SeriesDescription", "")).strip().upper() == "WB RAPIDE":
                planar_count += 1

    if dataset.qspect_dir.exists():
        for series_dir in dataset.qspect_dir.iterdir():
            dicom_path = _first_dicom(series_dir) if series_dir.is_dir() else None
            if dicom_path is None:
                continue
            ds = pydicom.dcmread(dicom_path, stop_before_pixels=True, force=True)
            description = str(getattr(ds, "SeriesDescription", "")).strip()
            modality = str(getattr(ds, "Modality", "")).strip()
            if modality != "PT" or re.fullmatch(
                r"WB\s+QSPECT\s+Day\d+", description, re.IGNORECASE
            ) is None:
                continue
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
