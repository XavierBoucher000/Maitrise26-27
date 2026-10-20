"""Detect exact duplicate DICOM series without deleting source data."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pydicom


@dataclass(frozen=True)
class DuplicateSeries:
    kept: Path
    excluded: Path
    pixel_sha256: str
    file_count: int


def _header(path: Path):
    return pydicom.dcmread(path, stop_before_pixels=True, force=True)


def _sort_key(path: Path) -> tuple:
    ds = _header(path)
    position = getattr(ds, "ImagePositionPatient", None)
    if position is not None and len(position) >= 3:
        try:
            return (0, float(position[2]), path.name)
        except (TypeError, ValueError):
            pass
    try:
        return (1, int(getattr(ds, "InstanceNumber")), path.name)
    except (TypeError, ValueError):
        return (2, path.name)


def series_pixel_sha256(series_dir: Path) -> str:
    """Hash the ordered, encoded PixelData of every file in one series."""
    files = sorted(Path(series_dir).glob("*.dcm"), key=_sort_key)
    if not files:
        raise ValueError(f"No DICOM files found in {series_dir}")
    digest = hashlib.sha256()
    for path in files:
        ds = pydicom.dcmread(path, force=True)
        pixel_data = getattr(ds, "PixelData", None)
        if pixel_data is None:
            raise ValueError(f"PixelData is missing from {path}")
        # Boundaries make [b'ab', b'c'] distinct from [b'a', b'bc'].
        digest.update(len(pixel_data).to_bytes(8, "little"))
        digest.update(pixel_data)
    return digest.hexdigest()


def _identity_metadata(series_dir: Path) -> tuple:
    first = next(iter(sorted(Path(series_dir).glob("*.dcm"))), None)
    if first is None:
        raise ValueError(f"No DICOM files found in {series_dir}")
    ds = _header(first)
    return (
        str(getattr(ds, "Modality", "")).strip().upper(),
        str(getattr(ds, "SeriesDescription", "")).strip().upper(),
        str(getattr(ds, "StudyDate", "")).strip(),
        len(list(Path(series_dir).glob("*.dcm"))),
    )


def deduplicate_series_dirs(
    series_dirs: Iterable[Path],
) -> Tuple[List[Path], List[DuplicateSeries]]:
    """Keep one representative of each exact same-acquisition pixel copy.

    The source folders are never modified. Series are grouped only when their
    modality, description, study date, file count and complete ordered
    PixelData hash all agree. Acquisition time is deliberately not required:
    duplicate exports can alter header times while preserving every pixel.
    """
    ordered = sorted((Path(path) for path in series_dirs), key=lambda p: str(p))
    metadata_by_dir = {series_dir: _identity_metadata(series_dir) for series_dir in ordered}
    metadata_counts = {}
    for metadata in metadata_by_dir.values():
        metadata_counts[metadata] = metadata_counts.get(metadata, 0) + 1

    kept: List[Path] = []
    duplicates: List[DuplicateSeries] = []
    seen = {}
    for series_dir in ordered:
        metadata = metadata_by_dir[series_dir]
        # A unique metadata group cannot have an exact same-acquisition copy,
        # so avoid reading its potentially large PixelData unnecessarily.
        if metadata_counts[metadata] == 1:
            kept.append(series_dir)
            continue
        pixel_hash = series_pixel_sha256(series_dir)
        key = metadata + (pixel_hash,)
        representative = seen.get(key)
        if representative is None:
            seen[key] = series_dir
            kept.append(series_dir)
        else:
            duplicates.append(
                DuplicateSeries(
                    kept=representative,
                    excluded=series_dir,
                    pixel_sha256=pixel_hash,
                    file_count=metadata[-1],
                )
            )
    return kept, duplicates


def format_duplicate_report(duplicates: Sequence[DuplicateSeries]) -> str:
    lines = [
        "Exact DICOM duplicate-series filtering",
        "======================================",
        "Source files were not deleted or moved.",
        "",
    ]
    if not duplicates:
        lines.append("No exact duplicate series detected.")
        return "\n".join(lines) + "\n"
    for item in duplicates:
        lines.extend(
            [
                f"KEPT:     {item.kept}",
                f"EXCLUDED: {item.excluded}",
                f"FILES:    {item.file_count}",
                f"SHA256:   {item.pixel_sha256}",
                "",
            ]
        )
    return "\n".join(lines)
