"""Write an auditable list of exact duplicate DICOM series."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dicom_series_dedup


def find_series_dirs(data_dir: Path):
    """Return every leaf directory containing DICOM files."""
    root = Path(data_dir).expanduser().resolve()
    if any(root.glob("*.dcm")):
        return [root]
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_dir() and any(path.glob("*.dcm"))
    )


def write_report(data_dirs, output_path: Path) -> Path:
    if isinstance(data_dirs, (str, Path)):
        data_dirs = [Path(data_dirs)]
    series_dirs = []
    for data_dir in data_dirs:
        series_dirs.extend(find_series_dirs(Path(data_dir)))
    _kept, duplicates = dicom_series_dedup.deduplicate_series_dirs(series_dirs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        dicom_series_dedup.format_duplicate_report(duplicates), encoding="utf-8"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path, nargs="+")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "fig/quality_control/dicom_duplicates/shared/"
            "exact_dicom_duplicates.txt"
        ),
    )
    args = parser.parse_args()
    print(write_report(args.data_dir, args.output))


if __name__ == "__main__":
    main()
