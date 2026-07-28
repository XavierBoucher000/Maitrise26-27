from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import pydicom


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "2026-05_studies"
QSPECT_ROOT = DATA_ROOT / "2026-05__Studies"
PLANAR_ROOT = DATA_ROOT / "2026-05__Studies_WBP"
OUT_DIR = ROOT / "fig" / "metadata"


def first_dicom_file(series_dir: Path) -> Optional[Path]:
    return next(iter(sorted(series_dir.glob("*.dcm"))), None)


def read_header(path: Path) -> pydicom.dataset.FileDataset:
    return pydicom.dcmread(path, stop_before_pixels=True, force=True)


def series_dirs(root_dir: Path) -> Iterable[Path]:
    if not root_dir.exists():
        return []
    return sorted(path for path in root_dir.iterdir() if path.is_dir() and first_dicom_file(path) is not None)


def find_series(
    root_dir: Path,
    predicate: Callable[[Path, pydicom.dataset.FileDataset], bool],
) -> Optional[tuple[Path, Path, pydicom.dataset.FileDataset]]:
    for series_dir in series_dirs(root_dir):
        dicom_path = first_dicom_file(series_dir)
        if dicom_path is None:
            continue
        try:
            header = read_header(dicom_path)
        except Exception:
            continue
        if predicate(series_dir, header):
            return series_dir, dicom_path, header
    return None


def modality(header: pydicom.dataset.FileDataset) -> str:
    return str(getattr(header, "Modality", "") or "")


def description(header: pydicom.dataset.FileDataset) -> str:
    return str(getattr(header, "SeriesDescription", "") or "")


def save_header_dump(name: str, series_dir: Path, dicom_path: Path, header: pydicom.dataset.FileDataset) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUT_DIR / f"{name}.txt"
    lines = [
        f"Header dump: {name}",
        "=" * (13 + len(name)),
        "",
        f"Series directory: {series_dir}",
        f"DICOM file: {dicom_path}",
        "",
        "Note: PixelData is intentionally not loaded/printed. This is the complete readable DICOM header.",
        "",
        str(header),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def representative_cases() -> Dict[str, Optional[tuple[Path, Path, pydicom.dataset.FileDataset]]]:
    return {
        "planar_wb_rapide_nm_header": find_series(
            PLANAR_ROOT,
            lambda _series_dir, header: modality(header) == "NM",
        ),
        "qspect_pt_bqml_header": find_series(
            QSPECT_ROOT,
            lambda _series_dir, header: modality(header) == "PT" and "QSPECT" in description(header).upper(),
        ),
        "spect_reconstructed_nm_header": find_series(
            QSPECT_ROOT,
            lambda _series_dir, header: modality(header) == "NM" and description(header).upper().startswith("SPECT"),
        ),
        "spect_tomo_raw_nm_header": find_series(
            QSPECT_ROOT,
            lambda _series_dir, header: modality(header) == "NM" and "TOMO" in description(header).upper(),
        ),
        "ct_header": find_series(
            QSPECT_ROOT,
            lambda _series_dir, header: modality(header) == "CT",
        ),
    }


def main() -> None:
    saved_paths = []
    missing = []
    for name, match in representative_cases().items():
        if match is None:
            missing.append(name)
            continue
        series_dir, dicom_path, header = match
        saved_paths.append(save_header_dump(name, series_dir, dicom_path, header))

    index_lines = [
        "DICOM Metadata Header Dumps",
        "===========================",
        "",
        "These files are full pydicom header prints for representative data types.",
        "PixelData is not loaded or printed, so the files stay readable.",
        "",
        "Saved headers:",
    ]
    index_lines.extend(f"- {path.name}" for path in saved_paths)
    if missing:
        index_lines.extend(["", "Missing representative cases:"])
        index_lines.extend(f"- {name}" for name in missing)
    index_path = OUT_DIR / "README.txt"
    index_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    for path in saved_paths:
        print(f"Saved: {path}")
    print(f"Saved: {index_path}")


if __name__ == "__main__":
    main()
