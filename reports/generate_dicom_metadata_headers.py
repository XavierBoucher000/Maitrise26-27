from pathlib import Path
import re
import sys
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional

import pydicom


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DATA_DIR = ROOT / "Data"
FIG_ROOT = ROOT / "fig" / "metadonné"

# Un dossier de cycle "QSPECT" ressemble a "2026-06__Studies" (jamais "_WBP").
# Le dossier planar correspondant est le meme nom + "_WBP".
CYCLE_DIR_RE = re.compile(r"^.+__Studies(-\d+)?$")


class Cycle(NamedTuple):
    name: str            # nom du dossier de cycle, ex: "2026-06__Studies-2"
    qspect_root: Path    # .../patient 2/2026-06__Studies-2
    planar_root: Path    # .../patient 2/2026-06__Studies-2_WBP


def patient_dirs(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        return []
    return sorted(p for p in data_dir.iterdir() if p.is_dir())


def cycles_for_patient(patient_dir: Path) -> List[Cycle]:
    cycles = []
    for sub in sorted(patient_dir.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name.endswith("_WBP"):
            continue
        if not CYCLE_DIR_RE.match(sub.name):
            continue
        planar_dir = patient_dir / f"{sub.name}_WBP"
        cycles.append(Cycle(name=sub.name, qspect_root=sub, planar_root=planar_dir))
    return cycles


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


def save_header_dump(out_dir: Path, name: str, series_dir: Path, dicom_path: Path, header: pydicom.dataset.FileDataset) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{name}.txt"
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


def representative_cases(qspect_root: Path, planar_root: Path) -> Dict[str, Optional[tuple[Path, Path, pydicom.dataset.FileDataset]]]:
    return {
        "planar_wb_rapide_nm_header": find_series(
            planar_root,
            lambda _series_dir, header: modality(header) == "NM",
        ),
        "qspect_pt_bqml_header": find_series(
            qspect_root,
            lambda _series_dir, header: modality(header) == "PT" and "QSPECT" in description(header).upper(),
        ),
        "spect_reconstructed_nm_header": find_series(
            qspect_root,
            lambda _series_dir, header: modality(header) == "NM" and description(header).upper().startswith("SPECT"),
        ),
        "spect_tomo_raw_nm_header": find_series(
            qspect_root,
            lambda _series_dir, header: modality(header) == "NM" and "TOMO" in description(header).upper(),
        ),
        "ct_header": find_series(
            qspect_root,
            lambda _series_dir, header: modality(header) == "CT",
        ),
    }


def process_cycle(patient_name: str, cycle: Cycle) -> None:
    out_dir = FIG_ROOT / patient_name / cycle.name

    saved_paths = []
    missing = []
    for name, match in representative_cases(cycle.qspect_root, cycle.planar_root).items():
        if match is None:
            missing.append(name)
            continue
        series_dir, dicom_path, header = match
        saved_paths.append(save_header_dump(out_dir, name, series_dir, dicom_path, header))

    index_lines = [
        "DICOM Metadata Header Dumps",
        "===========================",
        "",
        f"Patient: {patient_name}",
        f"Cycle: {cycle.name}",
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
    index_path = out_dir / "README.txt"
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    for path in saved_paths:
        print(f"Saved: {path}")
    print(f"Saved: {index_path}")


def main() -> None:
    for patient_dir in patient_dirs(DATA_DIR):
        patient_name = patient_dir.name
        cycles = cycles_for_patient(patient_dir)
        if not cycles:
            print(f"No cycle found for {patient_name}, skipped.")
            continue
        for cycle in cycles:
            print(f"--- {patient_name} / {cycle.name} ---")
            process_cycle(patient_name, cycle)


if __name__ == "__main__":
    main()