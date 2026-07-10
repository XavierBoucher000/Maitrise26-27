import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

import planar_processing
import qspect_processing

AUTO_CLOSE_FIGURES = True
AUTO_CLOSE_DELAY_SECONDS = 0.1


def enable_auto_close_figures() -> None:
    original_show = plt.show

    def show_and_close(*args, **kwargs):
        kwargs["block"] = False
        original_show(*args, **kwargs)
        for figure_number in plt.get_fignums():
            plt.figure(figure_number).canvas.flush_events()
        time.sleep(AUTO_CLOSE_DELAY_SECONDS)
        plt.close("all")

    plt.show = show_and_close


def first_dicom_file(path: Path) -> Path | None:
    if path.is_file() and path.suffix.lower() == ".dcm":
        return path
    if any(path.glob("*.dcm")):
        return next(path.glob("*.dcm"))
    return next(path.rglob("*.dcm"), None)


def looks_like_qspect_folder(path: Path) -> bool:
    return bool(qspect_processing.find_qspect_series_dirs(path))


def contains_direct_dicom_files(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.dcm"))


def resolve_default_dataset(path: Path) -> Path:
    planar_subfolder = path / "2026-05__Studies_WBP"
    if planar_subfolder.exists():
        return planar_subfolder
    return path


def main() -> None:
    if AUTO_CLOSE_FIGURES:
        enable_auto_close_figures()

    scan_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else planar_processing.default_planar_study_dir()
    scan_dir = resolve_default_dataset(scan_dir.expanduser().resolve())
    if not scan_dir.exists():
        raise FileNotFoundError(f"Scan folder not found: {scan_dir}")

    if looks_like_qspect_folder(scan_dir):
        qspect_processing.run_qspect_workflow(scan_dir)
    elif contains_direct_dicom_files(scan_dir):
        planar_processing.run_planar_workflow(scan_dir)
    else:
        planar_processing.run_planar_study_workflow(scan_dir)


if __name__ == "__main__":
    main()
