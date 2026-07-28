from pathlib import Path
import re
import sys
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pydicom

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dicom_loader
import planar_processing
import qspect_processing


SELECTED_DAY = "Day0"
OUTPUT_DIR = Path("fig") / "imaging"
OUTPUT_FILE = "planar_tew_corrected.png"
QSPECT_CENTER_FILE = "qspect_center_slices.png"
QSPECT_MAX_FILE = "qspect_max_slices.png"
CT_CENTER_FILE = "ct_center_slices.png"
QSPECT_FLIP_VERTICAL = True


def day_index(day_label: str) -> int:
    match = re.search(r"day\s*(\d+)", day_label, re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid day label: {day_label}")
    return int(match.group(1))


def sorted_planar_scan_dirs(study_dir: Path) -> List[Path]:
    scan_dirs = dicom_loader.find_scan_dirs(study_dir)
    dated_scans = []
    for scan_dir in scan_dirs:
        images = dicom_loader.load_scan_directory(scan_dir)
        scan = {"scan_name": scan_dir.name, "scan_path": scan_dir, "images": images}
        dated_scans.append((dicom_loader.scan_datetime(scan), scan_dir))
    return [scan_dir for _dt, scan_dir in sorted(dated_scans, key=lambda item: (item[0] is None, item[0], item[1].name))]


def load_planar_day(study_dir: Path, selected_day: str) -> Dict[str, Any]:
    scan_dirs = sorted_planar_scan_dirs(study_dir)
    index = day_index(selected_day)
    if index >= len(scan_dirs):
        raise ValueError(f"{selected_day} not available in planar study. Found {len(scan_dirs)} scans.")

    scan_dir = scan_dirs[index]
    images = dicom_loader.load_scan_directory(scan_dir)
    geometric_images = planar_processing.geometric_mean_images(images)
    correction = planar_processing.apply_tew_correction(geometric_images)
    by_energy = {item["energy_window"]: item for item in geometric_images}
    return {
        "scan_dir": scan_dir,
        "geometric_by_energy": by_energy,
        "correction": correction,
    }


def load_qspect_day(qspect_dir: Path, selected_day: str) -> Dict[str, Any]:
    series = qspect_processing.load_qspect_study(qspect_dir)
    return qspect_processing.selected_series(series, selected_day)


def is_ct_series_dir(series_dir: Path) -> bool:
    first_file = next(series_dir.glob("*.dcm"), None)
    if first_file is None:
        return False
    ds = pydicom.dcmread(first_file, stop_before_pixels=True, force=True)
    return str(getattr(ds, "Modality", "")) == "CT"


def find_ct_series_dirs(root_dir: Path) -> List[Path]:
    return [
        series_dir
        for series_dir in qspect_processing.find_dicom_series_dirs(root_dir)
        if is_ct_series_dir(series_dir)
    ]


def ct_priority(series_description: str) -> int:
    description = series_description.upper()
    if "ACCT" in description and "TRANSFORMED" in description:
        return 0
    if "ACCT" in description:
        return 1
    if description == "CT":
        return 2
    if "TOPOGRAM" in description:
        return 99
    return 10


def ct_series_datetime(series_dir: Path) -> Any:
    first_file = next(series_dir.glob("*.dcm"), None)
    if first_file is None:
        return None
    ds = pydicom.dcmread(first_file, stop_before_pixels=True, force=True)
    return qspect_processing.parse_dicom_datetime(
        getattr(ds, "StudyDate", None),
        getattr(ds, "AcquisitionTime", None),
    )


def load_ct_series(series_dir: Path) -> Dict[str, Any]:
    files = sorted(series_dir.glob("*.dcm"), key=qspect_processing.sort_key_for_slice)
    if not files:
        raise ValueError(f"No DICOM files found in {series_dir}")

    slices = []
    first_ds = None
    for path in files:
        ds = pydicom.dcmread(path, force=True)
        if first_ds is None:
            first_ds = ds
        image = ds.pixel_array.astype(np.float64)
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        slices.append(image * slope + intercept)

    if first_ds is None:
        raise ValueError(f"No readable DICOM files found in {series_dir}")

    return {
        "series_dir": series_dir,
        "series_description": str(getattr(first_ds, "SeriesDescription", series_dir.name)),
        "study_date": getattr(first_ds, "StudyDate", None),
        "acquisition_time": getattr(first_ds, "AcquisitionTime", None),
        "acquisition_datetime": qspect_processing.parse_dicom_datetime(
            getattr(first_ds, "StudyDate", None),
            getattr(first_ds, "AcquisitionTime", None),
        ),
        "pixel_spacing": list(getattr(first_ds, "PixelSpacing", []) or []),
        "slice_thickness": getattr(first_ds, "SliceThickness", None),
        "shape": None,
        "volume": np.stack(slices, axis=0),
    }


def load_ct_for_qspect_day(qspect_dir: Path, qspect: Dict[str, Any]) -> Dict[str, Any]:
    ct_dirs = find_ct_series_dirs(qspect_dir)
    if not ct_dirs:
        raise ValueError(f"No CT series found in {qspect_dir}")

    qspect_datetime = qspect.get("acquisition_datetime")
    qspect_date = qspect_datetime.date() if qspect_datetime is not None else None
    candidates = []
    for series_dir in ct_dirs:
        first_file = next(series_dir.glob("*.dcm"), None)
        if first_file is None:
            continue
        ds = pydicom.dcmread(first_file, stop_before_pixels=True, force=True)
        description = str(getattr(ds, "SeriesDescription", series_dir.name))
        if "TOPOGRAM" in description.upper():
            continue
        series_datetime = qspect_processing.parse_dicom_datetime(
            getattr(ds, "StudyDate", None),
            getattr(ds, "AcquisitionTime", None),
        )
        same_day_penalty = 0 if qspect_date is not None and series_datetime is not None and series_datetime.date() == qspect_date else 1
        time_delta = abs((series_datetime - qspect_datetime).total_seconds()) if series_datetime is not None and qspect_datetime is not None else float("inf")
        candidates.append((same_day_penalty, ct_priority(description), time_delta, series_dir))

    if not candidates:
        raise ValueError(f"No usable CT/ACCT series found in {qspect_dir}")

    _same_day, _priority, _delta, series_dir = min(candidates, key=lambda item: item[:3])
    ct = load_ct_series(series_dir)
    ct["shape"] = ct["volume"].shape
    return ct


def display_limits(image: np.ndarray) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    positive = finite[finite > 0]
    if positive.size == 0:
        return 0.0, 1.0
    return 0.0, float(np.percentile(positive, 99.5))


def add_image_panel(ax: plt.Axes, image: np.ndarray, cmap: str = "magma") -> None:
    vmin, vmax = display_limits(image)
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.axis("off")


def add_ct_panel(ax: plt.Axes, image: np.ndarray) -> None:
    ax.imshow(image, cmap="gray", vmin=-200, vmax=300, aspect="auto")
    ax.axis("off")


def orient_qspect_display(image: np.ndarray) -> np.ndarray:
    if QSPECT_FLIP_VERTICAL:
        return np.flipud(image)
    return image


def save_figure(fig: plt.Figure, output_file: Path) -> Path:
    output_file = output_file.expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0)
    plt.show()
    return output_file


def make_planar_corrected_figure(
    planar_study_dir: Path = planar_processing.default_planar_study_dir(),
    selected_day: str = SELECTED_DAY,
    output_file: Path = OUTPUT_DIR / OUTPUT_FILE,
) -> Path:
    planar = load_planar_day(planar_study_dir, selected_day)

    corrected = planar["correction"]["corrected_image"]["image"]

    fig, ax = plt.subplots(figsize=(6.5, 6), facecolor="white")
    add_image_panel(ax, corrected, cmap="magma")
    fig.tight_layout(pad=0)
    return save_figure(fig, output_file)


def qspect_slice_indices(volume: np.ndarray, mode: str) -> tuple[int, int, int]:
    if mode == "center":
        return volume.shape[0] // 2, volume.shape[1] // 2, volume.shape[2] // 2
    if mode == "max":
        max_index = np.unravel_index(int(np.nanargmax(volume)), volume.shape)
        return int(max_index[0]), int(max_index[1]), int(max_index[2])
    raise ValueError(f"Unknown Q/SPECT slice mode: {mode}")


def make_qspect_slice_figure(
    qspect_dir: Path = qspect_processing.default_qspect_dir(),
    selected_day: str = SELECTED_DAY,
    mode: str = "max",
    output_file: Path = OUTPUT_DIR / QSPECT_MAX_FILE,
    qspect: Dict[str, Any] | None = None,
) -> Path:
    if qspect is None:
        qspect = load_qspect_day(qspect_dir, selected_day)
    volume = qspect["volume"]
    axial_index, coronal_index, sagittal_index = qspect_slice_indices(volume, mode)
    slices = [
        orient_qspect_display(volume[axial_index, :, :]),
        orient_qspect_display(volume[:, coronal_index, :]),
        orient_qspect_display(volume[:, :, sagittal_index]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="white")
    for ax, image in zip(axes, slices):
        add_image_panel(ax, image, cmap="magma")
    fig.tight_layout(pad=0)
    return save_figure(fig, output_file)


def make_ct_slice_figure(
    qspect_dir: Path = qspect_processing.default_qspect_dir(),
    selected_day: str = SELECTED_DAY,
    output_file: Path = OUTPUT_DIR / CT_CENTER_FILE,
    qspect: Dict[str, Any] | None = None,
    ct: Dict[str, Any] | None = None,
) -> Path:
    if qspect is None:
        qspect = load_qspect_day(qspect_dir, selected_day)
    if ct is None:
        ct = load_ct_for_qspect_day(qspect_dir, qspect)

    volume = ct["volume"]
    axial_index = volume.shape[0] // 2
    coronal_index = volume.shape[1] // 2
    sagittal_index = volume.shape[2] // 2
    slices = [
        np.flipud(volume[axial_index, :, :]),
        np.flipud(volume[:, coronal_index, :]),
        np.flipud(volume[:, :, sagittal_index]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="white")
    for ax, image in zip(axes, slices):
        add_ct_panel(ax, image)
    fig.tight_layout(pad=0)
    return save_figure(fig, output_file)


def main() -> None:
    selected_day = sys.argv[1] if len(sys.argv) > 1 else SELECTED_DAY
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT_DIR
    qspect = load_qspect_day(qspect_processing.default_qspect_dir(), selected_day)
    ct = load_ct_for_qspect_day(qspect_processing.default_qspect_dir(), qspect)
    saved_planar = make_planar_corrected_figure(
        selected_day=selected_day,
        output_file=output_dir / f"{selected_day.lower()}_{OUTPUT_FILE}",
    )
    saved_qspect_center = make_qspect_slice_figure(
        selected_day=selected_day,
        mode="center",
        output_file=output_dir / f"{selected_day.lower()}_{QSPECT_CENTER_FILE}",
        qspect=qspect,
    )
    saved_qspect_max = make_qspect_slice_figure(
        selected_day=selected_day,
        mode="max",
        output_file=output_dir / f"{selected_day.lower()}_{QSPECT_MAX_FILE}",
        qspect=qspect,
    )
    saved_ct_center = make_ct_slice_figure(
        selected_day=selected_day,
        output_file=output_dir / f"{selected_day.lower()}_{CT_CENTER_FILE}",
        qspect=qspect,
        ct=ct,
    )
    print(f"Saved figure: {saved_planar}")
    print(f"Saved figure: {saved_qspect_center}")
    print(f"Saved figure: {saved_qspect_max}")
    print(f"Saved figure: {saved_ct_center}")
    print(f"Selected CT: {ct['series_description']} | shape={ct['shape']} | spacing={ct['pixel_spacing']} | slice_thickness={ct['slice_thickness']}")


if __name__ == "__main__":
    main()
