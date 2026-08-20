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
PLANAR_DAY_LABELS = ("Day0", "Day1", "Day2", "Day3", "Day6")
OUTPUT_DIR = Path("fig") / "imaging"
TEST_MATCH_DIR = OUTPUT_DIR / "test_match"
SEPARATE_MODALITIES_DIR = TEST_MATCH_DIR / "separate_modalities"
OUTPUT_FILE = "planar_tew_corrected.png"
PLANAR_QSPECT_LENGTH_FILE = "planar_tew_corrected_qspect_length.png"
PLANAR_QSPECT_LENGTH_SIDE_BY_SIDE_FILE = "planar_qspect_length_side_by_side.png"
PLANAR_QSPECT_CROP_SIDE_BY_SIDE_FILE = "planar_qspect_crop_side_by_side.png"
QSPECT_CENTER_FILE = "qspect_center_slices.png"
QSPECT_MAX_FILE = "qspect_max_slices.png"
CT_CENTER_FILE = "ct_center_slices.png"
QSPECT_FLIP_VERTICAL = True


def day_index(day_label: str) -> int:
    match = re.search(r"day\s*(\d+)", day_label, re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid day label: {day_label}")
    normalized_label = f"Day{int(match.group(1))}"
    if normalized_label not in PLANAR_DAY_LABELS:
        available = ", ".join(PLANAR_DAY_LABELS)
        raise ValueError(f"Planar day {day_label!r} not available. Available days: {available}")
    return PLANAR_DAY_LABELS.index(normalized_label)


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
        "images": images,
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


def planar_pixel_spacing_mm(planar: Dict[str, Any]) -> float:
    for item in planar["images"]:
        pixel_spacing = item.get("pixel_spacing")
        if pixel_spacing is not None and len(pixel_spacing) >= 1:
            return float(pixel_spacing[0])
    raise ValueError("Planar pixel spacing is required to draw Q/SPECT length")


def qspect_head_to_toe_length_mm(qspect: Dict[str, Any]) -> float:
    first_file = next(qspect["series_dir"].glob("*.dcm"), None)
    if first_file is None:
        raise ValueError(f"No Q/SPECT DICOM files found in {qspect['series_dir']}")
    ds = pydicom.dcmread(first_file, stop_before_pixels=True, force=True)
    slice_thickness = float(getattr(ds, "SliceThickness"))
    return float(qspect["volume"].shape[0]) * slice_thickness


def qspect_length_line_on_planar(
    planar_image: np.ndarray,
    planar: Dict[str, Any],
    qspect: Dict[str, Any],
) -> Dict[str, float]:
    pixel_spacing_mm = planar_pixel_spacing_mm(planar)
    qspect_length_mm = qspect_head_to_toe_length_mm(qspect)
    qspect_length_pixels = qspect_length_mm / pixel_spacing_mm
    y_center = (planar_image.shape[0] - 1) / 2.0
    x_center = (planar_image.shape[1] - 1) / 2.0
    y_top = y_center - qspect_length_pixels / 2.0
    y_bottom = y_center + qspect_length_pixels / 2.0
    return {
        "pixel_spacing_mm": pixel_spacing_mm,
        "qspect_length_mm": qspect_length_mm,
        "qspect_length_pixels": qspect_length_pixels,
        "x_center": x_center,
        "y_center": y_center,
        "y_top": y_top,
        "y_bottom": y_bottom,
    }


def measure_signal_length_y(
    image: np.ndarray,
    spacing_y_mm: float,
    threshold_fraction: float = 0.5,
) -> Dict[str, float]:
    finite_image = np.nan_to_num(np.asarray(image, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    finite_image = np.clip(finite_image, 0.0, None)
    profile = finite_image.mean(axis=1)
    if profile.size == 0 or float(profile.max()) <= 0.0:
        raise ValueError("Cannot measure Y length from an empty image profile")

    normalized_profile = profile / float(profile.max())
    mask = normalized_profile >= threshold_fraction
    indices = np.where(mask)[0]
    if indices.size == 0:
        raise ValueError(f"No pixels above threshold fraction {threshold_fraction:g}")

    y_top = int(indices[0])
    y_bottom = int(indices[-1])
    length_pixels = y_bottom - y_top + 1
    masked_weights = profile[indices]
    center_of_mass_y = float(np.average(indices, weights=masked_weights))
    return {
        "threshold_fraction": threshold_fraction,
        "y_top": float(y_top),
        "y_bottom": float(y_bottom),
        "center_of_mass_y": center_of_mass_y,
        "length_pixels": float(length_pixels),
        "length_cm": float(length_pixels) * spacing_y_mm / 10.0,
    }


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


def make_planar_qspect_length_figure(
    planar_study_dir: Path = planar_processing.default_planar_study_dir(),
    qspect_dir: Path = qspect_processing.default_qspect_dir(),
    selected_day: str = SELECTED_DAY,
    output_file: Path = OUTPUT_DIR / PLANAR_QSPECT_LENGTH_FILE,
    qspect: Dict[str, Any] | None = None,
) -> Path:
    planar = load_planar_day(planar_study_dir, selected_day)
    if qspect is None:
        qspect = load_qspect_day(qspect_dir, selected_day)

    corrected = planar["correction"]["corrected_image"]["image"]
    line = qspect_length_line_on_planar(corrected, planar, qspect)

    fig, ax = plt.subplots(figsize=(6.5, 6), facecolor="white")
    vmin, vmax = display_limits(corrected)
    ax.imshow(corrected, cmap="magma", vmin=vmin, vmax=vmax, aspect="auto")
    ax.plot([line["x_center"], line["x_center"]], [line["y_top"], line["y_bottom"]], color="cyan", linewidth=2.5)
    ax.plot(
        [line["x_center"] - 10, line["x_center"] + 10],
        [line["y_top"], line["y_top"]],
        color="cyan",
        linewidth=2.5,
    )
    ax.plot(
        [line["x_center"] - 10, line["x_center"] + 10],
        [line["y_bottom"], line["y_bottom"]],
        color="cyan",
        linewidth=2.5,
    )
    ax.axis("off")
    fig.tight_layout(pad=0)

    print(
        "Q/SPECT centered planar line: "
        f"length={line['qspect_length_mm'] / 10.0:.1f} cm, "
        f"planar_pixels={line['qspect_length_pixels']:.1f}, "
        f"y_top={line['y_top']:.1f}, y_bottom={line['y_bottom']:.1f}, x_center={line['x_center']:.1f}"
    )
    return save_figure(fig, output_file)


def make_planar_qspect_length_side_by_side_figure(
    planar_study_dir: Path = planar_processing.default_planar_study_dir(),
    qspect_dir: Path = qspect_processing.default_qspect_dir(),
    selected_day: str = SELECTED_DAY,
    output_file: Path = TEST_MATCH_DIR / PLANAR_QSPECT_LENGTH_SIDE_BY_SIDE_FILE,
    qspect: Dict[str, Any] | None = None,
) -> Path:
    planar = load_planar_day(planar_study_dir, selected_day)
    if qspect is None:
        qspect = load_qspect_day(qspect_dir, selected_day)

    planar_image = planar["correction"]["corrected_image"]["image"]
    line = qspect_length_line_on_planar(planar_image, planar, qspect)
    planar_measurement = measure_signal_length_y(planar_image, line["pixel_spacing_mm"], threshold_fraction=0.1)
    qspect_volume = qspect["volume"]
    coronal_index = qspect_volume.shape[1] // 2
    qspect_coronal = orient_qspect_display(qspect_volume[:, coronal_index, :])
    qspect_spacing_y_mm = line["qspect_length_mm"] / qspect_coronal.shape[0]
    qspect_measurement = measure_signal_length_y(qspect_coronal, qspect_spacing_y_mm, threshold_fraction=0.1)
    matched_line_length_pixels = qspect_measurement["length_cm"] * 10.0 / line["pixel_spacing_mm"]
    matched_line_center_y = planar_measurement["center_of_mass_y"]
    matched_line_top = matched_line_center_y - matched_line_length_pixels / 2.0
    matched_line_bottom = matched_line_center_y + matched_line_length_pixels / 2.0
    qspect_gap_pixels = float(qspect_coronal.shape[0]) - qspect_measurement["length_pixels"]
    qspect_gap_cm = qspect_gap_pixels * qspect_spacing_y_mm / 10.0
    qspect_gap_planar_pixels = qspect_gap_cm * 10.0 / line["pixel_spacing_mm"]
    adjusted_line_top = max(0.0, matched_line_top - qspect_gap_planar_pixels)
    adjusted_line_bottom = min(float(planar_image.shape[0] - 1), matched_line_bottom)
    adjusted_line_length_pixels = adjusted_line_bottom - adjusted_line_top + 1.0

    fig, axes = plt.subplots(1, 2, figsize=(9, 5), facecolor="white")

    vmin, vmax = display_limits(planar_image)
    axes[0].imshow(planar_image, cmap="magma", vmin=vmin, vmax=vmax, aspect="auto")
    axes[0].plot([line["x_center"], line["x_center"]], [adjusted_line_top, adjusted_line_bottom], color="cyan", linewidth=2.5)
    axes[0].plot(
        [line["x_center"] - 10, line["x_center"] + 10],
        [adjusted_line_top, adjusted_line_top],
        color="cyan",
        linewidth=2.5,
    )
    axes[0].plot(
        [line["x_center"] - 10, line["x_center"] + 10],
        [adjusted_line_bottom, adjusted_line_bottom],
        color="cyan",
        linewidth=2.5,
    )
    axes[0].axhline(planar_measurement["y_top"], color="yellow", linewidth=1.6)
    axes[0].axhline(planar_measurement["y_bottom"], color="yellow", linewidth=1.6)
    axes[0].set_title("Planar WB")
    axes[0].axis("off")

    qmin, qmax = display_limits(qspect_coronal)
    axes[1].imshow(qspect_coronal, cmap="magma", vmin=qmin, vmax=qmax, aspect="auto")
    axes[1].axhline(qspect_measurement["y_top"], color="yellow", linewidth=1.6)
    axes[1].axhline(qspect_measurement["y_bottom"], color="yellow", linewidth=1.6)
    axes[1].set_title("Q/SPECT coronal center")
    axes[1].axis("off")

    fig.tight_layout(pad=0.4)
    print(
        "Threshold Y length at 0.1 mean-profile max: "
        f"planar={planar_measurement['length_cm']:.1f} cm "
        f"({planar_measurement['length_pixels']:.0f} px, y={planar_measurement['y_top']:.0f}-{planar_measurement['y_bottom']:.0f}, "
        f"com_y={planar_measurement['center_of_mass_y']:.1f}), "
        f"qspect={qspect_measurement['length_cm']:.1f} cm "
        f"({qspect_measurement['length_pixels']:.0f} px, y={qspect_measurement['y_top']:.0f}-{qspect_measurement['y_bottom']:.0f}, "
        f"com_y={qspect_measurement['center_of_mass_y']:.1f})"
    )
    print(
        "Matched cyan planar line with Q/SPECT head-side gap added: "
        f"qspect_gap={qspect_gap_cm:.1f} cm, "
        f"length={adjusted_line_length_pixels * line['pixel_spacing_mm'] / 10.0:.1f} cm, "
        f"planar_pixels={adjusted_line_length_pixels:.1f}, center_y={matched_line_center_y:.1f}, "
        f"y_top={adjusted_line_top:.1f}, y_bottom={adjusted_line_bottom:.1f}"
    )
    return save_figure(fig, output_file)


def make_planar_qspect_crop_side_by_side_figure(
    planar_study_dir: Path = planar_processing.default_planar_study_dir(),
    qspect_dir: Path = qspect_processing.default_qspect_dir(),
    selected_day: str = SELECTED_DAY,
    output_file: Path = TEST_MATCH_DIR / PLANAR_QSPECT_CROP_SIDE_BY_SIDE_FILE,
    qspect: Dict[str, Any] | None = None,
) -> Path:
    planar = load_planar_day(planar_study_dir, selected_day)
    if qspect is None:
        qspect = load_qspect_day(qspect_dir, selected_day)

    planar_image = planar["correction"]["corrected_image"]["image"]
    line = qspect_length_line_on_planar(planar_image, planar, qspect)
    planar_measurement = measure_signal_length_y(planar_image, line["pixel_spacing_mm"], threshold_fraction=0.1)
    qspect_volume = qspect["volume"]
    coronal_index = qspect_volume.shape[1] // 2
    qspect_coronal = orient_qspect_display(qspect_volume[:, coronal_index, :])
    qspect_spacing_y_mm = line["qspect_length_mm"] / qspect_coronal.shape[0]
    qspect_measurement = measure_signal_length_y(qspect_coronal, qspect_spacing_y_mm, threshold_fraction=0.1)

    matched_line_length_pixels = qspect_measurement["length_cm"] * 10.0 / line["pixel_spacing_mm"]
    matched_line_center_y = planar_measurement["center_of_mass_y"]
    matched_line_top = matched_line_center_y - matched_line_length_pixels / 2.0
    matched_line_bottom = matched_line_center_y + matched_line_length_pixels / 2.0
    qspect_gap_pixels = float(qspect_coronal.shape[0]) - qspect_measurement["length_pixels"]
    qspect_gap_cm = qspect_gap_pixels * qspect_spacing_y_mm / 10.0
    qspect_gap_planar_pixels = qspect_gap_cm * 10.0 / line["pixel_spacing_mm"]
    adjusted_line_top = max(0.0, matched_line_top - qspect_gap_planar_pixels)
    adjusted_line_bottom = min(float(planar_image.shape[0] - 1), matched_line_bottom)

    crop_top = int(np.floor(adjusted_line_top))
    crop_bottom = int(np.ceil(adjusted_line_bottom)) + 1
    planar_crop = planar_image[crop_top:crop_bottom, :]

    fig, axes = plt.subplots(1, 2, figsize=(8, 5), facecolor="white")
    vmin, vmax = display_limits(planar_crop)
    axes[0].imshow(planar_crop, cmap="magma", vmin=vmin, vmax=vmax, aspect="auto")
    axes[0].set_title("Planar crop")
    axes[0].axis("off")

    qmin, qmax = display_limits(qspect_coronal)
    axes[1].imshow(qspect_coronal, cmap="magma", vmin=qmin, vmax=qmax, aspect="auto")
    axes[1].set_title("Q/SPECT coronal center")
    axes[1].axis("off")
    fig.tight_layout(pad=0.4)

    print(
        "Planar crop from adjusted cyan line: "
        f"y={crop_top}-{crop_bottom - 1}, "
        f"height={planar_crop.shape[0]} px, "
        f"height={planar_crop.shape[0] * line['pixel_spacing_mm'] / 10.0:.1f} cm, "
        f"counts={float(np.sum(planar_crop)):.1f}"
    )
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


def save_single_modality_figure(
    image: np.ndarray,
    title: str,
    output_file: Path,
    is_ct: bool = False,
) -> Path:
    """Save one image panel filling a common test-match figure frame."""
    fig, ax = plt.subplots(figsize=(4, 7), facecolor="white")
    if is_ct:
        ax.imshow(image, cmap="gray", vmin=-200, vmax=300, aspect="auto")
    else:
        vmin, vmax = display_limits(image)
        ax.imshow(image, cmap="hot", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    return save_figure(fig, output_file)


def make_separate_modality_figures(
    planar_study_dir: Path = planar_processing.default_planar_study_dir(),
    qspect_dir: Path = qspect_processing.default_qspect_dir(),
    selected_day: str = SELECTED_DAY,
    output_dir: Path = SEPARATE_MODALITIES_DIR,
    qspect: Dict[str, Any] | None = None,
    ct: Dict[str, Any] | None = None,
) -> List[Path]:
    """Save separate planar WB, Q/SPECT coronal views, and matched CT figures."""
    planar = load_planar_day(planar_study_dir, selected_day)
    if qspect is None:
        qspect = load_qspect_day(qspect_dir, selected_day)
    if ct is None:
        ct = load_ct_for_qspect_day(qspect_dir, qspect)

    planar_image = planar["correction"]["corrected_image"]["image"]
    qspect_volume = np.asarray(qspect["volume"], dtype=np.float64)
    coronal_index = qspect_volume.shape[1] // 2
    qspect_coronal = orient_qspect_display(qspect_volume[:, coronal_index, :])
    qspect_coronal_mean = orient_qspect_display(np.mean(qspect_volume, axis=1))
    ct_volume = np.asarray(ct["volume"], dtype=np.float64)
    ct_description = str(ct.get("series_description", "CT"))
    ct_is_matched = "TRANSFORMED" in ct_description.upper() and ct_volume.shape == qspect_volume.shape
    ct_coronal_index = coronal_index if ct_is_matched else ct_volume.shape[1] // 2
    ct_coronal = orient_qspect_display(ct_volume[:, ct_coronal_index, :])
    ct_title = (
        "CT transformé\n(coupe coronale centrale)"
        if ct_is_matched
        else "CT non recalé\n(coupe coronale centrale)"
    )

    day = selected_day.lower()
    output_dir = output_dir.expanduser().resolve()
    return [
        save_single_modality_figure(
            planar_image,
            "Planaire WB\n(projection GM AP/PA, TEW)",
            output_dir / f"{day}_planar_wb_tew_gm.png",
        ),
        save_single_modality_figure(
            qspect_coronal,
            "Q/SPECT\n(coupe coronale centrale)",
            output_dir / f"{day}_qspect_coronal_center.png",
        ),
        save_single_modality_figure(
            qspect_coronal_mean,
            "Q/SPECT\n(projection coronale moyenne)",
            output_dir / f"{day}_qspect_coronal_mean_projection.png",
        ),
        save_single_modality_figure(
            ct_coronal,
            ct_title,
            output_dir / f"{day}_ct_coronal_center.png",
            is_ct=True,
        ),
    ]


def main() -> None:
    selected_day = sys.argv[1] if len(sys.argv) > 1 else SELECTED_DAY
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT_DIR
    qspect = load_qspect_day(qspect_processing.default_qspect_dir(), selected_day)
    ct = load_ct_for_qspect_day(qspect_processing.default_qspect_dir(), qspect)
    saved_planar = make_planar_corrected_figure(
        selected_day=selected_day,
        output_file=output_dir / f"{selected_day.lower()}_{OUTPUT_FILE}",
    )
    saved_planar_qspect_length = make_planar_qspect_length_figure(
        selected_day=selected_day,
        output_file=TEST_MATCH_DIR / f"{selected_day.lower()}_{PLANAR_QSPECT_LENGTH_FILE}",
        qspect=qspect,
    )
    saved_planar_qspect_side_by_side = make_planar_qspect_length_side_by_side_figure(
        selected_day=selected_day,
        output_file=TEST_MATCH_DIR / f"{selected_day.lower()}_{PLANAR_QSPECT_LENGTH_SIDE_BY_SIDE_FILE}",
        qspect=qspect,
    )
    saved_planar_qspect_crop_side_by_side = make_planar_qspect_crop_side_by_side_figure(
        selected_day=selected_day,
        output_file=TEST_MATCH_DIR / f"{selected_day.lower()}_{PLANAR_QSPECT_CROP_SIDE_BY_SIDE_FILE}",
        qspect=qspect,
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
    saved_separate_modalities = make_separate_modality_figures(
        selected_day=selected_day,
        qspect=qspect,
        ct=ct,
    )
    print(f"Saved figure: {saved_planar}")
    print(f"Saved figure: {saved_planar_qspect_length}")
    print(f"Saved figure: {saved_planar_qspect_side_by_side}")
    print(f"Saved figure: {saved_planar_qspect_crop_side_by_side}")
    print(f"Saved figure: {saved_qspect_center}")
    print(f"Saved figure: {saved_qspect_max}")
    print(f"Saved figure: {saved_ct_center}")
    for saved_figure in saved_separate_modalities:
        print(f"Saved figure: {saved_figure}")
    print(f"Selected CT: {ct['series_description']} | shape={ct['shape']} | spacing={ct['pixel_spacing']} | slice_thickness={ct['slice_thickness']}")


if __name__ == "__main__":
    main()
