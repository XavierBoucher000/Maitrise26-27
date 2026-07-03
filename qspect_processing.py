from datetime import datetime
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import pydicom


SELECTED_DAY = "Day0"


def find_dicom_series_dirs(root_dir: Path) -> List[Path]:
    """Return DICOM-containing directories below root_dir."""
    root_dir = root_dir.expanduser().resolve()
    if any(root_dir.glob("*.dcm")):
        return [root_dir]
    return sorted(path for path in root_dir.iterdir() if path.is_dir() and any(path.glob("*.dcm")))


def is_qspect_series_dir(series_dir: Path) -> bool:
    first_file = next(series_dir.glob("*.dcm"), None)
    if first_file is None:
        return False
    ds = pydicom.dcmread(first_file, stop_before_pixels=True, force=True)
    modality = str(getattr(ds, "Modality", ""))
    description = str(getattr(ds, "SeriesDescription", ""))
    return modality == "PT" and "QSPECT" in description.upper()


def find_qspect_series_dirs(root_dir: Path) -> List[Path]:
    return [series_dir for series_dir in find_dicom_series_dirs(root_dir) if is_qspect_series_dir(series_dir)]


def parse_dicom_datetime(study_date: Any, acquisition_time: Any) -> Optional[datetime]:
    if not study_date:
        return None
    date_text = str(study_date).strip()
    time_text = str(acquisition_time or "000000").strip().split(".")[0].ljust(6, "0")[:6]
    try:
        return datetime.strptime(date_text + time_text, "%Y%m%d%H%M%S")
    except ValueError:
        return None


def day_label(series_description: str, fallback_index: int) -> str:
    match = re.search(r"\bday\s*(\d+)\b", series_description, re.IGNORECASE)
    if match:
        return f"Day{match.group(1)}"
    return f"Series {fallback_index + 1}"


def sort_key_for_slice(path: Path) -> tuple:
    ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    instance_number = getattr(ds, "InstanceNumber", None)
    image_position = getattr(ds, "ImagePositionPatient", None)
    if image_position is not None and len(image_position) >= 3:
        try:
            return (0, float(image_position[2]))
        except (TypeError, ValueError):
            pass
    if instance_number is not None:
        try:
            return (1, int(instance_number))
        except (TypeError, ValueError):
            pass
    return (2, path.name)


def voxel_volume_ml(ds: pydicom.dataset.FileDataset) -> Optional[float]:
    pixel_spacing = getattr(ds, "PixelSpacing", None)
    slice_thickness = getattr(ds, "SliceThickness", None)
    if pixel_spacing is None or slice_thickness is None:
        return None
    try:
        return float(pixel_spacing[0]) * float(pixel_spacing[1]) * float(slice_thickness) / 1000.0
    except (TypeError, ValueError, IndexError):
        return None


def load_qspect_series(series_dir: Path, index: int = 0) -> Dict[str, Any]:
    """Load one axial PT/QSPECT series as a rescaled 3D volume."""
    files = sorted(series_dir.glob("*.dcm"), key=sort_key_for_slice)
    if not files:
        raise ValueError(f"No DICOM files found in {series_dir}")

    slices = []
    raw_total = 0.0
    first_ds = None
    for path in files:
        ds = pydicom.dcmread(path, force=True)
        if first_ds is None:
            first_ds = ds
        image = ds.pixel_array.astype(np.float64)
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        raw_total += float(image.sum())
        slices.append(image * slope + intercept)

    if first_ds is None:
        raise ValueError(f"No readable DICOM files found in {series_dir}")

    volume = np.stack(slices, axis=0)
    volume_ml = voxel_volume_ml(first_ds)
    scaled_sum = float(volume.sum())
    total_activity_bq = None if volume_ml is None else scaled_sum * volume_ml
    series_description = str(getattr(first_ds, "SeriesDescription", series_dir.name))
    acquisition_datetime = parse_dicom_datetime(
        getattr(first_ds, "StudyDate", None),
        getattr(first_ds, "AcquisitionTime", None),
    )

    return {
        "series_dir": series_dir,
        "label": day_label(series_description, index),
        "series_description": series_description,
        "acquisition_datetime": acquisition_datetime,
        "study_date": getattr(first_ds, "StudyDate", None),
        "acquisition_time": getattr(first_ds, "AcquisitionTime", None),
        "modality": getattr(first_ds, "Modality", None),
        "units": getattr(first_ds, "Units", None),
        "corrected_image": list(getattr(first_ds, "CorrectedImage", []) or []),
        "decay_correction": getattr(first_ds, "DecayCorrection", None),
        "decay_factor": getattr(first_ds, "DecayFactor", None),
        "voxel_volume_ml": volume_ml,
        "raw_total": raw_total,
        "scaled_sum": scaled_sum,
        "total_activity_bq": total_activity_bq,
        "nonzero_voxels": int(np.count_nonzero(volume)),
        "shape": volume.shape,
        "volume": volume,
    }


def load_qspect_study(root_dir: Path) -> List[Dict[str, Any]]:
    series_dirs = find_qspect_series_dirs(root_dir)
    if not series_dirs:
        raise ValueError(f"No PT/QSPECT series found in {root_dir}")
    series = [load_qspect_series(series_dir, index) for index, series_dir in enumerate(series_dirs)]
    return sorted(
        series,
        key=lambda item: (
            item["acquisition_datetime"] or datetime.max,
            item["series_description"],
            item["series_dir"].name,
        ),
    )


def day_offsets(series: List[Dict[str, Any]]) -> List[float]:
    datetimes = [item["acquisition_datetime"] for item in series if item["acquisition_datetime"] is not None]
    first = min(datetimes) if datetimes else None
    if first is None:
        return [float(index) for index, _item in enumerate(series)]
    return [
        0.0
        if item["acquisition_datetime"] is None
        else (item["acquisition_datetime"] - first).total_seconds() / 86400.0
        for item in series
    ]


def image_limits(volume: np.ndarray) -> tuple[float, float]:
    nonzero = volume[volume > 0]
    if nonzero.size == 0:
        return 0.0, 1.0
    return 0.0, float(np.percentile(nonzero, 99.5))


def plot_qspect_overview(series: List[Dict[str, Any]]) -> None:
    if not series:
        raise ValueError("No Q/SPECT series available to plot")

    fig, axes = plt.subplots(len(series), 3, figsize=(12, 3.3 * len(series)))
    if len(series) == 1:
        axes = np.array([axes])

    for row, item in enumerate(series):
        volume = item["volume"]
        vmin, vmax = image_limits(volume)
        mid_z = volume.shape[0] // 2
        axial = volume[mid_z]
        coronal_mip = volume.max(axis=1)
        sagittal_mip = volume.max(axis=2)
        images = [axial, coronal_mip, sagittal_mip]
        titles = ["Axial centre", "Coronal MIP", "Sagittal MIP"]

        for col, (image, title) in enumerate(zip(images, titles)):
            ax = axes[row, col]
            ax.imshow(image, cmap="hot", vmin=vmin, vmax=vmax, aspect="auto")
            ax.axis("off")
            label = item["label"]
            activity = item["total_activity_bq"]
            activity_text = "n/a" if activity is None else f"{activity / 1e6:.1f} MBq"
            ax.set_title(f"{label} - {title}\n{item['series_description']} | {activity_text}")

    fig.suptitle("Aperçu Q/SPECT par temps d'acquisition", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


def plot_qspect_decay(series: List[Dict[str, Any]]) -> None:
    if not series:
        raise ValueError("No Q/SPECT series available to plot")

    x = np.array(day_offsets(series))
    labels = [item["label"] for item in series]
    activities = np.array([
        np.nan if item["total_activity_bq"] is None else item["total_activity_bq"]
        for item in series
    ])

    fig, axes = plt.subplots(ncols=2, figsize=(12, 4.5))
    axes[0].plot(x, activities / 1e6, marker="o", color="tab:blue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_xlabel("Temps")
    axes[0].set_ylabel("Activité totale approx. (MBq)")
    axes[0].set_title("Activité totale au fil du temps")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    if np.isfinite(activities[0]) and activities[0] > 0:
        normalized = activities / activities[0]
        axes[1].plot(x, normalized, marker="o", color="tab:orange")
        axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel("Temps")
    axes[1].set_ylabel("Fraction du premier scan")
    axes[1].set_title("Décroissance relative")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    plt.show()


def selected_series(series: List[Dict[str, Any]], selected_day: str) -> Dict[str, Any]:
    for item in series:
        if item["label"].lower() == selected_day.lower():
            return item
    available = ", ".join(item["label"] for item in series)
    raise ValueError(f"Selected day {selected_day!r} not found. Available days: {available}")


def plot_qspect_slice_viewer(series: List[Dict[str, Any]], selected_day: str = SELECTED_DAY) -> None:
    if not series:
        raise ValueError("No Q/SPECT series available to plot")

    item = selected_series(series, selected_day)
    volume = item["volume"]
    axial_index = volume.shape[0] // 2
    coronal_index = volume.shape[1] // 2
    sagittal_index = volume.shape[2] // 2
    vmin, vmax = image_limits(volume)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
    plt.subplots_adjust(bottom=0.25)

    axial_artist = axes[0].imshow(volume[axial_index], cmap="hot", vmin=vmin, vmax=vmax, aspect="auto")
    coronal_artist = axes[1].imshow(volume[:, coronal_index, :], cmap="hot", vmin=vmin, vmax=vmax, aspect="auto")
    sagittal_artist = axes[2].imshow(volume[:, :, sagittal_index], cmap="hot", vmin=vmin, vmax=vmax, aspect="auto")

    axes[0].set_title(f"Axiale {axial_index + 1}/{volume.shape[0]}")
    axes[1].set_title(f"Coronale {coronal_index + 1}/{volume.shape[1]}")
    axes[2].set_title(f"Sagittale {sagittal_index + 1}/{volume.shape[2]}")
    for ax in axes:
        ax.axis("off")

    activity = item["total_activity_bq"]
    activity_text = "n/a" if activity is None else f"{activity / 1e6:.1f} MBq"
    fig.suptitle(f"{item['label']} - {item['series_description']} | {activity_text}")

    axial_slider_axis = fig.add_axes([0.18, 0.14, 0.64, 0.03])
    coronal_slider_axis = fig.add_axes([0.18, 0.09, 0.64, 0.03])
    sagittal_slider_axis = fig.add_axes([0.18, 0.04, 0.64, 0.03])

    axial_slider = Slider(axial_slider_axis, "Axiale", 1, volume.shape[0], valinit=axial_index + 1, valstep=1)
    coronal_slider = Slider(coronal_slider_axis, "Coronale", 1, volume.shape[1], valinit=coronal_index + 1, valstep=1)
    sagittal_slider = Slider(sagittal_slider_axis, "Sagittale", 1, volume.shape[2], valinit=sagittal_index + 1, valstep=1)

    def update(_value: float) -> None:
        ax_i = int(axial_slider.val) - 1
        cor_i = int(coronal_slider.val) - 1
        sag_i = int(sagittal_slider.val) - 1
        axial_artist.set_data(volume[ax_i])
        coronal_artist.set_data(volume[:, cor_i, :])
        sagittal_artist.set_data(volume[:, :, sag_i])
        axes[0].set_title(f"Axiale {ax_i + 1}/{volume.shape[0]}")
        axes[1].set_title(f"Coronale {cor_i + 1}/{volume.shape[1]}")
        axes[2].set_title(f"Sagittale {sag_i + 1}/{volume.shape[2]}")
        fig.canvas.draw_idle()

    axial_slider.on_changed(update)
    coronal_slider.on_changed(update)
    sagittal_slider.on_changed(update)
    plt.show()


def print_qspect_summary(series: List[Dict[str, Any]]) -> None:
    print("Q/SPECT series summary:")
    for item in series:
        activity = item["total_activity_bq"]
        activity_text = "n/a" if activity is None else f"{activity / 1e6:.1f} MBq"
        print(
            f"  {item['label']}: {item['shape']} | {item['series_description']} | "
            f"{item['study_date']} {item['acquisition_time']} | {activity_text} | "
            f"units={item['units']} corrected={','.join(item['corrected_image'])}"
        )


def qspect_decay_data(root_dir: Path) -> List[Dict[str, Any]]:
    """Return per-acquisition Q/SPECT activity data without plotting."""
    series = load_qspect_study(root_dir)
    offsets = day_offsets(series)
    rows = []
    for item, day_offset in zip(series, offsets):
        rows.append(
            {
                "label": item["label"],
                "datetime": item["acquisition_datetime"],
                "day_offset": day_offset,
                "activity_bq": item["total_activity_bq"],
                "activity_mbq": None if item["total_activity_bq"] is None else item["total_activity_bq"] / 1e6,
                "series_description": item["series_description"],
            }
        )
    return rows


def run_qspect_workflow(root_dir: Path) -> None:
    series = load_qspect_study(root_dir)
    print_qspect_summary(series)
    plot_qspect_slice_viewer(series, SELECTED_DAY)
    plot_qspect_decay(series)


def default_qspect_dir() -> Path:
    base = Path(__file__).resolve().parent
    data_path = base / "data" / "2026-05_studies" / "2026-05__Studies"
    if data_path.exists():
        return data_path
    return base / "2026-05__Studiesv2"


def main() -> None:
    root_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else default_qspect_dir()
    root_dir = root_dir.expanduser().resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"Q/SPECT folder not found: {root_dir}")
    run_qspect_workflow(root_dir)


if __name__ == "__main__":
    main()
