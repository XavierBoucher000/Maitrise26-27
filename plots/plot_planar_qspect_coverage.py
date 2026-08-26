from pathlib import Path
import re
import sys
import time
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
OUTPUT_DIR = Path("fig") / "coverage"


def day_index(day_label: str) -> int:
    match = re.search(r"day\s*(\d+)", day_label, re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid day label: {day_label}")
    return int(match.group(1))


def display_limits(image: np.ndarray) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    positive = finite[finite > 0]
    if positive.size == 0:
        return 0.0, 1.0
    return 0.0, float(np.percentile(positive, 99.5))


def sorted_planar_scan_dirs(study_dir: Path) -> List[Path]:
    scan_dirs = dicom_loader.find_scan_dirs(study_dir)
    dated_scans = []
    for scan_dir in scan_dirs:
        images = dicom_loader.load_scan_directory(scan_dir)
        scan = {"scan_name": scan_dir.name, "scan_path": scan_dir, "images": images}
        dated_scans.append((dicom_loader.scan_datetime(scan), scan_dir))
    return [scan_dir for _dt, scan_dir in sorted(dated_scans, key=lambda item: (item[0] is None, item[0], item[1].name))]


def load_planar_tew_image(study_dir: Path, selected_day: str) -> Dict[str, Any]:
    scan_dirs = sorted_planar_scan_dirs(study_dir)
    index = day_index(selected_day)
    if index >= len(scan_dirs):
        raise ValueError(f"{selected_day} not available in planar study. Found {len(scan_dirs)} scans.")

    scan_dir = scan_dirs[index]
    images = dicom_loader.load_scan_directory(scan_dir)
    geometric_images = planar_processing.geometric_mean_images(images)
    correction = planar_processing.apply_tew_correction(geometric_images)
    corrected = correction["corrected_image"]["image"]

    first_image = images[0]
    pixel_spacing_mm = float(first_image["pixel_spacing"][0])
    scan_length_mm = first_image.get("scan_length")
    return {
        "image": corrected,
        "scan_dir": scan_dir,
        "pixel_spacing_mm": pixel_spacing_mm,
        "shape": corrected.shape,
        "matrix_length_cm": corrected.shape[0] * pixel_spacing_mm / 10.0,
        "matrix_width_cm": corrected.shape[1] * pixel_spacing_mm / 10.0,
        "scan_length_cm": None if scan_length_mm is None else float(scan_length_mm) / 10.0,
    }


def qspect_spacing_and_length(qspect: Dict[str, Any]) -> Dict[str, Any]:
    files = sorted(qspect["series_dir"].glob("*.dcm"), key=qspect_processing.sort_key_for_slice)
    if not files:
        raise ValueError(f"No DICOM files found in {qspect['series_dir']}")

    first_ds = pydicom.dcmread(files[0], stop_before_pixels=True, force=True)
    pixel_spacing = [float(value) for value in first_ds.PixelSpacing]
    slice_thickness = float(getattr(first_ds, "SliceThickness", pixel_spacing[0]))

    positions = []
    for path in files:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        image_position = getattr(ds, "ImagePositionPatient", None)
        if image_position is not None:
            positions.append([float(value) for value in image_position])

    z_range_mm = None
    if len(positions) > 1:
        z_values = [position[2] for position in positions]
        z_range_mm = max(z_values) - min(z_values)

    volume = qspect["volume"]
    return {
        "shape": volume.shape,
        "pixel_spacing_mm": pixel_spacing,
        "slice_thickness_mm": slice_thickness,
        "width_cm": volume.shape[2] * pixel_spacing[1] / 10.0,
        "height_cm": volume.shape[1] * pixel_spacing[0] / 10.0,
        "head_to_toe_center_range_cm": None if z_range_mm is None else z_range_mm / 10.0,
        "head_to_toe_full_cm": volume.shape[0] * slice_thickness / 10.0,
    }


def load_qspect_day(qspect_dir: Path, selected_day: str) -> Dict[str, Any]:
    series = qspect_processing.load_qspect_study(qspect_dir)
    return qspect_processing.selected_series(series, selected_day)


def make_coverage_figure(
    selected_day: str = SELECTED_DAY,
    planar_dir: Path = planar_processing.default_planar_study_dir(),
    qspect_dir: Path = qspect_processing.default_qspect_dir(),
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    planar = load_planar_tew_image(planar_dir, selected_day)
    qspect = load_qspect_day(qspect_dir, selected_day)
    qspect_meta = qspect_spacing_and_length(qspect)

    planar_image = planar["image"]
    qspect_volume = qspect["volume"]
    qspect_coronal_mip = np.flipud(qspect_volume.max(axis=1))

    planar_vmin, planar_vmax = display_limits(planar_image)
    qspect_vmin, qspect_vmax = display_limits(qspect_coronal_mip)

    planar_width = planar["matrix_width_cm"]
    planar_length = planar["matrix_length_cm"]
    qspect_width = qspect_meta["width_cm"]
    qspect_length = qspect_meta["head_to_toe_full_cm"]

    fig, axes = plt.subplots(1, 2, figsize=(8, 9), facecolor="white")

    axes[0].imshow(
        planar_image,
        cmap="magma",
        vmin=planar_vmin,
        vmax=planar_vmax,
        extent=[-planar_width / 2.0, planar_width / 2.0, 0.0, planar_length],
        aspect="equal",
        origin="upper",
    )
    axes[0].set_title("Planar WB TEW corrected")
    axes[0].set_xlabel("Transverse width (cm)")
    axes[0].set_ylabel("Head-to-toe image length (cm)")

    axes[1].imshow(
        qspect_coronal_mip,
        cmap="magma",
        vmin=qspect_vmin,
        vmax=qspect_vmax,
        extent=[-qspect_width / 2.0, qspect_width / 2.0, 0.0, qspect_length],
        aspect="equal",
        origin="upper",
    )
    axes[1].set_title("Q/SPECT coronal MIP")
    axes[1].set_xlabel("Transverse width (cm)")
    axes[1].set_ylabel("Head-to-toe volume length (cm)")

    for ax in axes:
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25)

    info = [
        f"Planar matrix: {planar_width:.1f} x {planar_length:.1f} cm",
        f"Planar DICOM scan length: {planar['scan_length_cm']:.1f} cm" if planar["scan_length_cm"] is not None else "Planar DICOM scan length: n/a",
        f"Q/SPECT volume: {qspect_width:.1f} x {qspect_meta['height_cm']:.1f} x {qspect_length:.1f} cm",
        f"Q/SPECT slice-position range: {qspect_meta['head_to_toe_center_range_cm']:.1f} cm" if qspect_meta["head_to_toe_center_range_cm"] is not None else "Q/SPECT slice-position range: n/a",
    ]
    fig.text(0.5, 0.02, " | ".join(info), ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=[0.0, 0.05, 1.0, 1.0])

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"planar_qspect_coverage_{selected_day.lower()}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show(block=False)
    fig.canvas.flush_events()
    time.sleep(0.1)
    plt.close("all")

    print(f"Saved figure: {output_path}")
    print("")
    print("Image dimensions and spacing:")
    print(
        f"  Planar shape: {planar['shape']} "
        f"(rows, cols), pixel size: {planar['pixel_spacing_mm']:.4f} x {planar['pixel_spacing_mm']:.4f} mm"
    )
    print(
        f"  Q/SPECT shape: {qspect_meta['shape']} "
        f"(slices, rows, cols), voxel size: "
        f"{qspect_meta['slice_thickness_mm']:.4f} x "
        f"{qspect_meta['pixel_spacing_mm'][0]:.4f} x "
        f"{qspect_meta['pixel_spacing_mm'][1]:.4f} mm"
    )
    for line in info:
        print(line)
    return output_path


def main() -> None:
    selected_day = sys.argv[1] if len(sys.argv) > 1 else SELECTED_DAY
    make_coverage_figure(selected_day=selected_day)


if __name__ == "__main__":
    main()
