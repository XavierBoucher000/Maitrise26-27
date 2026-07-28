import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pydicom


def is_dicom_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".dcm"


def parse_scan_label(series_description: str) -> str:
    """Parse a scan label from the DICOM SeriesDescription or filename."""
    if not series_description:
        return "unknown"
    match = re.search(r"(S\d+D\d+|D\d+|S\d+)", series_description, re.I)
    if match:
        return match.group(1)
    return series_description.strip()


def parse_view(series_description: str) -> str:
    """Extract the view position from the SeriesDescription."""
    if not series_description:
        return "unknown"
    if re.search(r"ant|anterior", series_description, re.I):
        return "ant"
    if re.search(r"post|posterior", series_description, re.I):
        return "post"
    if re.search(r"right", series_description, re.I):
        return "right"
    if re.search(r"left", series_description, re.I):
        return "left"
    return "other"


def canonical_energy_window_label(
    label: str,
    lower_limit: Optional[float] = None,
    upper_limit: Optional[float] = None,
) -> str:
    if lower_limit is not None and upper_limit is not None:
        center = (lower_limit + upper_limit) / 2.0
        if 185 <= center <= 230:
            return "Photopeak"
        if 150 <= center < 185:
            return "Lower Scatter"
        if 230 < center <= 260:
            return "Upper Scatter"
        if center < 150:
            return "Low Energy Scatter"

    if not label:
        return "Unknown Window"
    label = label.strip().lower()
    if "lower" in label:
        return "Lower Scatter"
    if "upper" in label:
        return "Upper Scatter"
    if "lutetium" in label or "177" in label or "photo" in label or "main" in label:
        return "Photopeak"
    return label.title()


def get_energy_window_limits(item: pydicom.dataset.Dataset) -> tuple[Optional[float], Optional[float]]:
    lower = getattr(item, "EnergyWindowLowerLimit", None)
    upper = getattr(item, "EnergyWindowUpperLimit", None)

    range_sequence = getattr(item, "EnergyWindowRangeSequence", None)
    if (lower is None or upper is None) and range_sequence:
        first_range = range_sequence[0]
        lower = getattr(first_range, "EnergyWindowLowerLimit", lower)
        upper = getattr(first_range, "EnergyWindowUpperLimit", upper)

    try:
        lower = None if lower is None else float(lower)
        upper = None if upper is None else float(upper)
    except (TypeError, ValueError):
        lower = None
        upper = None
    return lower, upper


def get_energy_window_info(ds: pydicom.dataset.FileDataset, window_index: int = 0) -> Dict[str, Any]:
    energy_window = {
        "energy_window": "Unknown Window",
        "energy_window_name": None,
        "energy_window_lower_limit": None,
        "energy_window_upper_limit": None,
    }
    sequence = getattr(ds, "EnergyWindowInformationSequence", None)
    if not sequence:
        return energy_window

    window_index = max(0, min(window_index, len(sequence) - 1))
    item = sequence[window_index]
    name = getattr(item, "EnergyWindowName", None)
    lower, upper = get_energy_window_limits(item)

    energy_window["energy_window_name"] = str(name) if name is not None else None
    energy_window["energy_window_lower_limit"] = lower
    energy_window["energy_window_upper_limit"] = upper
    energy_window["energy_window"] = canonical_energy_window_label(str(name or ""), lower, upper)
    return energy_window


def parse_energy_window(series_description: str, ds: pydicom.dataset.FileDataset) -> str:
    """Extract a readable energy window label from DICOM metadata."""
    if hasattr(ds, "EnergyWindowInformationSequence") and ds.EnergyWindowInformationSequence:
        return get_energy_window_info(ds)["energy_window"]
    if re.search(r"lower", series_description, re.I):
        return "Lower Scatter"
    if re.search(r"upper", series_description, re.I):
        return "Upper Scatter"
    if re.search(r"lutetium|177|photo|main", series_description, re.I):
        return "Photopeak"
    return "Unknown Window"


def extract_energy_window(ds: pydicom.dataset.FileDataset) -> Dict[str, Any]:
    return get_energy_window_info(ds)


def frame_view_label(ds: pydicom.dataset.FileDataset, frame_index: int, series_description: str) -> str:
    detector_vector = getattr(ds, "DetectorVector", None)
    if detector_vector and frame_index < len(detector_vector):
        detector = int(detector_vector[frame_index])
        if detector == 1:
            return "AP"
        if detector == 2:
            return "PA"
    return parse_view(series_description)


def frame_energy_window_info(ds: pydicom.dataset.FileDataset, frame_index: int) -> Dict[str, Any]:
    energy_vector = getattr(ds, "EnergyWindowVector", None)
    window_index = 0
    if energy_vector and frame_index < len(energy_vector):
        window_index = int(energy_vector[frame_index]) - 1
    return get_energy_window_info(ds, window_index)


def expand_multiframe_item(item: Dict[str, Any], ds: pydicom.dataset.FileDataset) -> List[Dict[str, Any]]:
    image = item.get("image")
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        return [item]

    expanded = []
    series_description = item.get("series_description", "")
    for frame_index in range(image.shape[0]):
        frame = image[frame_index]
        frame_item = item.copy()
        frame_item["image"] = frame
        frame_item["frame_index"] = frame_index + 1
        frame_item["frame_count"] = image.shape[0]
        frame_item["view"] = frame_view_label(ds, frame_index, series_description)
        frame_item.update(frame_energy_window_info(ds, frame_index))
        frame_item["pixel_sum"] = float(frame.sum())
        frame_item["pixel_mean"] = float(frame.mean())
        frame_item["pixel_dtype"] = str(frame.dtype)
        expanded.append(frame_item)
    return expanded


def print_dicom_metadata(path: Path) -> None:
    ds = pydicom.dcmread(path)
    print(f"\n=== DICOM metadata for {path.name} ===")
    for elem in ds:
        if elem.tag == (0x7fe0, 0x0010) or elem.keyword == 'PixelData':
            print(f"{elem.tag} {elem.name}: <PixelData omitted>")
            continue
        value = repr(elem.value)
        if len(value) > 250:
            value = value[:250] + '...'
        print(f"{elem.tag} {elem.name}: {value}")
    print("=== End DICOM metadata ===\n")


def load_dicom_file(path: Path, read_pixels: bool = True) -> Dict[str, Any]:
    ds = pydicom.dcmread(path)
    image_data: Optional[np.ndarray] = None
    if read_pixels:
        image_data = ds.pixel_array

    series_description = getattr(ds, "SeriesDescription", "")
    labels = parse_scan_label(series_description)
    view = parse_view(series_description)
    energy_window_label = parse_energy_window(series_description, ds)
    energy_data = extract_energy_window(ds)

    return {
        "path": path,
        "series_description": series_description,
        "scan_label": labels,
        "view": view,
        "energy_window": energy_window_label,
        "energy_window_name": energy_data["energy_window_name"],
        "energy_window_lower_limit": energy_data["energy_window_lower_limit"],
        "energy_window_upper_limit": energy_data["energy_window_upper_limit"],
        "acquisition_time": getattr(ds, "AcquisitionTime", None),
        "actual_frame_duration_ms": getattr(ds, "ActualFrameDuration", None),
        "scan_velocity": getattr(ds, "ScanVelocity", None),
        "scan_length": getattr(ds, "ScanLength", None),
        "study_date": getattr(ds, "StudyDate", None),
        "modality": getattr(ds, "Modality", None),
        "rows": getattr(ds, "Rows", None),
        "columns": getattr(ds, "Columns", None),
        "pixel_spacing": getattr(ds, "PixelSpacing", None),
        "window_center": getattr(ds, "WindowCenter", None),
        "window_width": getattr(ds, "WindowWidth", None),
        "photometric_interpretation": getattr(ds, "PhotometricInterpretation", None),
        "rescale_intercept": getattr(ds, "RescaleIntercept", None),
        "rescale_slope": getattr(ds, "RescaleSlope", None),
        "image": image_data,
        "pixel_sum": None if image_data is None else float(image_data.sum()),
        "pixel_mean": None if image_data is None else float(image_data.mean()),
        "pixel_dtype": None if image_data is None else str(image_data.dtype),
    }


def load_scan_directory(scan_dir: Path) -> List[Dict[str, Any]]:
    files = sorted(scan_dir.glob("*.dcm"))
    scan_data = []
    for path in files:
        if not is_dicom_file(path):
            continue
        item = load_dicom_file(path)
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        scan_data.extend(expand_multiframe_item(item, ds))
    return scan_data


def find_scan_dirs(root_dir: Path) -> List[Path]:
    scan_dirs: List[Path] = []
    if any(root_dir.glob("*.dcm")):
        return [root_dir]
    for child in sorted(root_dir.iterdir()):
        if child.is_dir():
            scan_dirs.extend(find_scan_dirs(child))
    return scan_dirs


def summarize_scan(scan_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "num_files": len(scan_data),
        "scan_label": None,
        "views": {},
        "total_counts": 0.0,
        "median_counts": None,
    }
    if not scan_data:
        return summary

    scan_label = scan_data[0].get("scan_label")
    summary["scan_label"] = scan_label
    view_sums = {}
    pixel_sums = []
    for item in scan_data:
        view = item.get("view", "unknown")
        view_sums.setdefault(view, 0.0)
        if item["pixel_sum"] is not None:
            view_sums[view] += item["pixel_sum"]
            pixel_sums.append(item["pixel_sum"])
    summary["views"] = view_sums
    summary["total_counts"] = float(sum(pixel_sums)) if pixel_sums else 0.0
    summary["median_counts"] = float(np.median(pixel_sums)) if pixel_sums else None
    return summary


def parse_dicom_datetime(study_date: Any, acquisition_time: Any = None) -> Optional[datetime]:
    if not study_date:
        return None
    date_text = str(study_date).strip()
    time_text = str(acquisition_time or "000000").strip().split(".")[0]
    time_text = time_text.ljust(6, "0")[:6]
    try:
        return datetime.strptime(date_text + time_text, "%Y%m%d%H%M%S")
    except ValueError:
        return None


def scan_datetime(scan: Dict[str, Any]) -> Optional[datetime]:
    datetimes = []
    for item in scan.get("images", []):
        parsed = parse_dicom_datetime(item.get("study_date"), item.get("acquisition_time"))
        if parsed is not None:
            datetimes.append(parsed)
    return min(datetimes) if datetimes else None


def ordered_scan_labels(patient_scans: Dict[str, Any]) -> List[str]:
    scans = patient_scans["scans"]
    return [
        scan["scan_name"]
        for scan in sorted(scans, key=lambda scan: (scan_datetime(scan) or datetime.max, scan["scan_name"]))
    ]


def scan_day_offsets(patient_scans: Dict[str, Any], scan_labels: List[str]) -> List[float]:
    scans_by_name = {scan["scan_name"]: scan for scan in patient_scans["scans"]}
    datetimes = {label: scan_datetime(scans_by_name[label]) for label in scan_labels}
    first_datetime = min((dt for dt in datetimes.values() if dt is not None), default=None)
    if first_datetime is None:
        return [float(index) for index, _ in enumerate(scan_labels)]
    return [
        0.0 if datetimes[label] is None else (datetimes[label] - first_datetime).total_seconds() / 86400.0
        for label in scan_labels
    ]


def short_scan_label(scan_name: str) -> str:
    match = re.search(r"NM_(\d{4})-(\d{2})-(\d{2})_(\d{2})(\d{2})(\d{2})", scan_name)
    if match:
        year, month, day, hour, minute, _second = match.groups()
        return f"{month}-{day} {hour}:{minute}"
    return scan_name


def build_scan_collection(scan_path: Path, scan_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "patient_path": scan_path.parent,
        "patient_id": scan_path.parent.name,
        "scans": [
            {
                "scan_name": scan_path.name,
                "scan_path": scan_path,
                "images": scan_data,
                "summary": summarize_scan(scan_data),
            }
        ],
    }


def plot_scan_images(scan_data: List[Dict[str, Any]], title: Optional[str] = None) -> None:
    """Plot the images from a scan directory with metadata annotations."""
    if not scan_data:
        raise ValueError("scan_data must contain at least one image")

    energy_order = {"Lower Scatter": 0, "Photopeak": 1, "Upper Scatter": 2}
    view_order = {"AP": 0, "ant": 0, "PA": 1, "post": 1}

    def sort_key(item: Dict[str, Any]) -> tuple:
        energy = item.get("energy_window") or item.get("energy_window_name") or ""
        view = item.get("view", "unknown")
        return (
            view_order.get(view, len(view_order)),
            energy_order.get(energy, max(energy_order.values()) + 1),
        )

    sorted_data = sorted(scan_data, key=sort_key)

    expanded_data = []
    for item in sorted_data:
        image = item.get("image")
        if isinstance(image, np.ndarray) and image.ndim == 3:
            num_frames = image.shape[0]
            # Show all frames for small stacks, otherwise show representative frames.
            frame_indices = list(range(num_frames)) if num_frames <= 12 else [0, num_frames // 2, num_frames - 1]
            for frame_idx in frame_indices:
                new_item = item.copy()
                new_item["image"] = image[frame_idx]
                new_item["frame_index"] = frame_idx + 1
                new_item["frame_count"] = num_frames
                expanded_data.append(new_item)
        else:
            item.setdefault("frame_index", None)
            item.setdefault("frame_count", None)
            expanded_data.append(item)

    sorted_data = expanded_data
    max_images = 24
    if len(sorted_data) > max_images:
        sampled_data = []
        groups: Dict[tuple, List[Dict[str, Any]]] = {}
        for item in sorted_data:
            group_key = (
                item.get("view", "unknown"),
                item.get("energy_window") or item.get("energy_window_name") or "Unknown Window",
            )
            groups.setdefault(group_key, []).append(item)

        for group_items in groups.values():
            if len(group_items) <= 3:
                sampled_data.extend(group_items)
            else:
                sampled_data.extend(
                    [
                        group_items[0],
                        group_items[len(group_items) // 2],
                        group_items[-1],
                    ]
                )
        sorted_data = sorted(sampled_data, key=sort_key)
        title = f"{title} (representative frames)" if title else "Representative frames"

    n = len(sorted_data)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.5 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    axes = axes.reshape(nrows, ncols)

    for i, item in enumerate(sorted_data):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]
        image = item.get("image")
        if image is None:
            ax.text(0.5, 0.5, "No image", ha="center", va="center")
            ax.set_axis_off()
            continue

        frame_info = ""
        if item.get("frame_index") is not None and item.get("frame_count") is not None:
            frame_info = f" (frame {item['frame_index']}/{item['frame_count']})"

        ax.imshow(image, cmap="gray", aspect="auto")
        energy = item.get("energy_window") or item.get("energy_window_name") or "Unknown Window"
        ax.set_title(
            f"{item.get('view')} {energy}{frame_info}\n{item.get('series_description')}\ncounts={item.get('pixel_sum'):.0f}"
        )
        ax.axis("off")

    for j in range(i + 1, nrows * ncols):
        row = j // ncols
        col = j % ncols
        axes[row, col].set_axis_off()

    if title:
        fig.suptitle(title, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_summary_grid(patient_scans: Dict[str, Any]) -> None:
    """Plot a structured overview of counts per scan and energy window."""
    rows = []
    for scan in patient_scans["scans"]:
        scan_name = scan["scan_name"]
        for item in scan["images"]:
            rows.append(
                {
                    "scan": scan_name,
                    "energy_window": item.get("energy_window"),
                    "counts": item.get("pixel_sum", 0.0),
                }
            )

    if not rows:
        raise ValueError("No image data available for plotting summary grid")

    scan_labels = ordered_scan_labels(patient_scans)
    default_order = ["Lower Scatter", "Photopeak", "Upper Scatter"]
    energy_windows = [energy for energy in default_order if energy in {row["energy_window"] for row in rows}]
    extra = sorted({row["energy_window"] for row in rows if row["energy_window"] not in default_order})
    energy_windows.extend(extra)
    if not energy_windows:
        energy_windows = ["Unknown Window"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(scan_labels))
    width = 0.8 / max(len(energy_windows), 1)

    for i, energy in enumerate(energy_windows):
        energy_counts = []
        for scan_name in scan_labels:
            total = sum(
                row["counts"]
                for row in rows
                if row["scan"] == scan_name and row["energy_window"] == energy
            )
            energy_counts.append(total)
        ax.bar(x + (i - (len(energy_windows) - 1) / 2) * width, energy_counts, width, label=energy)

    ax.set_xticks(x)
    ax.set_xticklabels([short_scan_label(label) for label in scan_labels], rotation=20, ha="right")
    ax.set_xlabel("Acquisition")
    ax.set_ylabel("Total counts")
    ax.set_title("Counts par acquisition et fenêtre d'énergie")
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_decay_curve(patient_scans: Dict[str, Any]) -> None:
    """Plot the count decay across scan time points, both total and by energy window."""
    rows = []
    for scan in patient_scans["scans"]:
        scan_name = scan["scan_name"]
        for item in scan["images"]:
            rows.append(
                {
                    "scan": scan_name,
                    "energy_window": item.get("energy_window"),
                    "counts": item.get("pixel_sum", 0.0),
                }
            )

    if not rows:
        raise ValueError("No image data available for plotting decay curve")

    scan_labels = ordered_scan_labels(patient_scans)
    day_offsets = scan_day_offsets(patient_scans, scan_labels)
    default_order = ["Lower Scatter", "Photopeak", "Upper Scatter"]
    energy_windows = [energy for energy in default_order if energy in {row["energy_window"] for row in rows}]
    energy_windows.extend(sorted({row["energy_window"] for row in rows if row["energy_window"] not in default_order}))
    totals_by_scan = {
        scan_name: sum(row["counts"] for row in rows if row["scan"] == scan_name)
        for scan_name in scan_labels
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.array(day_offsets)
    ax.plot(x, [totals_by_scan[scan] for scan in scan_labels], marker="o", color="black", linewidth=2, label="Total (ant+post)")

    for energy in energy_windows:
        energy_counts = [
            sum(row["counts"] for row in rows if row["scan"] == scan_name and row["energy_window"] == energy)
            for scan_name in scan_labels
        ]
        ax.plot(x, energy_counts, marker="o", linestyle="--", label=energy)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{day:g}" for day in day_offsets])
    ax.set_xlabel("Jour depuis la première acquisition")
    ax.set_ylabel("Counts")
    ax.set_title("Décroissance des counts au fil des jours")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    plt.show()

    if totals_by_scan[scan_labels[0]] > 0:
        normalized = [totals_by_scan[scan] / totals_by_scan[scan_labels[0]] for scan in scan_labels]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(x, normalized, marker="o", color="tab:orange")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{day:g}" for day in day_offsets])
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Jour depuis la première acquisition")
        ax.set_ylabel("Fraction du jour 0")
        ax.set_title("Évolution relative des counts totaux")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        plt.show()


def load_patient_scans(patient_dir: Path, scan_name_filter: Optional[str] = None) -> Dict[str, Any]:
    patient_dir = patient_dir.expanduser().resolve()
    scans = []
    for scan_dir in find_scan_dirs(patient_dir):
        scan_name = scan_dir.relative_to(patient_dir).as_posix()
        if scan_name_filter and scan_name_filter.lower() not in scan_name.lower():
            continue
        scan_data = load_scan_directory(scan_dir)
        scans.append(
            {
                "scan_name": scan_name,
                "scan_path": scan_dir,
                "images": scan_data,
                "summary": summarize_scan(scan_data),
            }
        )
    return {
        "patient_path": patient_dir,
        "patient_id": patient_dir.name,
        "scans": scans,
    }


def find_patient_dirs(base_dir: Path) -> List[Path]:
    base_dir = base_dir.expanduser().resolve()
    return sorted([p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("patient_")])


def load_patients(base_dir: Path, scan_name_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    patients = []
    for patient_dir in find_patient_dirs(base_dir):
        patients.append(load_patient_scans(patient_dir, scan_name_filter=scan_name_filter))
    return patients


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    default_path = (
        base
        / "2026-05__Studies"
        / "DOE^JOHN_ANON62096_NM_2026-05-21_082240_MN.LU177.POST.TRAITEMENT-EN_WB.RAPIDE_n"
    )
    if not default_path.exists():
        default_path = base / "2026-05__Studies" 
    target_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path
    target_path = target_path.expanduser().resolve()

    if target_path.is_file():
        print(f"Loading DICOM file: {target_path}")
        scan_data = [load_dicom_file(target_path)]
        num_frames = scan_data[0]["image"].shape[0] if isinstance(scan_data[0]["image"], np.ndarray) and scan_data[0]["image"].ndim == 3 else 1
        print(f"Loaded 1 DICOM file from {target_path.name} containing {num_frames} frame(s)")
        plot_scan_images(scan_data, title=f"{target_path.name}")
    elif target_path.is_dir():
        print(f"Loading DICOM folder: {target_path}")
        # Check if this directory contains DICOM files directly
        if any(target_path.glob("*.dcm")):
            # Single scan directory
            scan_data = load_scan_directory(target_path)
            if len(scan_data) == 1 and isinstance(scan_data[0]["image"], np.ndarray) and scan_data[0]["image"].ndim == 3:
                num_frames = scan_data[0]["image"].shape[0]
                print(f"Loaded 1 DICOM file containing {num_frames} frame(s) from folder {target_path.name}")
            else:
                print(f"Loaded {len(scan_data)} image file(s) from folder {target_path.name}")
            if scan_data:
                print(scan_data[0]["series_description"])
                plot_scan_images(scan_data, title=f"{target_path.name}")
                scan_collection = build_scan_collection(target_path, scan_data)
                print("\nPlotting counts by energy window...")
                plot_summary_grid(scan_collection)
                print("\nPlotting decay curve...")
                plot_decay_curve(scan_collection)
        else:
            # Patient directory with multiple timepoint scans
            print(f"Loading patient dataset from {target_path.name}")
            scan_name_filter = "rapide" if target_path.name.lower() == "wb" else None
            if scan_name_filter:
                print(f"Filtering scans to names containing: {scan_name_filter}")
            patient_scans = load_patient_scans(target_path, scan_name_filter=scan_name_filter)
            total_images = sum(len(scan["images"]) for scan in patient_scans["scans"])
            print(f"Loaded {len(patient_scans['scans'])} scan timepoint(s) with {total_images} total image file(s)")
            
            # Plot images grouped by scan timepoint
            for scan in patient_scans["scans"]:
                if scan["images"]:
                    scan_title = f"{scan['scan_name']}"
                    print(f"\nPlotting {len(scan['images'])} image(s) from {scan_title}")
                    plot_scan_images(scan["images"], title=scan_title)
            
            # Plot summary statistics
            if total_images > 0:
                print(f"\nPlotting counts summary...")
                plot_summary_grid(patient_scans)
                print(f"\nPlotting decay curves...")
                plot_decay_curve(patient_scans)
    else:
        raise FileNotFoundError(f"Path not found: {target_path}")

    print("\nFinished loading and plotting DICOM data.")
