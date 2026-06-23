from pathlib import Path
import re
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


def canonical_energy_window_label(label: str) -> str:
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


def parse_energy_window(series_description: str, ds: pydicom.dataset.FileDataset) -> str:
    """Extract a readable energy window label from DICOM metadata."""
    if hasattr(ds, "EnergyWindowInformationSequence") and ds.EnergyWindowInformationSequence:
        item = ds.EnergyWindowInformationSequence[0]
        name = getattr(item, "EnergyWindowName", None)
        if name:
            return canonical_energy_window_label(str(name))
        lower = getattr(item, "EnergyWindowLowerLimit", None)
        upper = getattr(item, "EnergyWindowUpperLimit", None)
        if lower is not None and upper is not None:
            return canonical_energy_window_label(f"{lower:.0f}-{upper:.0f} keV")
    if re.search(r"lower", series_description, re.I):
        return "Lower Scatter"
    if re.search(r"upper", series_description, re.I):
        return "Upper Scatter"
    if re.search(r"lutetium|177|photo|main", series_description, re.I):
        return "Photopeak"
    return "Unknown Window"


def extract_energy_window(ds: pydicom.dataset.FileDataset) -> Dict[str, Any]:
    energy_window = {
        "energy_window_name": None,
        "energy_window_lower_limit": None,
        "energy_window_upper_limit": None,
    }
    if not hasattr(ds, "EnergyWindowInformationSequence") or not ds.EnergyWindowInformationSequence:
        return energy_window

    item = ds.EnergyWindowInformationSequence[0]
    energy_window["energy_window_name"] = getattr(item, "EnergyWindowName", None)
    energy_window["energy_window_lower_limit"] = float(getattr(item, "EnergyWindowLowerLimit", None)) if getattr(item, "EnergyWindowLowerLimit", None) is not None else None
    energy_window["energy_window_upper_limit"] = float(getattr(item, "EnergyWindowUpperLimit", None)) if getattr(item, "EnergyWindowUpperLimit", None) is not None else None
    return energy_window


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
    return [load_dicom_file(path) for path in files if is_dicom_file(path)]


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


def plot_scan_images(scan_data: List[Dict[str, Any]], title: Optional[str] = None) -> None:
    """Plot the images from a scan directory with metadata annotations."""
    if not scan_data:
        raise ValueError("scan_data must contain at least one image")

    energy_order = {"Lower Scatter": 0, "Photopeak": 1, "Upper Scatter": 2}
    view_order = {"ant": 0, "post": 1}

    def sort_key(item: Dict[str, Any]) -> tuple:
        energy = item.get("energy_window") or item.get("energy_window_name") or ""
        view = item.get("view", "unknown")
        return (
            view_order.get(view, len(view_order)),
            energy_order.get(energy, max(energy_order.values()) + 1),
        )

    sorted_data = sorted(scan_data, key=sort_key)

    n = len(sorted_data)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
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

        ax.imshow(image, cmap="gray", aspect="auto")
        energy = item.get("energy_window") or item.get("energy_window_name") or "Unknown Window"
        ax.set_title(
            f"{item.get('view')} {energy}\n{item.get('series_description')}\ncounts={item.get('pixel_sum'):.0f}"
        )
        ax.axis("off")

    for j in range(i + 1, nrows * ncols):
        row = j // ncols
        col = j % ncols
        axes[row, col].set_axis_off()

    if title:
        fig.suptitle(title)
    fig.tight_layout()
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

    scan_labels = sorted({row["scan"] for row in rows})
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
    ax.set_xticklabels(scan_labels)
    ax.set_xlabel("Scan time point")
    ax.set_ylabel("Total counts")
    ax.set_title("Counts by time point and energy window")
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

    scan_labels = sorted({row["scan"] for row in rows})
    default_order = ["Lower Scatter", "Photopeak", "Upper Scatter"]
    energy_windows = [energy for energy in default_order if energy in {row["energy_window"] for row in rows}]
    energy_windows.extend(sorted({row["energy_window"] for row in rows if row["energy_window"] not in default_order}))
    totals_by_scan = {
        scan_name: sum(row["counts"] for row in rows if row["scan"] == scan_name)
        for scan_name in scan_labels
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(scan_labels))
    ax.plot(x, [totals_by_scan[scan] for scan in scan_labels], marker="o", color="black", linewidth=2, label="Total (ant+post)")

    for energy in energy_windows:
        energy_counts = [
            sum(row["counts"] for row in rows if row["scan"] == scan_name and row["energy_window"] == energy)
            for scan_name in scan_labels
        ]
        ax.plot(x, energy_counts, marker="o", linestyle="--", label=energy)

    ax.set_xticks(x)
    ax.set_xticklabels(scan_labels)
    ax.set_xlabel("Scan time point")
    ax.set_ylabel("Counts")
    ax.set_title("Décroissance des counts par scan")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    plt.show()

    if totals_by_scan[scan_labels[0]] > 0:
        normalized = [totals_by_scan[scan] / totals_by_scan[scan_labels[0]] for scan in scan_labels]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(x, normalized, marker="o", color="tab:orange")
        ax.set_xticks(x)
        ax.set_xticklabels(scan_labels)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Scan time point")
        ax.set_ylabel("Fraction of scan1")
        ax.set_title("Évolution relative des counts totaux")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        plt.show()


def load_patient_scans(patient_dir: Path) -> Dict[str, Any]:
    patient_dir = patient_dir.expanduser().resolve()
    scans = []
    for scan_dir in find_scan_dirs(patient_dir):
        scan_data = load_scan_directory(scan_dir)
        scans.append(
            {
                "scan_name": scan_dir.relative_to(patient_dir).as_posix(),
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


def load_patients(base_dir: Path) -> List[Dict[str, Any]]:
    patients = []
    for patient_dir in find_patient_dirs(base_dir):
        patients.append(load_patient_scans(patient_dir))
    return patients


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    patient_dir = base / "1j92g763g" / "patient_4" / "PlanarWholeBodyScans"
    if not patient_dir.exists():
        raise FileNotFoundError(f"Patient folder not found: {patient_dir}")

    scans = load_patient_scans(patient_dir)
    print(f"Loaded {len(scans['scans'])} scans from {scans['patient_id']}")
    for scan in scans["scans"]:
        print(scan["scan_name"], scan["summary"])
        if scan["images"]:
            plot_scan_images(scan["images"], title=f"{scan['scan_name']} images")

    if scans["scans"]:
        plot_summary_grid(scans)
        plot_decay_curve(scans)

    first_scan = scans["scans"][0]
    if first_scan["images"]:
        ds = pydicom.dcmread(first_scan["images"][0]["path"])
        print("\n=== DICOM Dataset object ===")
        print(ds)
        print("=== End DICOM Dataset object ===\n")

    print("Finished loading and plotting patient scans.")
