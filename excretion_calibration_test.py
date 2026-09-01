"""Exploratory one-patient planar calibration from measured excretion.

The urine/diaper mass balance is used as a scalar whole-body reference.  Blood
concentrations are retained as a kinetic diagnostic only: without a validated
distribution volume or compartment model, kBq/mL cannot be converted into a
whole-body activity in MBq.

Day1 is the pre-specified calibration point because its planar acquisition is
close to the 22.95 h completed urine collection while counts remain higher than
at Day2.  Q/SPECT is never used to calculate the calibration factor.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sang_urine


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "fig" / "excretion_calibration"
FIGURE_PATH = OUTPUT_DIR / "urine_calibrated_planar_vs_old_qspect_excretion.png"
VALUES_PATH = OUTPUT_DIR / "urine_calibration_values.csv"
REPORT_PATH = OUTPUT_DIR / "urine_calibration_report.txt"
CALIBRATION_LABEL = "Day1"


def calculate_urine_calibration(
    data: Dict[str, Any],
    balance: Dict[str, np.ndarray],
    imaging: Dict[str, np.ndarray],
    calibration_label: str = CALIBRATION_LABEL,
) -> Dict[str, Any]:
    """Scale the existing planar CTAC activity to the urine balance at one day."""
    labels = [str(label) for label in imaging["labels"]]
    if calibration_label not in labels:
        raise ValueError(f"Unknown calibration label: {calibration_label}")

    calibration_index = labels.index(calibration_label)
    planar_times_h = np.asarray(imaging["planar_times_h"], dtype=float)
    qspect_times_h = np.asarray(imaging["qspect_times_h"], dtype=float)
    old_planar_mbq = np.asarray(imaging["planar_ctac_mbq"], dtype=float)
    qspect_mbq = np.asarray(imaging["qspect_mbq"], dtype=float)
    reference_at_planar_mbq = sang_urine.predicted_body_activity(
        planar_times_h, data, balance
    )
    reference_at_qspect_mbq = sang_urine.predicted_body_activity(
        qspect_times_h, data, balance
    )

    first_collection_h = float(balance["times_h"][0])
    last_collection_h = float(balance["times_h"][-1])
    calibration_time_h = float(planar_times_h[calibration_index])
    if calibration_time_h < first_collection_h:
        raise ValueError("Calibration requires at least one completed urine collection")

    calibration_reference_mbq = float(reference_at_planar_mbq[calibration_index])
    calibration_planar_mbq = float(old_planar_mbq[calibration_index])
    if calibration_planar_mbq <= 0.0:
        raise ValueError("Planar calibration activity must be positive")
    calibration_factor = calibration_reference_mbq / calibration_planar_mbq
    new_planar_mbq = calibration_factor * old_planar_mbq

    rows: List[Dict[str, Any]] = []
    for index, label in enumerate(labels):
        planar_reference_available = bool(planar_times_h[index] >= first_collection_h)
        qspect_reference_available = bool(qspect_times_h[index] >= first_collection_h)
        planar_reference = (
            float(reference_at_planar_mbq[index])
            if planar_reference_available
            else math.nan
        )
        qspect_reference = (
            float(reference_at_qspect_mbq[index])
            if qspect_reference_available
            else math.nan
        )
        rows.append(
            {
                "label": label,
                "is_calibration_point": label == calibration_label,
                "planar_time_h": float(planar_times_h[index]),
                "qspect_time_h": float(qspect_times_h[index]),
                "old_planar_ctac_mbq": float(old_planar_mbq[index]),
                "new_urine_calibrated_planar_mbq": float(new_planar_mbq[index]),
                "excretion_reference_at_planar_time_mbq": planar_reference,
                "qspect_mbq": float(qspect_mbq[index]),
                "excretion_reference_at_qspect_time_mbq": qspect_reference,
                "old_planar_over_excretion_reference": float(
                    old_planar_mbq[index] / planar_reference
                )
                if planar_reference_available
                else math.nan,
                "new_planar_over_excretion_reference": float(
                    new_planar_mbq[index] / planar_reference
                )
                if planar_reference_available
                else math.nan,
                "qspect_over_excretion_reference": float(
                    qspect_mbq[index] / qspect_reference
                )
                if qspect_reference_available
                else math.nan,
                "after_last_urine_collection": bool(
                    planar_times_h[index] > last_collection_h
                ),
            }
        )

    return {
        "calibration_label": calibration_label,
        "calibration_index": calibration_index,
        "calibration_factor": float(calibration_factor),
        "calibration_time_h": calibration_time_h,
        "calibration_reference_mbq": calibration_reference_mbq,
        "calibration_planar_mbq": calibration_planar_mbq,
        "rows": rows,
    }


def plot_calibration_comparison(
    data: Dict[str, Any],
    balance: Dict[str, np.ndarray],
    imaging: Dict[str, np.ndarray],
    calibration: Dict[str, Any],
    output_path: Path = FIGURE_PATH,
) -> Path:
    """Compare old/new planar calibration, Q/SPECT and excretion retention."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    maximum_time_h = max(
        float(np.max(imaging["planar_times_h"])),
        float(np.max(imaging["qspect_times_h"])),
    )
    first_collection_h = float(balance["times_h"][0])
    last_collection_h = float(balance["times_h"][-1])
    curve_times_h = np.linspace(first_collection_h, maximum_time_h, 1000)
    excretion_curve_mbq = sang_urine.predicted_body_activity(
        curve_times_h, data, balance
    )

    rows = calibration["rows"]
    planar_times_h = np.asarray([row["planar_time_h"] for row in rows])
    old_planar_mbq = np.asarray([row["old_planar_ctac_mbq"] for row in rows])
    new_planar_mbq = np.asarray(
        [row["new_urine_calibrated_planar_mbq"] for row in rows]
    )
    qspect_times_h = np.asarray([row["qspect_time_h"] for row in rows])
    qspect_mbq = np.asarray([row["qspect_mbq"] for row in rows])

    fig, axis = plt.subplots(figsize=(8.8, 5.8), facecolor="white", layout="constrained")
    axis.axvspan(
        last_collection_h,
        maximum_time_h,
        color="#7f7f7f",
        alpha=0.08,
        label="Après 46,62 h : excrétion additionnelle non mesurée",
    )
    axis.plot(
        curve_times_h,
        excretion_curve_mbq,
        color="#2ca02c",
        linewidth=2.2,
        label="Activité corporelle restante — bilan d'excrétion",
    )
    axis.scatter(
        balance["times_h"],
        balance["predicted_body_mbq"],
        color="#2ca02c",
        edgecolor="white",
        linewidth=0.7,
        s=40,
        zorder=4,
        label="Fins des collectes urine/couches",
    )
    axis.plot(
        planar_times_h,
        old_planar_mbq,
        color="#1f77b4",
        marker="s",
        linewidth=2.0,
        label="Planaire CTAC — calibration actuelle",
    )
    axis.plot(
        planar_times_h,
        new_planar_mbq,
        color="#ff7f0e",
        marker="o",
        linewidth=2.2,
        label=(
            "Planaire — calibration urine J1 "
            f"(k={calibration['calibration_factor']:.3f})"
        ),
    )
    axis.plot(
        qspect_times_h,
        qspect_mbq,
        color="black",
        marker="D",
        linewidth=2.0,
        label="Q/SPECT — comparaison indépendante",
    )
    calibration_index = int(calibration["calibration_index"])
    axis.scatter(
        [planar_times_h[calibration_index]],
        [new_planar_mbq[calibration_index]],
        marker="*",
        s=180,
        facecolor="#ff7f0e",
        edgecolor="black",
        linewidth=0.8,
        zorder=6,
        label="Point de calibration J1",
    )
    axis.set_xlabel("Temps après l'injection (h)")
    axis.set_ylabel("Activité (MBq)")
    axis.set_title("Calibration planaire par le bilan d'excrétion")
    axis.grid(True, linestyle="--", alpha=0.3)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, fontsize=8)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def write_values(calibration: Dict[str, Any], output_path: Path = VALUES_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = calibration["rows"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: round(float(value), 6)
                    if isinstance(value, (float, np.floating)) and np.isfinite(value)
                    else value
                    for key, value in row.items()
                }
            )
    return output_path


def write_report(
    data: Dict[str, Any],
    balance: Dict[str, np.ndarray],
    calibration: Dict[str, Any],
    output_path: Path = REPORT_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = calibration["rows"]
    day2_row = next((row for row in rows if row["label"] == "Day2"), None)
    day2_factor = (
        day2_row["excretion_reference_at_planar_time_mbq"]
        / day2_row["old_planar_ctac_mbq"]
        if day2_row is not None
        else math.nan
    )
    lines = [
        "Exploratory urine-based planar calibration",
        "============================================",
        "",
        "Calibration target:",
        "  Remaining whole-body activity from injected activity minus completed",
        "  urine/diaper collections, with every activity on the same physical-decay",
        "  time reference.",
        "",
        "Blood decision:",
        "  Blood is not used to calibrate whole-body MBq. It is a concentration",
        "  measurement (kBq/mL), and converting it requires a validated distribution",
        "  volume or compartment model. It remains a kinetic diagnostic.",
        "",
        f"Calibration point: {calibration['calibration_label']}",
        f"Calibration time: {calibration['calibration_time_h']:.3f} h after injection",
        f"Old planar CTAC activity: {calibration['calibration_planar_mbq']:.2f} MBq",
        f"Excretion reference: {calibration['calibration_reference_mbq']:.2f} MBq",
        f"Multiplicative calibration factor: {calibration['calibration_factor']:.6f}",
        f"Day2 single-point sensitivity factor: {day2_factor:.6f}",
        "",
        "Important limitations:",
        "  Q/SPECT is not used in the calibration; it is an independent comparison.",
        "  Day0 has no completed urine collection at either imaging time.",
        "  After 46.62 h, later excretion is unmeasured, so the green curve is an",
        "  upper bound rather than a complete activity balance.",
        "  This scalar calibrates the complete current planar CTAC scale; it does not",
        "  isolate attenuation from sensitivity, crop, TEW, or geometry errors.",
        "",
        " label | old planar | new planar | excretion ref at planar | Q/SPECT | excretion ref at Q/SPECT",
        "------ | ---------- | ---------- | ----------------------- | ------- | -------------------------",
    ]
    for row in rows:
        lines.append(
            f"{row['label']:6s} | {row['old_planar_ctac_mbq']:10.2f} | "
            f"{row['new_urine_calibrated_planar_mbq']:10.2f} | "
            f"{row['excretion_reference_at_planar_time_mbq']:23.2f} | "
            f"{row['qspect_mbq']:7.2f} | "
            f"{row['excretion_reference_at_qspect_time_mbq']:25.2f}"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def run_calibration_test(
    output_dir: Path = OUTPUT_DIR,
    calibration_label: str = CALIBRATION_LABEL,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = sang_urine.load_sang_urine_csv()
    balance = sang_urine.calculate_urine_mass_balance(data)
    imaging = sang_urine.load_imaging_comparison(data["injection_datetime"])
    if imaging is None:
        raise FileNotFoundError("The planar/QSPECT comparison table is unavailable")
    calibration = calculate_urine_calibration(
        data, balance, imaging, calibration_label=calibration_label
    )
    paths = {
        "figure": plot_calibration_comparison(
            data,
            balance,
            imaging,
            calibration,
            output_dir / FIGURE_PATH.name,
        ),
        "values": write_values(calibration, output_dir / VALUES_PATH.name),
        "report": write_report(
            data, balance, calibration, output_dir / REPORT_PATH.name
        ),
    }
    return {
        "data": data,
        "balance": balance,
        "imaging": imaging,
        "calibration": calibration,
        "paths": paths,
    }


def main() -> None:
    result = run_calibration_test()
    calibration = result["calibration"]
    print(
        f"Urine calibration {calibration['calibration_label']}: "
        f"k={calibration['calibration_factor']:.6f}"
    )
    for key, path in result["paths"].items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
