"""Analyse exploratory blood and urine measurements for the P-011 Lu-177 study.

The raw multi-table CSV is kept unchanged. Activities measured in urine and
diapers are first placed on an injection-time-equivalent scale before a
decay-consistent whole-body mass balance is calculated. Blood measurements are
reported as concentration only; they are not counted as excreted activity.
"""

from __future__ import annotations

import csv
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import attenuation_correction as ac
import planar_processing
import qspect_processing


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_CSV = PROJECT_DIR / "Data" / "2026-05_studies" / "P-011csv.csv"
DEFAULT_IMAGING_CSV = (
    PROJECT_DIR / "fig" / "crop_method_comparison" / "crop_methods_values.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "fig" / "sang_urine"


def _cell(row: List[str], index: int) -> str:
    return row[index].strip() if index < len(row) else ""


def _number(row: List[str], index: int) -> Optional[float]:
    text = _cell(row, index).replace(" ", "").replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _datetime(text: str) -> datetime:
    return datetime.strptime(text.strip(), "%Y-%m-%d %H:%M")


def _urine_hour_key(text: str) -> Optional[int]:
    match = re.search(r"urine\s*(\d+)\s*hr", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _diaper_hour_key(text: str) -> Optional[int]:
    match = re.search(r"diaper.*\((\d+)h\)", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def load_sang_urine_csv(csv_path: Path = DEFAULT_SOURCE_CSV) -> Dict[str, Any]:
    """Parse the blood summary and urine/diaper summary from the source CSV."""
    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))

    injected_activity_mbq = None
    half_life_h = None
    injection_datetime = None
    for row in rows:
        key = _cell(row, 1).lower()
        if "injected acctivity" in key or "injected activity" in key:
            injected_activity_mbq = _number(row, 2)
        elif "half-life" in key:
            half_life_h = _number(row, 2)
        elif "time of injection" in key and _cell(row, 2):
            injection_datetime = _datetime(_cell(row, 2))

    if injected_activity_mbq is None or half_life_h is None or injection_datetime is None:
        raise ValueError("Injected activity, half-life or injection time is missing")

    blood = []
    for row in rows:
        label = _cell(row, 11)
        concentration_kbq_ml = _number(row, 17)
        elapsed_h = _number(row, 18)
        if not label.lower().startswith("blood") or concentration_kbq_ml is None:
            continue
        if elapsed_h is None:
            continue
        blood.append(
            {
                "label": label,
                "elapsed_h": elapsed_h,
                "concentration_kbq_ml": concentration_kbq_ml,
                "concentration_mbq_ml": concentration_kbq_ml / 1000.0,
                "sampling_datetime": _datetime(_cell(row, 13)),
                "counting_datetime": _datetime(_cell(row, 14)),
            }
        )
    blood.sort(key=lambda item: item["elapsed_h"])

    urine_volumes_ml: Dict[int, float] = {}
    diaper_activities_mbq: Dict[int, float] = {}
    for row in rows:
        raw_label = _cell(row, 6)
        hour_key = _urine_hour_key(raw_label)
        volume_ml = _number(row, 11)
        if hour_key is not None and "(a)" in raw_label.lower() and volume_ml is not None:
            urine_volumes_ml[hour_key] = volume_ml
        diaper_key = _diaper_hour_key(_cell(row, 13))
        diaper_activity = _number(row, 18)
        if diaper_key is not None and diaper_activity is not None:
            diaper_activities_mbq[diaper_key] = diaper_activity

    urine = []
    for row in rows:
        label = _cell(row, 13)
        hour_key = _urine_hour_key(label)
        urine_mbq = _number(row, 18)
        total_mbq = _number(row, 19)
        elapsed_h = _number(row, 21)
        if hour_key is None or urine_mbq is None or total_mbq is None or elapsed_h is None:
            continue
        diaper_mbq = diaper_activities_mbq.get(hour_key, total_mbq - urine_mbq)
        urine.append(
            {
                "label": label,
                "nominal_collection_h": hour_key,
                "elapsed_h": elapsed_h,
                "urine_volume_ml": urine_volumes_ml.get(hour_key, math.nan),
                "urine_activity_mbq": urine_mbq,
                "diaper_activity_mbq": diaper_mbq,
                "total_excreted_activity_mbq": total_mbq,
                "component_sum_difference_mbq": urine_mbq + diaper_mbq - total_mbq,
                "naive_remaining_activity_mbq": _number(row, 20),
                "sampling_datetime": _datetime(_cell(row, 15)),
            }
        )
    urine.sort(key=lambda item: item["elapsed_h"])

    if len(blood) != 9:
        raise ValueError(f"Expected 9 blood measurements, found {len(blood)}")
    if len(urine) != 6:
        raise ValueError(f"Expected 6 urine intervals, found {len(urine)}")

    return {
        "source_csv": csv_path,
        "injected_activity_mbq": float(injected_activity_mbq),
        "half_life_h": float(half_life_h),
        "injection_datetime": injection_datetime,
        "blood": blood,
        "urine": urine,
    }


def decay_factor(elapsed_h: np.ndarray | float, half_life_h: float) -> np.ndarray:
    """Return the remaining physical activity fraction after ``elapsed_h``."""
    elapsed = np.asarray(elapsed_h, dtype=np.float64)
    return np.exp(-math.log(2.0) * elapsed / float(half_life_h))


def calculate_urine_mass_balance(data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Calculate a decay-consistent balance at each urine collection time.

    Each interval activity is measured at its own collection time. It is first
    decay-compensated to injection time. The cumulative amount is then decayed
    forward to the time at which the balance is evaluated.
    """
    times_h = np.asarray([item["elapsed_h"] for item in data["urine"]], dtype=float)
    urine_mbq = np.asarray(
        [item["urine_activity_mbq"] for item in data["urine"]], dtype=float
    )
    diaper_mbq = np.asarray(
        [item["diaper_activity_mbq"] for item in data["urine"]], dtype=float
    )
    interval_mbq = np.asarray(
        [item["total_excreted_activity_mbq"] for item in data["urine"]], dtype=float
    )
    half_life_h = float(data["half_life_h"])
    injected_mbq = float(data["injected_activity_mbq"])

    interval_injection_equivalent_mbq = interval_mbq / decay_factor(
        times_h, half_life_h
    )
    cumulative_injection_equivalent_mbq = np.cumsum(
        interval_injection_equivalent_mbq
    )
    physical_available_mbq = injected_mbq * decay_factor(times_h, half_life_h)
    cumulative_excreted_at_time_mbq = (
        cumulative_injection_equivalent_mbq * decay_factor(times_h, half_life_h)
    )
    predicted_body_mbq = physical_available_mbq - cumulative_excreted_at_time_mbq
    closure_error_mbq = (
        predicted_body_mbq
        + cumulative_excreted_at_time_mbq
        - physical_available_mbq
    )

    return {
        "times_h": times_h,
        "urine_activity_mbq": urine_mbq,
        "diaper_activity_mbq": diaper_mbq,
        "interval_excreted_mbq": interval_mbq,
        "interval_injection_equivalent_mbq": interval_injection_equivalent_mbq,
        "cumulative_injection_equivalent_mbq": cumulative_injection_equivalent_mbq,
        "physical_available_mbq": physical_available_mbq,
        "cumulative_excreted_at_time_mbq": cumulative_excreted_at_time_mbq,
        "predicted_body_mbq": predicted_body_mbq,
        "closure_error_mbq": closure_error_mbq,
    }


def predicted_body_activity(
    elapsed_h: np.ndarray | float,
    data: Dict[str, Any],
    balance: Dict[str, np.ndarray],
) -> np.ndarray:
    """Evaluate the stepwise urine-derived body estimate at arbitrary times."""
    evaluation_times = np.atleast_1d(np.asarray(elapsed_h, dtype=float))
    collection_times = balance["times_h"]
    interval_equivalent = balance["interval_injection_equivalent_mbq"]
    injected_mbq = float(data["injected_activity_mbq"])
    output = np.empty_like(evaluation_times)
    for index, time_h in enumerate(evaluation_times):
        removed_equivalent = float(interval_equivalent[collection_times <= time_h].sum())
        output[index] = (
            injected_mbq - removed_equivalent
        ) * decay_factor(time_h, float(data["half_life_h"]))
    return output


def cumulative_excreted_activity(
    elapsed_h: np.ndarray | float,
    data: Dict[str, Any],
    balance: Dict[str, np.ndarray],
) -> np.ndarray:
    """Return measured cumulative urine/diaper activity at evaluation times.

    Interval activities are first expressed at injection time, summed only when
    their collection time has passed, then decayed forward to the evaluation
    time. After the last collection this excludes any later unmeasured
    excretion.
    """
    evaluation_times = np.atleast_1d(np.asarray(elapsed_h, dtype=float))
    collection_times = balance["times_h"]
    interval_equivalent = balance["interval_injection_equivalent_mbq"]
    output = np.empty_like(evaluation_times)
    for index, time_h in enumerate(evaluation_times):
        removed_equivalent = float(interval_equivalent[collection_times <= time_h].sum())
        output[index] = removed_equivalent * decay_factor(
            time_h, float(data["half_life_h"])
        )
    return output


def _dicom_datetime(study_date: str, acquisition_time: str) -> datetime:
    clean_time = acquisition_time.split(".")[0].ljust(6, "0")
    return datetime.strptime(f"{study_date}{clean_time}", "%Y%m%d%H%M%S")


def load_imaging_comparison(
    injection_datetime: datetime,
    imaging_csv: Path = DEFAULT_IMAGING_CSV,
    crop_method: str = "fixed_j0",
) -> Optional[Dict[str, np.ndarray]]:
    """Load fixed-crop CTAC/QSPECT activities and their exact acquisition times."""
    imaging_csv = Path(imaging_csv)
    if not imaging_csv.exists():
        return None
    with imaging_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["method"] == crop_method]
    if not rows:
        return None
    rows.sort(key=lambda row: float(row["day"]))

    scans = ac.sorted_planar_scans(planar_processing.default_planar_study_dir())
    qspect_series = qspect_processing.load_qspect_study(
        qspect_processing.default_qspect_dir()
    )
    planar_times_h = np.asarray(
        [
            (
                _dicom_datetime(
                    scan["images"][0]["study_date"],
                    scan["images"][0]["acquisition_time"],
                )
                - injection_datetime
            ).total_seconds()
            / 3600.0
            for scan in scans[: len(rows)]
        ]
    )
    qspect_by_label = {
        item["label"]: item["acquisition_datetime"] for item in qspect_series
    }
    qspect_times_h = np.asarray(
        [
            (qspect_by_label[row["label"]] - injection_datetime).total_seconds()
            / 3600.0
            for row in rows
        ]
    )
    return {
        "labels": np.asarray([row["label"] for row in rows]),
        "planar_times_h": planar_times_h,
        "qspect_times_h": qspect_times_h,
        "planar_ctac_mbq": np.asarray(
            [float(row["ctac_activity_mbq"]) for row in rows]
        ),
        "qspect_mbq": np.asarray([float(row["qspect_activity_mbq"]) for row in rows]),
    }


def calculate_imaging_excretion_comparison(
    data: Dict[str, Any],
    balance: Dict[str, np.ndarray],
    imaging: Dict[str, np.ndarray],
) -> List[Dict[str, Any]]:
    """Combine each imaging activity with measured excretion at the same time."""
    rows: List[Dict[str, Any]] = []
    last_collection_h = float(balance["times_h"][-1])
    modalities = (
        (
            "planar_ctac_fixed_j0",
            imaging["planar_times_h"],
            imaging["planar_ctac_mbq"],
        ),
        ("qspect", imaging["qspect_times_h"], imaging["qspect_mbq"]),
    )
    for modality, times_h, body_activities_mbq in modalities:
        excreted_mbq = cumulative_excreted_activity(times_h, data, balance)
        physical_available_mbq = float(data["injected_activity_mbq"]) * decay_factor(
            times_h, float(data["half_life_h"])
        )
        predicted_body_mbq = physical_available_mbq - excreted_mbq
        closure_mbq = body_activities_mbq + excreted_mbq
        closure_ratio = closure_mbq / physical_available_mbq
        for index, label in enumerate(imaging["labels"]):
            completed_intervals = int(np.count_nonzero(balance["times_h"] <= times_h[index]))
            future_collections = balance["times_h"][balance["times_h"] > times_h[index]]
            rows.append(
                {
                    "label": str(label),
                    "modality": modality,
                    "elapsed_h": float(times_h[index]),
                    "body_activity_mbq": float(body_activities_mbq[index]),
                    "measured_cumulative_excreted_mbq": float(excreted_mbq[index]),
                    "urine_balance_predicted_body_mbq": float(
                        predicted_body_mbq[index]
                    ),
                    "physical_available_mbq": float(physical_available_mbq[index]),
                    "body_plus_excreted_mbq": float(closure_mbq[index]),
                    "mass_balance_closure_ratio": float(closure_ratio[index]),
                    "completed_urine_intervals": completed_intervals,
                    "next_urine_collection_h": float(future_collections[0])
                    if future_collections.size
                    else math.nan,
                    "imaging_after_last_urine_collection": bool(
                        times_h[index] > last_collection_h
                    ),
                }
            )
    return rows


def calculate_qspect_reference_ratios(
    data: Dict[str, Any],
    balance: Dict[str, np.ndarray],
    imaging: Dict[str, np.ndarray],
) -> List[Dict[str, Any]]:
    """Calculate planar/QSPECT and excretion-balance/QSPECT at QSPECT times.

    Planar activities are propagated from the planar acquisition time to the
    paired Q/SPECT time using physical decay only.  The urine-derived body
    activity is evaluated from collection intervals completed by the Q/SPECT
    time.  It is unavailable before the first completed collection.
    """
    half_life_h = float(data["half_life_h"])
    qspect_times_h = np.asarray(imaging["qspect_times_h"], dtype=float)
    planar_times_h = np.asarray(imaging["planar_times_h"], dtype=float)
    qspect_mbq = np.asarray(imaging["qspect_mbq"], dtype=float)
    planar_mbq = np.asarray(imaging["planar_ctac_mbq"], dtype=float)
    planar_at_qspect_mbq = planar_mbq * decay_factor(
        qspect_times_h - planar_times_h, half_life_h
    )
    balance_at_qspect_mbq = predicted_body_activity(qspect_times_h, data, balance)
    first_collection_h = float(balance["times_h"][0])
    last_collection_h = float(balance["times_h"][-1])

    rows: List[Dict[str, Any]] = []
    for index, label in enumerate(imaging["labels"]):
        balance_available = bool(qspect_times_h[index] >= first_collection_h)
        rows.append(
            {
                "label": str(label),
                "planar_time_h": float(planar_times_h[index]),
                "qspect_time_h": float(qspect_times_h[index]),
                "planar_activity_mbq": float(planar_mbq[index]),
                "planar_at_qspect_time_mbq": float(planar_at_qspect_mbq[index]),
                "qspect_activity_mbq": float(qspect_mbq[index]),
                "mass_balance_at_qspect_time_mbq": float(
                    balance_at_qspect_mbq[index]
                )
                if balance_available
                else math.nan,
                "planar_over_qspect": float(
                    planar_at_qspect_mbq[index] / qspect_mbq[index]
                ),
                "mass_balance_over_qspect": float(
                    balance_at_qspect_mbq[index] / qspect_mbq[index]
                )
                if balance_available
                else math.nan,
                "after_last_urine_collection": bool(
                    qspect_times_h[index] > last_collection_h
                ),
            }
        )
    return rows


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(True, linestyle="--", alpha=0.3)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_blood_clearance(data: Dict[str, Any], output_path: Path) -> Path:
    times = np.asarray([item["elapsed_h"] for item in data["blood"]])
    concentration = np.asarray(
        [item["concentration_kbq_ml"] for item in data["blood"]]
    )
    fig, axis = plt.subplots(figsize=(8.2, 5.4), facecolor="white", layout="constrained")
    axis.plot(times, concentration, color="#8c2d6b", marker="o", linewidth=2.2)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Temps après l'injection (h, échelle logarithmique)")
    axis.set_ylabel("Concentration sanguine (kBq/mL, échelle logarithmique)")
    axis.set_title("Clairance sanguine du Lu-177")
    _style_axis(axis)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_interval_excretion(
    data: Dict[str, Any], balance: Dict[str, np.ndarray], output_path: Path
) -> Path:
    positions = np.arange(balance["times_h"].size)
    labels = [f"{value:.2f} h" for value in balance["times_h"]]
    fig, axis = plt.subplots(figsize=(8.2, 5.4), facecolor="white", layout="constrained")
    axis.bar(
        positions,
        balance["urine_activity_mbq"],
        color="#1f77b4",
        label="Urine",
    )
    axis.bar(
        positions,
        balance["diaper_activity_mbq"],
        bottom=balance["urine_activity_mbq"],
        color="#ff7f0e",
        label="Couches",
    )
    axis.set_xticks(positions, labels, rotation=25, ha="right")
    axis.set_xlabel("Fin de l'intervalle de collecte après l'injection")
    axis.set_ylabel("Activité à la collecte (MBq)")
    axis.set_title("Activité excrétée par intervalle")
    axis.legend(frameon=False)
    _style_axis(axis)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_mass_balance(
    data: Dict[str, Any], balance: Dict[str, np.ndarray], output_path: Path
) -> Path:
    times = np.concatenate(([0.0], balance["times_h"]))
    available = np.concatenate(
        ([data["injected_activity_mbq"]], balance["physical_available_mbq"])
    )
    excreted = np.concatenate(([0.0], balance["cumulative_excreted_at_time_mbq"]))
    body_times = balance["times_h"]
    body = balance["predicted_body_mbq"]
    fig, axis = plt.subplots(figsize=(8.2, 5.4), facecolor="white", layout="constrained")
    axis.plot(
        times,
        available,
        color="black",
        linestyle="--",
        marker="D",
        linewidth=2.0,
        label="Activité disponible avec décroissance physique",
    )
    axis.plot(
        times,
        excreted,
        color="#ff7f0e",
        marker="o",
        linewidth=2.0,
        label="Activité excrétée cumulative, ramenée au temps considéré",
    )
    axis.plot(
        body_times,
        body,
        color="#2ca02c",
        marker="s",
        linewidth=2.2,
        label="Activité corporelle estimée par bilan",
    )
    axis.set_xlabel("Temps après l'injection (h)")
    axis.set_ylabel("Activité (MBq)")
    axis.set_title("Bilan d'activité corrigé pour la décroissance")
    axis.legend(frameon=False, fontsize=8)
    _style_axis(axis)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_mass_balance_vs_imaging(
    data: Dict[str, Any],
    balance: Dict[str, np.ndarray],
    imaging: Dict[str, np.ndarray],
    output_path: Path,
) -> Path:
    maximum_time = max(
        float(imaging["planar_times_h"].max()),
        float(imaging["qspect_times_h"].max()),
    )
    first_collection = float(balance["times_h"][0])
    curve_times = np.linspace(first_collection, maximum_time, 900)
    predicted_curve = predicted_body_activity(curve_times, data, balance)
    last_collection = float(balance["times_h"][-1])

    fig, axis = plt.subplots(figsize=(8.6, 5.6), facecolor="white", layout="constrained")
    axis.axvspan(
        last_collection,
        maximum_time,
        color="#7f7f7f",
        alpha=0.08,
        label="Après 46,62 h : aucune nouvelle collecte urinaire",
    )
    axis.plot(
        curve_times,
        predicted_curve,
        color="#2ca02c",
        linewidth=2.2,
        label="Activité corporelle estimée par le bilan d'excrétion",
    )
    axis.scatter(
        balance["times_h"],
        balance["predicted_body_mbq"],
        color="#2ca02c",
        edgecolor="white",
        linewidth=0.7,
        marker="o",
        s=42,
        zorder=4,
        label="Fin des collectes urine/couches",
    )
    axis.plot(
        imaging["planar_times_h"],
        imaging["planar_ctac_mbq"],
        color="#1f77b4",
        marker="s",
        linewidth=1.8,
        markersize=6,
        label="Planaire CTAC — crop fixe J0",
        zorder=3,
    )
    axis.plot(
        imaging["qspect_times_h"],
        imaging["qspect_mbq"],
        color="black",
        marker="D",
        linewidth=1.8,
        markersize=6,
        label="Q/SPECT",
        zorder=3,
    )
    axis.set_xlabel("Temps après l'injection (h)")
    axis.set_ylabel("Activité corporelle (MBq)")
    axis.set_title("Activité corporelle : bilan d'excrétion et imagerie")
    axis.legend(frameon=False, fontsize=8)
    _style_axis(axis)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_excreted_activity_vs_imaging(
    data: Dict[str, Any],
    balance: Dict[str, np.ndarray],
    imaging: Dict[str, np.ndarray],
    output_path: Path,
) -> Path:
    """Compare measured cumulative excretion with planar and Q/SPECT body activity."""
    maximum_time = max(
        float(imaging["planar_times_h"].max()),
        float(imaging["qspect_times_h"].max()),
    )
    curve_times = np.linspace(0.0, maximum_time, 900)
    excreted_curve = cumulative_excreted_activity(curve_times, data, balance)
    physical_curve = float(data["injected_activity_mbq"]) * decay_factor(
        curve_times, float(data["half_life_h"])
    )
    last_collection = float(balance["times_h"][-1])

    fig, axis = plt.subplots(figsize=(8.6, 5.6), facecolor="white", layout="constrained")
    axis.axvspan(
        last_collection,
        maximum_time,
        color="#7f7f7f",
        alpha=0.08,
        label="Après 46,62 h : excrétion additionnelle non mesurée",
    )
    axis.plot(
        curve_times,
        physical_curve,
        color="#7f7f7f",
        linestyle="--",
        linewidth=1.6,
        label="Activité injectée après décroissance physique",
    )
    axis.plot(
        curve_times,
        excreted_curve,
        color="#ff7f0e",
        linewidth=2.2,
        label="Cumul excrété mesuré — intervalles terminés",
    )
    axis.scatter(
        imaging["planar_times_h"],
        imaging["planar_ctac_mbq"],
        color="#1f77b4",
        marker="s",
        s=48,
        label="Activité corporelle planaire CTAC — crop fixe J0",
        zorder=3,
    )
    axis.scatter(
        imaging["qspect_times_h"],
        imaging["qspect_mbq"],
        color="black",
        marker="D",
        s=45,
        label="Activité corporelle Q/SPECT",
        zorder=3,
    )
    axis.set_xlabel("Temps après l'injection (h)")
    axis.set_ylabel("Activité (MBq)")
    axis.set_title("Activité excrétée comparée à l'activité corporelle")
    axis.legend(frameon=False, fontsize=8)
    _style_axis(axis)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_mass_balance_closure(
    comparison_rows: List[Dict[str, Any]],
    last_collection_h: float,
    output_path: Path,
) -> Path:
    """Plot (body imaging + measured excretion) / decay-corrected injected activity."""
    maximum_time = max(row["elapsed_h"] for row in comparison_rows)
    fig, axis = plt.subplots(figsize=(8.4, 5.4), facecolor="white", layout="constrained")
    axis.axvspan(
        last_collection_h,
        maximum_time,
        color="#7f7f7f",
        alpha=0.08,
        label="Après 46,62 h : excrétion additionnelle non mesurée",
    )
    axis.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label="Fermeture idéale du bilan",
    )
    styles = {
        "planar_ctac_fixed_j0": ("Planaire CTAC — crop fixe J0", "#1f77b4", "s"),
        "qspect": ("Q/SPECT", "black", "D"),
    }
    for modality, (label, color, marker) in styles.items():
        rows = sorted(
            (row for row in comparison_rows if row["modality"] == modality),
            key=lambda row: row["elapsed_h"],
        )
        axis.plot(
            [row["elapsed_h"] for row in rows],
            [row["mass_balance_closure_ratio"] for row in rows],
            color=color,
            marker=marker,
            linewidth=2.0,
            label=label,
        )
    axis.set_xlabel("Temps après l'injection (h)")
    axis.set_ylabel("Fermeture partielle du bilan (sans unité)")
    axis.set_title("Fermeture partielle du bilan d'activité")
    axis.legend(frameon=False, fontsize=8)
    _style_axis(axis)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_qspect_reference_ratios(
    ratio_rows: List[Dict[str, Any]],
    last_collection_h: float,
    output_path: Path,
) -> Path:
    """Plot planar and urine-balance body activities relative to Q/SPECT."""
    qspect_times_h = np.asarray(
        [row["qspect_time_h"] for row in ratio_rows], dtype=float
    )
    planar_ratios = np.asarray(
        [row["planar_over_qspect"] for row in ratio_rows], dtype=float
    )
    balance_ratios = np.asarray(
        [row["mass_balance_over_qspect"] for row in ratio_rows], dtype=float
    )
    labels = [row["label"] for row in ratio_rows]
    maximum_time = float(qspect_times_h.max())

    fig, axis = plt.subplots(figsize=(8.4, 5.4), facecolor="white", layout="constrained")
    axis.axvspan(
        last_collection_h,
        maximum_time,
        color="#7f7f7f",
        alpha=0.08,
        label="Après 46,62 h : excrétion additionnelle non mesurée",
    )
    axis.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label="Accord avec le Q/SPECT",
    )
    axis.plot(
        qspect_times_h,
        planar_ratios,
        color="#1f77b4",
        marker="s",
        linewidth=2.0,
        label="Planaire CTAC / Q/SPECT",
    )
    axis.plot(
        qspect_times_h,
        balance_ratios,
        color="#2ca02c",
        marker="o",
        linewidth=2.0,
        label="Bilan d'excrétion / Q/SPECT",
    )
    axis.set_xticks(qspect_times_h, labels)
    axis.set_xlabel("Temps du Q/SPECT après l'injection")
    axis.set_ylabel("Rapport d'activité relatif au Q/SPECT")
    axis.set_title("Activités planaire et bilan d'excrétion relatives au Q/SPECT")
    axis.legend(frameon=False, fontsize=8)
    _style_axis(axis)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _write_rows(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        cleaned_rows = []
        for row in rows:
            cleaned_rows.append(
                {
                    key: round(float(value), 6)
                    if isinstance(value, (float, np.floating)) and np.isfinite(value)
                    else value
                    for key, value in row.items()
                }
            )
        writer.writerows(cleaned_rows)
    return path


def write_processed_values(
    data: Dict[str, Any], balance: Dict[str, np.ndarray], output_dir: Path
) -> Dict[str, Path]:
    blood_path = output_dir / "blood_processed_values.csv"
    urine_path = output_dir / "urine_mass_balance_values.csv"
    blood_rows = [
        {
            "label": item["label"],
            "elapsed_h": item["elapsed_h"],
            "concentration_kbq_ml": item["concentration_kbq_ml"],
            "sampling_datetime": item["sampling_datetime"].isoformat(sep=" "),
            "counting_datetime": item["counting_datetime"].isoformat(sep=" "),
        }
        for item in data["blood"]
    ]
    urine_rows = []
    for index, item in enumerate(data["urine"]):
        urine_rows.append(
            {
                "label": item["label"],
                "elapsed_h": item["elapsed_h"],
                "urine_volume_ml": item["urine_volume_ml"],
                "urine_activity_mbq": balance["urine_activity_mbq"][index],
                "diaper_activity_mbq": balance["diaper_activity_mbq"][index],
                "interval_excreted_mbq": balance["interval_excreted_mbq"][index],
                "component_sum_difference_mbq": item[
                    "component_sum_difference_mbq"
                ],
                "interval_injection_equivalent_mbq": balance[
                    "interval_injection_equivalent_mbq"
                ][index],
                "cumulative_injection_equivalent_mbq": balance[
                    "cumulative_injection_equivalent_mbq"
                ][index],
                "physical_available_mbq": balance["physical_available_mbq"][index],
                "cumulative_excreted_at_time_mbq": balance[
                    "cumulative_excreted_at_time_mbq"
                ][index],
                "predicted_body_mbq": balance["predicted_body_mbq"][index],
                "closure_error_mbq": balance["closure_error_mbq"][index],
            }
        )
    _write_rows(
        blood_path,
        blood_rows,
        [
            "label",
            "elapsed_h",
            "concentration_kbq_ml",
            "sampling_datetime",
            "counting_datetime",
        ],
    )
    _write_rows(urine_path, urine_rows, list(urine_rows[0]))
    return {"blood_values": blood_path, "urine_values": urine_path}


def write_imaging_excretion_values(
    comparison_rows: List[Dict[str, Any]], output_dir: Path
) -> Path:
    """Save the same-time imaging/excretion comparison as an auditable table."""
    output_path = output_dir / "imaging_excretion_comparison_values.csv"
    return _write_rows(output_path, comparison_rows, list(comparison_rows[0]))


def write_qspect_reference_ratios(
    ratio_rows: List[Dict[str, Any]], output_dir: Path
) -> Path:
    """Save the Q/SPECT-reference ratios shown in the ratio figure."""
    output_path = output_dir / "qspect_reference_ratio_values.csv"
    return _write_rows(output_path, ratio_rows, list(ratio_rows[0]))


def write_report(
    data: Dict[str, Any],
    balance: Dict[str, np.ndarray],
    output_path: Path,
    comparison_rows: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    lines = [
        "Blood and urine exploratory analysis",
        "======================================",
        "",
        f"Source: {data['source_csv']}",
        f"Injected activity: {data['injected_activity_mbq']:.2f} MBq",
        f"Physical half-life used: {data['half_life_h']:.2f} h",
        f"Blood measurements: {len(data['blood'])}",
        f"Urine/diaper intervals: {len(data['urine'])}",
        "",
        "Important interpretation:",
        "  Blood values are concentrations and are not added to excreted activity.",
        "  Urine/diaper interval activities are decay-compensated to injection time",
        "  before the cumulative activity is propagated to each evaluation time.",
        "  After the final urine collection, the predicted body curve assumes no",
        "  additional measured excretion and must be treated as an upper bound.",
        "",
        " time h | urine MBq | diaper MBq | interval MBq | body estimate MBq",
        "------- | --------- | ---------- | ------------ | -----------------",
    ]
    for index, time_h in enumerate(balance["times_h"]):
        lines.append(
            f"{time_h:7.2f} | {balance['urine_activity_mbq'][index]:9.2f} | "
            f"{balance['diaper_activity_mbq'][index]:10.2f} | "
            f"{balance['interval_excreted_mbq'][index]:12.2f} | "
            f"{balance['predicted_body_mbq'][index]:17.2f}"
        )
    lines.extend(
        [
            "",
            f"Maximum numerical closure error: {np.max(np.abs(balance['closure_error_mbq'])):.6g} MBq",
        ]
    )
    if comparison_rows:
        lines.extend(
            [
                "",
                "Imaging plus measured cumulative excretion:",
                "  The cumulative excretion at an imaging time includes completed",
                "  collection intervals only. Excretion during an interval still in",
                "  progress is unavailable, so the closure ratio is a partial balance.",
                "",
                " label | modality               | time h | body MBq | excreted MBq | closure ratio | completed intervals | after last",
                "----- | ---------------------- | ------ | -------- | ------------- | ------------- | ------------------- | ----------",
            ]
        )
        for row in comparison_rows:
            lines.append(
                f"{row['label']:5s} | {row['modality']:22s} | "
                f"{row['elapsed_h']:6.2f} | {row['body_activity_mbq']:8.2f} | "
                f"{row['measured_cumulative_excreted_mbq']:13.2f} | "
                f"{row['mass_balance_closure_ratio']:13.3f} | "
                f"{row['completed_urine_intervals']:19d} | "
                f"{row['imaging_after_last_urine_collection']}"
            )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def sang_urine(
    csv_path: Path = DEFAULT_SOURCE_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    imaging_csv: Path = DEFAULT_IMAGING_CSV,
) -> Dict[str, Any]:
    """Run the blood/urine analysis and generate PNG figures at 300 dpi."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_sang_urine_csv(csv_path)
    balance = calculate_urine_mass_balance(data)
    imaging = load_imaging_comparison(data["injection_datetime"], imaging_csv)
    comparison_rows = (
        calculate_imaging_excretion_comparison(data, balance, imaging)
        if imaging is not None
        else None
    )
    ratio_rows = (
        calculate_qspect_reference_ratios(data, balance, imaging)
        if imaging is not None
        else None
    )

    paths: Dict[str, Path] = {
        "blood_figure": plot_blood_clearance(
            data, output_dir / "blood_concentration_vs_time.png"
        ),
        "interval_figure": plot_interval_excretion(
            data, balance, output_dir / "urine_diaper_activity_by_interval.png"
        ),
        "mass_balance_figure": plot_mass_balance(
            data, balance, output_dir / "decay_corrected_mass_balance.png"
        ),
    }
    if imaging is not None:
        paths["imaging_comparison_figure"] = plot_mass_balance_vs_imaging(
            data,
            balance,
            imaging,
            output_dir / "mass_balance_vs_planar_qspect.png",
        )
        paths["excreted_vs_imaging_figure"] = plot_excreted_activity_vs_imaging(
            data,
            balance,
            imaging,
            output_dir / "cumulative_excreted_vs_planar_qspect.png",
        )
        paths["closure_figure"] = plot_mass_balance_closure(
            comparison_rows,
            float(balance["times_h"][-1]),
            output_dir / "mass_balance_closure_planar_qspect.png",
        )
        paths["qspect_reference_ratio_figure"] = plot_qspect_reference_ratios(
            ratio_rows,
            float(balance["times_h"][-1]),
            output_dir / "qspect_reference_activity_ratios.png",
        )
        paths["imaging_excretion_values"] = write_imaging_excretion_values(
            comparison_rows, output_dir
        )
        paths["qspect_reference_ratio_values"] = write_qspect_reference_ratios(
            ratio_rows, output_dir
        )
    paths["report"] = write_report(
        data,
        balance,
        output_dir / "sang_urine_report.txt",
        comparison_rows=comparison_rows,
    )
    paths.update(write_processed_values(data, balance, output_dir))
    return {
        "data": data,
        "balance": balance,
        "imaging": imaging,
        "comparison_rows": comparison_rows,
        "ratio_rows": ratio_rows,
        "paths": paths,
    }


def main() -> None:
    result = sang_urine()
    balance = result["balance"]
    print(
        f"Parsed {len(result['data']['blood'])} blood measurements and "
        f"{len(result['data']['urine'])} urine/diaper intervals."
    )
    print(
        f"Predicted body activity at {balance['times_h'][-1]:.2f} h: "
        f"{balance['predicted_body_mbq'][-1]:.2f} MBq"
    )
    for label, path in result["paths"].items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
