from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

import planar_processing
import qspect_processing


COMPARISON_MODE = "time_normalized"
LU177_PHYSICAL_HALF_LIFE_DAYS = 6.65


def default_planar_dir() -> Path:
    return planar_processing.default_planar_study_dir()


def default_qspect_dir() -> Path:
    return qspect_processing.default_qspect_dir()


def normalize(values: List[float]) -> List[float]:
    if not values or values[0] == 0:
        return [np.nan for _value in values]
    return [value / values[0] for value in values]


def fit_exponential_half_life(day_offsets: List[float], values: List[float]) -> Dict[str, float]:
    """Fit y = A0 * exp(-lambda * t) and return effective half-life in days."""
    x = np.asarray(day_offsets, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y) & (y > 0)
    if valid.sum() < 2:
        raise ValueError("At least two positive points are required for an exponential fit")

    slope, intercept = np.polyfit(x[valid], np.log(y[valid]), 1)
    decay_constant = -float(slope)
    if decay_constant <= 0:
        raise ValueError("Fit did not produce a positive decay constant")

    half_life_days = float(np.log(2) / decay_constant)
    physical_decay_constant = float(np.log(2) / LU177_PHYSICAL_HALF_LIFE_DAYS)
    biological_decay_constant = decay_constant - physical_decay_constant
    biological_half_life_days = (
        float(np.log(2) / biological_decay_constant)
        if biological_decay_constant > 0
        else np.inf
    )
    initial_value = float(np.exp(intercept))
    fitted = initial_value * np.exp(-decay_constant * x)
    residuals = y[valid] - (initial_value * np.exp(-decay_constant * x[valid]))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y[valid] - np.mean(y[valid])) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "initial_value": initial_value,
        "decay_constant": decay_constant,
        "half_life_days": half_life_days,
        "physical_half_life_days": LU177_PHYSICAL_HALF_LIFE_DAYS,
        "physical_decay_constant": physical_decay_constant,
        "biological_decay_constant": biological_decay_constant,
        "biological_half_life_days": biological_half_life_days,
        "r_squared": r_squared,
        "fitted_values": fitted,
    }


def describe_half_life_fit(name: str, fit: Dict[str, float]) -> str:
    bio_half_life = fit["biological_half_life_days"]
    bio_text = "inf" if not np.isfinite(bio_half_life) else f"{bio_half_life:.2f}"
    return (
        f"  {name}: T_eff={fit['half_life_days']:.2f} days, "
        f"T_phys={fit['physical_half_life_days']:.2f} days, "
        f"T_bio={bio_text} days, "
        f"lambda_eff={fit['decay_constant']:.4f} 1/day, "
        f"lambda_bio={fit['biological_decay_constant']:.4f} 1/day, "
        f"R2={fit['r_squared']:.3f}"
    )


def nearest_qspect_point(day_offset: float, qspect_rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not qspect_rows:
        return None
    return min(qspect_rows, key=lambda row: abs(row["day_offset"] - day_offset))


def paired_decay_data(
    planar_rows: List[Dict[str, Any]],
    qspect_rows: List[Dict[str, Any]],
    max_day_difference: float = 0.6,
) -> List[Dict[str, Any]]:
    pairs = []
    for planar_row in planar_rows:
        qspect_row = nearest_qspect_point(planar_row["day_offset"], qspect_rows)
        if qspect_row is None:
            continue
        day_difference = abs(qspect_row["day_offset"] - planar_row["day_offset"])
        if day_difference > max_day_difference:
            continue
        pairs.append(
            {
                "day_offset": planar_row["day_offset"],
                "planar_label": planar_row["label"],
                "qspect_label": qspect_row["label"],
                "planar_tew_counts": planar_row["tew_corrected_counts"],
                "planar_tew_cps": planar_row["tew_corrected_cps"],
                "planar_activity_mbq": planar_row["planar_activity_mbq"],
                "duration_seconds": planar_row["duration_seconds"],
                "qspect_activity_mbq": qspect_row["activity_mbq"],
                "day_difference": day_difference,
            }
        )
    return pairs


def print_pairs(pairs: List[Dict[str, Any]]) -> None:
    print("Planar TEW vs Q/SPECT paired decay data:")
    for pair in pairs:
        print(
            f"  day={pair['day_offset']:.2f} | "
            f"planar={pair['planar_tew_counts']:.1f} counts | "
            f"planar_rate={pair['planar_tew_cps']:.1f} cps | "
            f"planar_activity={pair['planar_activity_mbq']:.1f} MBq | "
            f"qspect={pair['qspect_activity_mbq']:.1f} MBq | "
            f"duration={pair['duration_seconds']:.1f}s | "
            f"qspect_label={pair['qspect_label']}"
        )


def plot_normalized_decay_comparison(pairs: List[Dict[str, Any]]) -> None:
    if not pairs:
        raise ValueError("No paired planar/QSPECT data available")

    x = [pair["day_offset"] for pair in pairs]
    planar_values = [pair["planar_tew_counts"] for pair in pairs]
    qspect_values = [pair["qspect_activity_mbq"] for pair in pairs]
    planar_norm = normalize(planar_values)
    qspect_norm = normalize(qspect_values)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, planar_norm, marker="o", label="Planar TEW corrected counts")
    ax.plot(x, qspect_norm, marker="s", label="Q/SPECT activity")
    ax.set_xlabel("Jour depuis la première acquisition")
    ax.set_ylabel("Valeur normalisée au jour 0")
    ax.set_title("Décroissance normalisée: planar corrigé vs Q/SPECT")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_activity_comparison(pairs: List[Dict[str, Any]]) -> None:
    if not pairs:
        raise ValueError("No paired planar/QSPECT data available")

    x = [pair["day_offset"] for pair in pairs]
    planar_values = [pair["planar_activity_mbq"] for pair in pairs]
    qspect_values = [pair["qspect_activity_mbq"] for pair in pairs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, planar_values, marker="o", label="Planar TEW activity estimate")
    ax.plot(x, qspect_values, marker="s", label="Q/SPECT activity")
    ax.set_xlabel("Jour depuis la première acquisition")
    ax.set_ylabel("Activité (MBq)")
    ax.set_title(f"Activité planar vs Q/SPECT (sensibilité={planar_processing.CAMERA_SENSITIVITY_CPS_PER_MBQ:g} cps/MBq)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_time_normalized_decay_comparison(pairs: List[Dict[str, Any]]) -> None:
    if not pairs:
        raise ValueError("No paired planar/QSPECT data available")

    x = [pair["day_offset"] for pair in pairs]
    planar_rates = [pair["planar_tew_cps"] for pair in pairs]
    qspect_values = [pair["qspect_activity_mbq"] for pair in pairs]
    planar_norm = normalize(planar_rates)
    qspect_norm = normalize(qspect_values)
    planar_fit = fit_exponential_half_life(x, planar_norm)
    qspect_fit = fit_exponential_half_life(x, qspect_norm)
    fit_x = np.linspace(min(x), max(x), 200)
    planar_fit_y = planar_fit["initial_value"] * np.exp(-planar_fit["decay_constant"] * fit_x)
    qspect_fit_y = qspect_fit["initial_value"] * np.exp(-qspect_fit["decay_constant"] * fit_x)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, planar_norm, marker="o", label="Planar TEW count rate (cps)")
    ax.plot(x, qspect_norm, marker="s", label="Q/SPECT activity")
    ax.plot(fit_x, planar_fit_y, linestyle="--", color="tab:blue", alpha=0.8, label=f"Fit planar T1/2={planar_fit['half_life_days']:.2f} j")
    ax.plot(fit_x, qspect_fit_y, linestyle="--", color="tab:orange", alpha=0.8, label=f"Fit Q/SPECT T1/2={qspect_fit['half_life_days']:.2f} j")
    ax.set_xlabel("Jour depuis la première acquisition")
    ax.set_ylabel("Valeur normalisée au jour 0")
    ax.set_title("Décroissance normalisée après correction du temps d'acquisition")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    plt.show()


def print_half_life_fits(pairs: List[Dict[str, Any]]) -> None:
    x = [pair["day_offset"] for pair in pairs]
    planar_rates = [pair["planar_tew_cps"] for pair in pairs]
    qspect_values = [pair["qspect_activity_mbq"] for pair in pairs]
    planar_fit = fit_exponential_half_life(x, planar_rates)
    qspect_fit = fit_exponential_half_life(x, qspect_values)
    print("Effective + biological half-life exponential fits:")
    print(describe_half_life_fit("Planar TEW count rate", planar_fit))
    print(describe_half_life_fit("Q/SPECT activity", qspect_fit))


def run_comparison(planar_dir: Path = default_planar_dir(), qspect_dir: Path = default_qspect_dir()) -> None:
    planar_rows = planar_processing.planar_tew_decay_data(planar_dir)
    qspect_rows = qspect_processing.qspect_decay_data(qspect_dir)
    pairs = paired_decay_data(planar_rows, qspect_rows)
    print_pairs(pairs)
    print_half_life_fits(pairs)
    if COMPARISON_MODE == "activity":
        plot_activity_comparison(pairs)
    elif COMPARISON_MODE == "normalized":
        plot_normalized_decay_comparison(pairs)
    elif COMPARISON_MODE == "time_normalized":
        plot_time_normalized_decay_comparison(pairs)
    else:
        raise ValueError(f"Unknown COMPARISON_MODE: {COMPARISON_MODE}")


if __name__ == "__main__":
    run_comparison()
