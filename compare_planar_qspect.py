from pathlib import Path
from datetime import datetime
import re
import time
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, nnls

import planar_processing
import qspect_processing
import dicom_loader


COMPARISON_MODE = "time_normalized"
LU177_PHYSICAL_HALF_LIFE_DAYS = 6.65
MIN_DECAY_CONSTANT = 1e-8
MAX_DECAY_CONSTANT = 100.0
AUTO_CLOSE_FIGURES = True
AUTO_CLOSE_DELAY_SECONDS = 0.1
SAVE_FIT_FIGURE = True
RUN_PATIENT_4V6_FIGURE = False
FIG_ROOT = Path(__file__).resolve().parent / "fig"
RETENTION_FIG_DIR = FIG_ROOT / "retention"
FIT_FIGURE_PATH = RETENTION_FIG_DIR / "planar_qspect_retention_fits.png"
ACTIVITY_FIT_FIGURE_PATH = RETENTION_FIG_DIR / "planar_qspect_activity_fits.png"
SELECTED_BI_FIT_FIGURE_PATH = RETENTION_FIG_DIR / "planar_qspect_activity_fits_selected_bi_only.png"
SELECTED_RATIO_FIGURE_PATH = RETENTION_FIG_DIR / "planar_qspect_selected_activity_ratios.png"
ALL_RATIO_FIGURE_PATH = RETENTION_FIG_DIR / "planar_qspect_all_activity_ratios.png"
POWERPOINT_FIT_FIGURE_PATH = RETENTION_FIG_DIR / "planar_qspect_activity_bi_fit_powerpoint.png"
TOTAL_ACTIVITY_FIT_FIGURE_PATH = RETENTION_FIG_DIR / "planar_total_windows_qspect_activity_fits.png"
ACTIVITY_FIT_VALUES_PATH = RETENTION_FIG_DIR / "planar_qspect_activity_fits_values.txt"
PLANAR_COUNTS_POWERPOINT_PATH = RETENTION_FIG_DIR / "planar_counts_bi_fit_powerpoint.png"
PATIENT_4V6_COUNTS_POWERPOINT_PATH = RETENTION_FIG_DIR / "patient_4v6_planar_counts_bi_fit_powerpoint.png"
PATIENT_4V6_ROOT = Path(__file__).resolve().parent / "Data" / "1j92g763g"


def default_planar_dir() -> Path:
    return planar_processing.default_planar_study_dir()


def default_qspect_dir() -> Path:
    return qspect_processing.default_qspect_dir()


def normalize(values: List[float]) -> List[float]:
    if not values or values[0] == 0:
        return [np.nan for _value in values]
    return [value / values[0] for value in values]


def show_figure(keep_open: bool = False) -> None:
    if keep_open:
        plt.show()
    elif AUTO_CLOSE_FIGURES:
        plt.show(block=False)
        for figure_number in plt.get_fignums():
            plt.figure(figure_number).canvas.flush_events()
        time.sleep(AUTO_CLOSE_DELAY_SECONDS)
        plt.close("all")
    else:
        plt.show()


def monoexponential(t_days: np.ndarray, a0: float, decay_constant: float) -> np.ndarray:
    return a0 * np.exp(-decay_constant * t_days)


def biexponential(
    t_days: np.ndarray,
    a1: float,
    decay_constant_1: float,
    a2: float,
    decay_constant_2: float,
) -> np.ndarray:
    return a1 * np.exp(-decay_constant_1 * t_days) + a2 * np.exp(-decay_constant_2 * t_days)


def clean_fit_data(day_offsets: List[float], values: List[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(day_offsets, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y > 0)
    if valid.sum() < 2:
        raise ValueError("At least two positive points are required for an exponential fit")
    x = x[valid]
    y = y[valid]
    order = np.argsort(x)
    return x[order], y[order]


def fit_metrics(observed: np.ndarray, fitted: np.ndarray, n_parameters: int) -> Dict[str, float]:
    residuals = observed - fitted
    rss = float(np.sum(residuals**2))
    n = int(observed.size)
    tss = float(np.sum((observed - np.mean(observed)) ** 2))
    mse = rss / n
    safe_mse = max(mse, np.finfo(float).tiny)
    return {
        "r_squared": float(1.0 - rss / tss) if tss > 0 else np.nan,
        "rmse": float(np.sqrt(mse)),
        "aic": float(n * np.log(safe_mse) + 2 * n_parameters),
        "bic": float(n * np.log(safe_mse) + n_parameters * np.log(n)),
    }


def mono_initial_guess(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(x, np.log(y), 1)
    decay_constant = -float(slope)
    initial_value = max(float(np.exp(intercept)), float(y[0]))
    return initial_value, min(max(decay_constant, MIN_DECAY_CONSTANT), MAX_DECAY_CONSTANT)


def fit_monoexponential(day_offsets: List[float], values: List[float]) -> Dict[str, Any]:
    """Fit A(t) = A0 * exp(-lambda_eff * t), with t in days."""
    x, y = clean_fit_data(day_offsets, values)
    p0 = mono_initial_guess(x, y)
    upper_value = max(float(y.max()) * 20.0, float(y[0]) * 20.0)
    popt, _pcov = curve_fit(
        monoexponential,
        x,
        y,
        p0=p0,
        bounds=([0.0, MIN_DECAY_CONSTANT], [upper_value, MAX_DECAY_CONSTANT]),
        maxfev=20000,
    )
    initial_value, decay_constant = map(float, popt)
    fitted = monoexponential(x, initial_value, decay_constant)

    half_life_days = float(np.log(2) / decay_constant)
    physical_decay_constant = float(np.log(2) / LU177_PHYSICAL_HALF_LIFE_DAYS)
    biological_decay_constant = decay_constant - physical_decay_constant
    biological_half_life_days = (
        float(np.log(2) / biological_decay_constant)
        if biological_decay_constant > 0
        else np.inf
    )
    metrics = fit_metrics(y, fitted, n_parameters=2)

    return {
        "model": "monoexponential",
        "x_days": x,
        "observed_values": y,
        "initial_value": initial_value,
        "decay_constant": decay_constant,
        "half_life_days": half_life_days,
        "physical_half_life_days": LU177_PHYSICAL_HALF_LIFE_DAYS,
        "physical_decay_constant": physical_decay_constant,
        "biological_decay_constant": biological_decay_constant,
        "biological_half_life_days": biological_half_life_days,
        "auc_0_inf": float(initial_value / decay_constant),
        "fitted_values": fitted,
        **metrics,
    }


def fit_exponential_half_life(day_offsets: List[float], values: List[float]) -> Dict[str, Any]:
    """Backward-compatible alias for the monoexponential fit."""
    return fit_monoexponential(day_offsets, values)


def fit_biexponential(day_offsets: List[float], values: List[float]) -> Dict[str, Any]:
    """Fit A(t) = A1 * exp(-lambda1 * t) + A2 * exp(-lambda2 * t), with t in days."""
    x, y = clean_fit_data(day_offsets, values)
    if len(y) < 4:
        raise ValueError("At least four positive points are required for a biexponential fit")

    lambda_grid = np.geomspace(0.03, 8.0, 70)
    best_rss = np.inf
    best_parameters = None

    for lambda_slow in lambda_grid:
        for lambda_fast in lambda_grid:
            if lambda_fast <= lambda_slow * 1.15:
                continue
            design = np.column_stack(
                [
                    np.exp(-lambda_slow * x),
                    np.exp(-lambda_fast * x),
                ]
            )
            amplitudes, _residual_norm = nnls(design, y)
            fitted_candidate = design @ amplitudes
            rss = float(np.sum((y - fitted_candidate) ** 2))
            if rss < best_rss:
                best_rss = rss
                best_parameters = (
                    float(amplitudes[0]),
                    float(lambda_slow),
                    float(amplitudes[1]),
                    float(lambda_fast),
                    fitted_candidate,
                )

    if best_parameters is None:
        raise RuntimeError("Biexponential grid fit failed")

    a_slow, lambda_slow, a_fast, lambda_fast, fitted = best_parameters
    a1, lambda1 = a_slow, lambda_slow
    a2, lambda2 = a_fast, lambda_fast

    metrics = fit_metrics(y, fitted, n_parameters=4)
    return {
        "model": "biexponential",
        "x_days": x,
        "observed_values": y,
        "a1": a1,
        "decay_constant_1": lambda1,
        "a2": a2,
        "decay_constant_2": lambda2,
        "a_fast": a_fast,
        "lambda_fast": lambda_fast,
        "a_slow": a_slow,
        "lambda_slow": lambda_slow,
        "half_life_fast_days": float(np.log(2) / lambda_fast),
        "half_life_slow_days": float(np.log(2) / lambda_slow),
        "auc_0_inf": float(a1 / lambda1 + a2 / lambda2),
        "fitted_values": fitted,
        **metrics,
    }


def describe_mono_fit(name: str, fit: Dict[str, Any]) -> str:
    bio_half_life = fit["biological_half_life_days"]
    bio_text = "inf" if not np.isfinite(bio_half_life) else f"{bio_half_life:.2f}"
    return (
        f"  {name} mono: A0={fit['initial_value']:.3g}, "
        f"lambda_eff={fit['decay_constant']:.4f} 1/day, "
        f"T_eff={fit['half_life_days']:.2f} days, "
        f"T_phys={fit['physical_half_life_days']:.2f} days, "
        f"T_bio={bio_text} days, "
        f"AUC={fit['auc_0_inf']:.3g}, "
        f"R2={fit['r_squared']:.3f}, RMSE={fit['rmse']:.3g}, "
        f"AIC={fit['aic']:.2f}, BIC={fit['bic']:.2f}"
    )


def describe_bi_fit(name: str, fit: Dict[str, Any]) -> str:
    return (
        f"  {name} bi: "
        f"A1={fit['a1']:.3g}, lambda1={fit['decay_constant_1']:.4f} 1/day, "
        f"A2={fit['a2']:.3g}, lambda2={fit['decay_constant_2']:.4f} 1/day, "
        f"T_fast={fit['half_life_fast_days']:.2f} days, "
        f"T_slow={fit['half_life_slow_days']:.2f} days, "
        f"AUC={fit['auc_0_inf']:.3g}, "
        f"R2={fit['r_squared']:.3f}, RMSE={fit['rmse']:.3g}, "
        f"AIC={fit['aic']:.2f}, BIC={fit['bic']:.2f}"
    )


def compact_fit_label(label: str, mono_fit: Dict[str, Any], bi_fit: Dict[str, Any]) -> str:
    return (
        f"{label}\n"
        f"mono: T1/2={mono_fit['half_life_days']:.2f} j, "
        f"AIC={mono_fit['aic']:.1f}, BIC={mono_fit['bic']:.1f}\n"
        f"bi: Tfast={bi_fit['half_life_fast_days']:.2f} j, "
        f"Tslow={bi_fit['half_life_slow_days']:.2f} j, "
        f"AIC={bi_fit['aic']:.1f}, BIC={bi_fit['bic']:.1f}"
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
                "planar_photopeak_raw_counts": planar_row.get("photopeak_counts_dead_time"),
                "planar_photopeak_raw_cps": planar_row.get("photopeak_cps_dead_time"),
                "planar_photopeak_raw_activity_mbq": planar_row.get("planar_photopeak_raw_activity_mbq"),
                "planar_tew_counts": planar_row["tew_corrected_counts"],
                "planar_tew_cps": planar_row["tew_corrected_cps"],
                "planar_activity_mbq": planar_row["planar_activity_mbq"],
                "planar_tew_scan_velocity_activity_mbq": planar_row.get("planar_tew_scan_velocity_activity_mbq"),
                "planar_tew_local_dwell_activity_mbq": planar_row.get("planar_tew_local_dwell_activity_mbq"),
                "planar_total_window_counts": planar_row.get("total_window_counts_dead_time"),
                "planar_total_window_cps": planar_row.get("total_window_cps_dead_time"),
                "planar_total_window_activity_mbq": planar_row.get("planar_total_window_activity_mbq"),
                "duration_seconds": planar_row["duration_seconds"],
                "scan_velocity_mm_per_s": planar_row.get("scan_velocity_mm_per_s"),
                "scan_length_mm": planar_row.get("scan_length_mm"),
                "detector_longitudinal_fov_mm": planar_row.get("detector_longitudinal_fov_mm"),
                "scan_time_from_velocity_seconds": planar_row.get("scan_time_from_velocity_seconds"),
                "local_dwell_time_seconds": planar_row.get("local_dwell_time_seconds"),
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
            f"photopeak_raw_activity={pair['planar_photopeak_raw_activity_mbq']:.1f} MBq | "
            f"planar_activity={pair['planar_activity_mbq']:.1f} MBq | "
            f"planar_scan_velocity_activity={pair['planar_tew_scan_velocity_activity_mbq']:.1f} MBq | "
            f"planar_local_dwell_activity={pair['planar_tew_local_dwell_activity_mbq']:.1f} MBq | "
            f"planar_total_activity={pair['planar_total_window_activity_mbq']:.1f} MBq | "
            f"qspect={pair['qspect_activity_mbq']:.1f} MBq | "
            f"duration={pair['duration_seconds']:.1f}s | "
            f"qspect_label={pair['qspect_label']}"
        )


def format_fixed_width_table(headers: List[str], rows: List[List[str]]) -> List[str]:
    widths = [
        max(len(header), *(len(row[column]) for row in rows))
        for column, header in enumerate(headers)
    ]

    def format_row(values: List[str]) -> str:
        return " | ".join(value.rjust(widths[column]) for column, value in enumerate(values))

    return [format_row(headers), format_row(["-" * width for width in widths])] + [
        format_row(row) for row in rows
    ]


def write_activity_fit_values_report(pairs: List[Dict[str, Any]]) -> None:
    if not pairs:
        raise ValueError("No paired planar/QSPECT data available")

    ACTIVITY_FIT_VALUES_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Values Used in planar_qspect_activity_fits",
        "==========================================",
        "",
        "Short names:",
        "  raw = geometric-mean photopeak, dead-time corrected, no TEW subtraction",
        "  frameTEW = TEW counts divided by DICOM ActualFrameDuration",
        "  scanTEW = TEW counts divided by ScanLength / ScanVelocity",
        "  localTEW = TEW counts divided by detector longitudinal FOV / ScanVelocity",
        "  tot = summed available windows, approx. 55-250 keV, dead-time corrected",
        "  QSP = Q/SPECT activity",
        "  localTEW assumes 387 mm is the detector FOV along the scan direction.",
        "",
        "Activity and ratios",
        "-------------------",
    ]

    activity_headers = [
        "day",
        "raw MBq",
        "frameTEW MBq",
        "scanTEW MBq",
        "localTEW MBq",
        "tot MBq",
        "QSP MBq",
        "raw/QSP",
        "frame/QSP",
        "scan/QSP",
        "local/QSP",
        "tot/QSP",
    ]
    activity_rows = []
    for pair in pairs:
        qspect = float(pair["qspect_activity_mbq"])
        raw = float(pair["planar_photopeak_raw_activity_mbq"])
        tew = float(pair["planar_activity_mbq"])
        velocity_tew = float(pair["planar_tew_scan_velocity_activity_mbq"])
        local_tew = float(pair["planar_tew_local_dwell_activity_mbq"])
        total = float(pair["planar_total_window_activity_mbq"])
        activity_rows.append(
            [
                f"{pair['day_offset']:.2f}",
                f"{raw:.1f}",
                f"{tew:.1f}",
                f"{velocity_tew:.1f}",
                f"{local_tew:.1f}",
                f"{total:.1f}",
                f"{qspect:.1f}",
                f"{raw / qspect:.4f}",
                f"{tew / qspect:.4f}",
                f"{velocity_tew / qspect:.4f}",
                f"{local_tew / qspect:.4f}",
                f"{total / qspect:.4f}",
            ]
        )
    lines.extend(format_fixed_width_table(activity_headers, activity_rows))

    lines.extend(
        [
            "",
            "Planar counts used",
            "------------------",
        ]
    )
    counts_headers = ["day", "raw ct", "TEW ct", "tot ct", "frame s", "scan s", "local s"]
    counts_rows = []
    for pair in pairs:
        counts_rows.append(
            [
                f"{pair['day_offset']:.2f}",
                f"{pair['planar_photopeak_raw_counts']:.1f}",
                f"{pair['planar_tew_counts']:.1f}",
                f"{pair['planar_total_window_counts']:.1f}",
                f"{pair['duration_seconds']:.1f}",
                f"{pair['scan_time_from_velocity_seconds']:.1f}",
                f"{pair['local_dwell_time_seconds']:.1f}",
            ]
        )
    lines.extend(format_fixed_width_table(counts_headers, counts_rows))

    first_pair = pairs[0]
    lines.extend(
        [
            "",
            "Timing/config used:",
            f"  sensitivity_cps_per_mbq = {planar_processing.CAMERA_SENSITIVITY_CPS_PER_MBQ:g}",
            f"  scan_velocity_mm_per_s = {first_pair['scan_velocity_mm_per_s']:.1f}",
            f"  scan_length_mm = {first_pair['scan_length_mm']:.1f}",
            f"  detector_longitudinal_fov_mm = {first_pair['detector_longitudinal_fov_mm']:.1f}",
            f"  t_scan = scan_length / scan_velocity = {first_pair['scan_time_from_velocity_seconds']:.1f} s",
            f"  t_local = detector_longitudinal_fov / scan_velocity = {first_pair['local_dwell_time_seconds']:.1f} s",
            "  sensitivity convention uncertainty: local files do not say whether 9.36 cps/MBq is per detector,",
            "  geometric mean, or summed detectors.",
        ]
    )

    lines.extend(
        [
            "",
            "Mean ratios:",
            f"  raw/QSP = {np.mean([pair['planar_photopeak_raw_activity_mbq'] / pair['qspect_activity_mbq'] for pair in pairs]):.4f}",
            f"  frameTEW/QSP = {np.mean([pair['planar_activity_mbq'] / pair['qspect_activity_mbq'] for pair in pairs]):.4f}",
            f"  scanTEW/QSP = {np.mean([pair['planar_tew_scan_velocity_activity_mbq'] / pair['qspect_activity_mbq'] for pair in pairs]):.4f}",
            f"  localTEW/QSP = {np.mean([pair['planar_tew_local_dwell_activity_mbq'] / pair['qspect_activity_mbq'] for pair in pairs]):.4f}",
            f"  tot/QSP = {np.mean([pair['planar_total_window_activity_mbq'] / pair['qspect_activity_mbq'] for pair in pairs]):.4f}",
            "",
        ]
    )
    ACTIVITY_FIT_VALUES_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved activity fit values report: {ACTIVITY_FIT_VALUES_PATH}")


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
    show_figure()


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
    show_figure()


def plot_time_normalized_decay_comparison(pairs: List[Dict[str, Any]]) -> None:
    if not pairs:
        raise ValueError("No paired planar/QSPECT data available")

    x = [pair["day_offset"] for pair in pairs]
    planar_rates = [pair["planar_tew_cps"] for pair in pairs]
    qspect_values = [pair["qspect_activity_mbq"] for pair in pairs]
    planar_norm = normalize(planar_rates)
    qspect_norm = normalize(qspect_values)
    planar_fit = fit_monoexponential(x, planar_norm)
    qspect_fit = fit_monoexponential(x, qspect_norm)
    planar_bi_fit = fit_biexponential(x, planar_norm)
    qspect_bi_fit = fit_biexponential(x, qspect_norm)
    fit_x = np.linspace(min(x), max(x), 200)
    planar_fit_y = monoexponential(fit_x, planar_fit["initial_value"], planar_fit["decay_constant"])
    qspect_fit_y = monoexponential(fit_x, qspect_fit["initial_value"], qspect_fit["decay_constant"])
    planar_bi_fit_y = biexponential(
        fit_x,
        planar_bi_fit["a1"],
        planar_bi_fit["decay_constant_1"],
        planar_bi_fit["a2"],
        planar_bi_fit["decay_constant_2"],
    )
    qspect_bi_fit_y = biexponential(
        fit_x,
        qspect_bi_fit["a1"],
        qspect_bi_fit["decay_constant_1"],
        qspect_bi_fit["a2"],
        qspect_bi_fit["decay_constant_2"],
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, planar_norm, marker="o", label="Planar TEW count rate (cps)")
    ax.plot(x, qspect_norm, marker="s", label="Q/SPECT activity")
    ax.plot(fit_x, planar_fit_y, linestyle="--", color="tab:blue", alpha=0.8, label=f"Planar mono T1/2={planar_fit['half_life_days']:.2f} j")
    ax.plot(fit_x, planar_bi_fit_y, linestyle=":", color="tab:blue", alpha=0.9, label=f"Planar bi Tslow={planar_bi_fit['half_life_slow_days']:.2f} j")
    ax.plot(fit_x, qspect_fit_y, linestyle="--", color="tab:orange", alpha=0.8, label=f"Q/SPECT mono T1/2={qspect_fit['half_life_days']:.2f} j")
    ax.plot(fit_x, qspect_bi_fit_y, linestyle=":", color="tab:orange", alpha=0.9, label=f"Q/SPECT bi Tslow={qspect_bi_fit['half_life_slow_days']:.2f} j")
    ax.set_xlabel("Jour depuis la première acquisition")
    ax.set_ylabel("Valeur normalisée au jour 0")
    ax.set_title("Décroissance normalisée après correction du temps d'acquisition")
    ax.grid(True, linestyle="--", alpha=0.4)
    fit_text = (
        compact_fit_label("Planar", planar_fit, planar_bi_fit)
        + "\n\n"
        + compact_fit_label("Q/SPECT", qspect_fit, qspect_bi_fit)
    )
    ax.text(
        0.98,
        0.98,
        fit_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )
    ax.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    if SAVE_FIT_FIGURE:
        FIT_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIT_FIGURE_PATH, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved fit figure: {FIT_FIGURE_PATH}")
    show_figure(keep_open=True)


def plot_activity_fit_comparison(pairs: List[Dict[str, Any]]) -> None:
    if not pairs:
        raise ValueError("No paired planar/QSPECT data available")

    x = [pair["day_offset"] for pair in pairs]
    photopeak_raw_values = [pair["planar_photopeak_raw_activity_mbq"] for pair in pairs]
    planar_values = [pair["planar_activity_mbq"] for pair in pairs]
    planar_velocity_values = [pair["planar_tew_scan_velocity_activity_mbq"] for pair in pairs]
    planar_local_values = [pair["planar_tew_local_dwell_activity_mbq"] for pair in pairs]
    planar_total_values = [pair["planar_total_window_activity_mbq"] for pair in pairs]
    qspect_values = [pair["qspect_activity_mbq"] for pair in pairs]
    photopeak_raw_fit = fit_monoexponential(x, photopeak_raw_values)
    planar_fit = fit_monoexponential(x, planar_values)
    planar_velocity_fit = fit_monoexponential(x, planar_velocity_values)
    planar_local_fit = fit_monoexponential(x, planar_local_values)
    planar_total_fit = fit_monoexponential(x, planar_total_values)
    qspect_fit = fit_monoexponential(x, qspect_values)
    photopeak_raw_bi_fit = fit_biexponential(x, photopeak_raw_values)
    planar_bi_fit = fit_biexponential(x, planar_values)
    planar_velocity_bi_fit = fit_biexponential(x, planar_velocity_values)
    planar_local_bi_fit = fit_biexponential(x, planar_local_values)
    planar_total_bi_fit = fit_biexponential(x, planar_total_values)
    qspect_bi_fit = fit_biexponential(x, qspect_values)
    fit_x = np.linspace(min(x), max(x), 200)

    raw_color = "tab:purple"
    tew_color = "tab:blue"
    velocity_color = "tab:cyan"
    local_color = "tab:red"
    total_color = "tab:green"
    qspect_color = "tab:orange"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, photopeak_raw_values, marker="d", color=raw_color, label="Planar raw photopeak activity estimate")
    ax.plot(x, planar_values, marker="o", color=tew_color, label="Planar TEW frame-duration estimate")
    ax.plot(x, planar_velocity_values, marker="v", color=velocity_color, label="Planar TEW scan-time estimate")
    ax.plot(x, planar_local_values, marker="P", color=local_color, label="Planar TEW local-dwell estimate (static CF)")
    ax.plot(x, planar_total_values, marker="^", color=total_color, label="Planar total-window activity estimate")
    ax.plot(x, qspect_values, marker="s", color=qspect_color, label="Q/SPECT activity")
    ax.plot(
        fit_x,
        monoexponential(fit_x, photopeak_raw_fit["initial_value"], photopeak_raw_fit["decay_constant"]),
        linestyle="--",
        color=raw_color,
        alpha=0.8,
        label=f"Raw photopeak mono T1/2={photopeak_raw_fit['half_life_days']:.2f} j",
    )
    ax.plot(
        fit_x,
        biexponential(
            fit_x,
            photopeak_raw_bi_fit["a1"],
            photopeak_raw_bi_fit["decay_constant_1"],
            photopeak_raw_bi_fit["a2"],
            photopeak_raw_bi_fit["decay_constant_2"],
        ),
        linestyle=":",
        color=raw_color,
        alpha=0.9,
        label=f"Raw photopeak bi Tslow={photopeak_raw_bi_fit['half_life_slow_days']:.2f} j",
    )
    ax.plot(
        fit_x,
        monoexponential(fit_x, planar_fit["initial_value"], planar_fit["decay_constant"]),
        linestyle="--",
        color=tew_color,
        alpha=0.8,
        label=f"Planar frameTEW mono T1/2={planar_fit['half_life_days']:.2f} j",
    )
    ax.plot(
        fit_x,
        biexponential(fit_x, planar_bi_fit["a1"], planar_bi_fit["decay_constant_1"], planar_bi_fit["a2"], planar_bi_fit["decay_constant_2"]),
        linestyle=":",
        color=tew_color,
        alpha=0.9,
        label=f"Planar frameTEW bi Tslow={planar_bi_fit['half_life_slow_days']:.2f} j",
    )
    ax.plot(
        fit_x,
        monoexponential(fit_x, planar_velocity_fit["initial_value"], planar_velocity_fit["decay_constant"]),
        linestyle="--",
        color=velocity_color,
        alpha=0.8,
        label=f"Planar scanTEW mono T1/2={planar_velocity_fit['half_life_days']:.2f} j",
    )
    ax.plot(
        fit_x,
        biexponential(
            fit_x,
            planar_velocity_bi_fit["a1"],
            planar_velocity_bi_fit["decay_constant_1"],
            planar_velocity_bi_fit["a2"],
            planar_velocity_bi_fit["decay_constant_2"],
        ),
        linestyle=":",
        color=velocity_color,
        alpha=0.9,
        label=f"Planar scanTEW bi Tslow={planar_velocity_bi_fit['half_life_slow_days']:.2f} j",
    )
    ax.plot(
        fit_x,
        monoexponential(fit_x, planar_local_fit["initial_value"], planar_local_fit["decay_constant"]),
        linestyle="--",
        color=local_color,
        alpha=0.8,
        label=f"Planar localTEW mono T1/2={planar_local_fit['half_life_days']:.2f} j",
    )
    ax.plot(
        fit_x,
        biexponential(
            fit_x,
            planar_local_bi_fit["a1"],
            planar_local_bi_fit["decay_constant_1"],
            planar_local_bi_fit["a2"],
            planar_local_bi_fit["decay_constant_2"],
        ),
        linestyle=":",
        color=local_color,
        alpha=0.9,
        label=f"Planar localTEW bi Tslow={planar_local_bi_fit['half_life_slow_days']:.2f} j",
    )
    ax.plot(
        fit_x,
        monoexponential(fit_x, planar_total_fit["initial_value"], planar_total_fit["decay_constant"]),
        linestyle="--",
        color=total_color,
        alpha=0.8,
        label=f"Planar total mono T1/2={planar_total_fit['half_life_days']:.2f} j",
    )
    ax.plot(
        fit_x,
        biexponential(
            fit_x,
            planar_total_bi_fit["a1"],
            planar_total_bi_fit["decay_constant_1"],
            planar_total_bi_fit["a2"],
            planar_total_bi_fit["decay_constant_2"],
        ),
        linestyle=":",
        color=total_color,
        alpha=0.9,
        label=f"Planar total bi Tslow={planar_total_bi_fit['half_life_slow_days']:.2f} j",
    )
    ax.plot(
        fit_x,
        monoexponential(fit_x, qspect_fit["initial_value"], qspect_fit["decay_constant"]),
        linestyle="--",
        color=qspect_color,
        alpha=0.8,
        label=f"Q/SPECT mono T1/2={qspect_fit['half_life_days']:.2f} j",
    )
    ax.plot(
        fit_x,
        biexponential(fit_x, qspect_bi_fit["a1"], qspect_bi_fit["decay_constant_1"], qspect_bi_fit["a2"], qspect_bi_fit["decay_constant_2"]),
        linestyle=":",
        color=qspect_color,
        alpha=0.9,
        label=f"Q/SPECT bi Tslow={qspect_bi_fit['half_life_slow_days']:.2f} j",
    )
    ax.set_xlabel("Jour depuis la première acquisition")
    ax.set_ylabel("Activité (MBq)")
    ax.set_title(f"Rétention d'activité planar vs Q/SPECT (sensibilité={planar_processing.CAMERA_SENSITIVITY_CPS_PER_MBQ:g} cps/MBq)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=7.8)
    fig.tight_layout()
    ACTIVITY_FIT_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ACTIVITY_FIT_FIGURE_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved activity fit figure: {ACTIVITY_FIT_FIGURE_PATH}")
    show_figure()


def plot_selected_activity_bi_fit_comparison(pairs: List[Dict[str, Any]]) -> None:
    if not pairs:
        raise ValueError("No paired planar/QSPECT data available")

    x = [pair["day_offset"] for pair in pairs]
    frame_tew_values = [pair["planar_activity_mbq"] for pair in pairs]
    local_tew_values = [pair["planar_tew_local_dwell_activity_mbq"] for pair in pairs]
    qspect_values = [pair["qspect_activity_mbq"] for pair in pairs]
    frame_tew_fit = fit_biexponential(x, frame_tew_values)
    local_tew_fit = fit_biexponential(x, local_tew_values)
    qspect_fit = fit_biexponential(x, qspect_values)
    fit_x = np.linspace(min(x), max(x), 300)

    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor="white")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.plot(
        x,
        frame_tew_values,
        marker="o",
        markersize=7,
        linewidth=0,
        color="#1f77b4",
        label="Acquisition time",
    )
    ax.plot(
        x,
        local_tew_values,
        marker="P",
        markersize=7,
        linewidth=0,
        color="#d62728",
        label="Local dwell time estimate",
    )
    ax.plot(
        x,
        qspect_values,
        marker="s",
        markersize=7,
        linewidth=0,
        color="#ff7f0e",
        label="Q/SPECT activity",
    )
    ax.plot(
        fit_x,
        biexponential(
            fit_x,
            frame_tew_fit["a1"],
            frame_tew_fit["decay_constant_1"],
            frame_tew_fit["a2"],
            frame_tew_fit["decay_constant_2"],
        ),
        linestyle="--",
        color="#1f77b4",
        linewidth=2.4,
        label="Acquisition time biexponential fit",
    )
    ax.plot(
        fit_x,
        biexponential(
            fit_x,
            local_tew_fit["a1"],
            local_tew_fit["decay_constant_1"],
            local_tew_fit["a2"],
            local_tew_fit["decay_constant_2"],
        ),
        linestyle="--",
        color="#d62728",
        linewidth=2.4,
        label="Local dwell time estimate biexponential fit",
    )
    ax.plot(
        fit_x,
        biexponential(
            fit_x,
            qspect_fit["a1"],
            qspect_fit["decay_constant_1"],
            qspect_fit["a2"],
            qspect_fit["decay_constant_2"],
        ),
        linestyle="--",
        color="#ff7f0e",
        linewidth=2.4,
        label="Q/SPECT biexponential fit",
    )
    ax.set_xlabel("Time after first acquisition (days)")
    ax.set_ylabel("Activity estimate (MBq)")
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.3)
    ax.legend(frameon=False, fontsize=10)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    SELECTED_BI_FIT_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SELECTED_BI_FIT_FIGURE_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(SELECTED_BI_FIT_FIGURE_PATH.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    print(f"Saved selected bi-only activity figure: {SELECTED_BI_FIT_FIGURE_PATH}")
    print(f"Saved selected bi-only activity figure: {SELECTED_BI_FIT_FIGURE_PATH.with_suffix('.svg')}")
    show_figure()


def ratio_series_to_qspect(pairs: List[Dict[str, Any]], value_key: str) -> List[float]:
    return [
        float(pair[value_key]) / float(pair["qspect_activity_mbq"])
        for pair in pairs
    ]


def save_ratio_figure(
    pairs: List[Dict[str, Any]],
    series: List[tuple[str, List[float], str, str]],
    output_path: Path,
) -> None:
    if not pairs:
        raise ValueError("No paired planar/QSPECT data available")

    x = [pair["day_offset"] for pair in pairs]
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor="white")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for label, values, color, marker in series:
        ax.plot(
            x,
            values,
            marker=marker,
            markersize=6,
            linewidth=2,
            color=color,
            label=label,
        )

    ax.set_xlabel("Time after first acquisition (days)")
    ax.set_ylabel("Activity ratio vs Q/SPECT")
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    print(f"Saved ratio figure: {output_path}")
    print(f"Saved ratio figure: {output_path.with_suffix('.svg')}")
    show_figure()


def plot_selected_activity_ratios(pairs: List[Dict[str, Any]]) -> None:
    save_ratio_figure(
        pairs,
        [
            (
                "Acquisition time / Q/SPECT",
                ratio_series_to_qspect(pairs, "planar_activity_mbq"),
                "#1f77b4",
                "o",
            ),
            (
                "Local dwell time estimate / Q/SPECT",
                ratio_series_to_qspect(pairs, "planar_tew_local_dwell_activity_mbq"),
                "#d62728",
                "P",
            ),
        ],
        SELECTED_RATIO_FIGURE_PATH,
    )


def plot_all_activity_ratios(pairs: List[Dict[str, Any]]) -> None:
    save_ratio_figure(
        pairs,
        [
            (
                "Raw photopeak / Q/SPECT",
                ratio_series_to_qspect(pairs, "planar_photopeak_raw_activity_mbq"),
                "#9467bd",
                "D",
            ),
            (
                "TEW frame-duration / Q/SPECT",
                ratio_series_to_qspect(pairs, "planar_activity_mbq"),
                "#1f77b4",
                "o",
            ),
            (
                "TEW scan-time / Q/SPECT",
                ratio_series_to_qspect(pairs, "planar_tew_scan_velocity_activity_mbq"),
                "#17becf",
                "^",
            ),
            (
                "TEW local-dwell / Q/SPECT",
                ratio_series_to_qspect(pairs, "planar_tew_local_dwell_activity_mbq"),
                "#d62728",
                "P",
            ),
            (
                "Total windows / Q/SPECT",
                ratio_series_to_qspect(pairs, "planar_total_window_activity_mbq"),
                "#2ca02c",
                "v",
            ),
            (
                "Q/SPECT / Q/SPECT",
                [1.0 for _pair in pairs],
                "#ff7f0e",
                "s",
            ),
        ],
        ALL_RATIO_FIGURE_PATH,
    )


def plot_total_window_activity_fit_comparison(pairs: List[Dict[str, Any]]) -> None:
    if not pairs:
        raise ValueError("No paired planar/QSPECT data available")

    x = [pair["day_offset"] for pair in pairs]
    planar_total_values = [pair["planar_total_window_activity_mbq"] for pair in pairs]
    qspect_values = [pair["qspect_activity_mbq"] for pair in pairs]
    planar_total_bi_fit = fit_biexponential(x, planar_total_values)
    qspect_bi_fit = fit_biexponential(x, qspect_values)
    fit_x = np.linspace(min(x), max(x), 300)

    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor="white")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.plot(x, planar_total_values, marker="^", markersize=7, linewidth=0, color="#2ca02c", label="Planar total windows")
    ax.plot(x, qspect_values, marker="s", markersize=7, linewidth=0, color="#ff7f0e", label="Q/SPECT")
    ax.plot(
        fit_x,
        biexponential(
            fit_x,
            planar_total_bi_fit["a1"],
            planar_total_bi_fit["decay_constant_1"],
            planar_total_bi_fit["a2"],
            planar_total_bi_fit["decay_constant_2"],
        ),
        linestyle="--",
        color="#2ca02c",
        linewidth=2.4,
        label="Planar total-window biexponential fit",
    )
    ax.plot(
        fit_x,
        biexponential(fit_x, qspect_bi_fit["a1"], qspect_bi_fit["decay_constant_1"], qspect_bi_fit["a2"], qspect_bi_fit["decay_constant_2"]),
        linestyle="--",
        color="#ff7f0e",
        linewidth=2.4,
        label="Q/SPECT biexponential fit",
    )
    ax.set_xlabel("Time after first acquisition (days)")
    ax.set_ylabel("Activity estimate (MBq, log scale)")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", linewidth=0.8, alpha=0.3)
    ax.legend(frameon=False, fontsize=10)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    TOTAL_ACTIVITY_FIT_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(TOTAL_ACTIVITY_FIT_FIGURE_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(TOTAL_ACTIVITY_FIT_FIGURE_PATH.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    print(f"Saved total-window activity fit figure: {TOTAL_ACTIVITY_FIT_FIGURE_PATH}")
    print(f"Saved total-window activity fit figure: {TOTAL_ACTIVITY_FIT_FIGURE_PATH.with_suffix('.svg')}")
    show_figure()


def plot_powerpoint_activity_bi_fit(pairs: List[Dict[str, Any]]) -> None:
    if not pairs:
        raise ValueError("No paired planar/QSPECT data available")

    x = [pair["day_offset"] for pair in pairs]
    photopeak_raw_values = [pair["planar_photopeak_raw_activity_mbq"] for pair in pairs]
    planar_values = [pair["planar_activity_mbq"] for pair in pairs]
    planar_velocity_values = [pair["planar_tew_scan_velocity_activity_mbq"] for pair in pairs]
    planar_local_values = [pair["planar_tew_local_dwell_activity_mbq"] for pair in pairs]
    planar_total_values = [pair["planar_total_window_activity_mbq"] for pair in pairs]
    qspect_values = [pair["qspect_activity_mbq"] for pair in pairs]
    photopeak_raw_bi_fit = fit_biexponential(x, photopeak_raw_values)
    planar_bi_fit = fit_biexponential(x, planar_values)
    planar_velocity_bi_fit = fit_biexponential(x, planar_velocity_values)
    planar_local_bi_fit = fit_biexponential(x, planar_local_values)
    planar_total_bi_fit = fit_biexponential(x, planar_total_values)
    qspect_bi_fit = fit_biexponential(x, qspect_values)
    fit_x = np.linspace(min(x), max(x), 300)

    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor="white")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.plot(x, photopeak_raw_values, marker="d", markersize=7, linewidth=0, color="#9467bd", label="Planar raw photopeak")
    ax.plot(x, planar_values, marker="o", markersize=7, linewidth=0, color="#1f77b4", label="Planar TEW")
    ax.plot(x, planar_velocity_values, marker="v", markersize=7, linewidth=0, color="#17becf", label="Planar TEW scan time")
    ax.plot(x, planar_local_values, marker="P", markersize=7, linewidth=0, color="#d62728", label="Planar TEW local dwell")
    ax.plot(x, planar_total_values, marker="^", markersize=7, linewidth=0, color="#2ca02c", label="Planar total windows")
    ax.plot(x, qspect_values, marker="s", markersize=7, linewidth=0, color="#ff7f0e", label="Q/SPECT")
    ax.plot(
        fit_x,
        biexponential(
            fit_x,
            photopeak_raw_bi_fit["a1"],
            photopeak_raw_bi_fit["decay_constant_1"],
            photopeak_raw_bi_fit["a2"],
            photopeak_raw_bi_fit["decay_constant_2"],
        ),
        color="#9467bd",
        linewidth=2.4,
        label="Planar raw photopeak biexponential fit",
    )
    ax.plot(
        fit_x,
        biexponential(fit_x, planar_bi_fit["a1"], planar_bi_fit["decay_constant_1"], planar_bi_fit["a2"], planar_bi_fit["decay_constant_2"]),
        color="#1f77b4",
        linewidth=2.4,
        label=f"Planar biexponential fit",
    )
    ax.plot(
        fit_x,
        biexponential(
            fit_x,
            planar_velocity_bi_fit["a1"],
            planar_velocity_bi_fit["decay_constant_1"],
            planar_velocity_bi_fit["a2"],
            planar_velocity_bi_fit["decay_constant_2"],
        ),
        color="#17becf",
        linewidth=2.4,
        label="Planar TEW scan-time biexponential fit",
    )
    ax.plot(
        fit_x,
        biexponential(
            fit_x,
            planar_local_bi_fit["a1"],
            planar_local_bi_fit["decay_constant_1"],
            planar_local_bi_fit["a2"],
            planar_local_bi_fit["decay_constant_2"],
        ),
        color="#d62728",
        linewidth=2.4,
        label="Planar TEW local-dwell biexponential fit",
    )
    ax.plot(
        fit_x,
        biexponential(
            fit_x,
            planar_total_bi_fit["a1"],
            planar_total_bi_fit["decay_constant_1"],
            planar_total_bi_fit["a2"],
            planar_total_bi_fit["decay_constant_2"],
        ),
        color="#2ca02c",
        linewidth=2.4,
        label="Planar total-window biexponential fit",
    )
    ax.plot(
        fit_x,
        biexponential(fit_x, qspect_bi_fit["a1"], qspect_bi_fit["decay_constant_1"], qspect_bi_fit["a2"], qspect_bi_fit["decay_constant_2"]),
        color="#ff7f0e",
        linewidth=2.4,
        label=f"Q/SPECT biexponential fit",
    )
    ax.set_xlabel("Time after first acquisition (days)")
    ax.set_ylabel("Activity (MBq, log scale)")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", linewidth=0.8, alpha=0.3)
    ax.legend(frameon=False, fontsize=10)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    POWERPOINT_FIT_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(POWERPOINT_FIT_FIGURE_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(POWERPOINT_FIT_FIGURE_PATH.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    print(f"Saved PowerPoint fit figure: {POWERPOINT_FIT_FIGURE_PATH}")
    print(f"Saved PowerPoint fit figure: {POWERPOINT_FIT_FIGURE_PATH.with_suffix('.svg')}")
    show_figure()


def plot_powerpoint_planar_counts_bi_fit(pairs: List[Dict[str, Any]]) -> None:
    if not pairs:
        raise ValueError("No paired planar/QSPECT data available")

    x = [pair["day_offset"] for pair in pairs]
    planar_counts = [pair["planar_tew_counts"] for pair in pairs]
    planar_bi_fit = fit_biexponential(x, planar_counts)
    fit_x = np.linspace(min(x), max(x), 300)
    fit_y = biexponential(
        fit_x,
        planar_bi_fit["a1"],
        planar_bi_fit["decay_constant_1"],
        planar_bi_fit["a2"],
        planar_bi_fit["decay_constant_2"],
    )

    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor="white")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.plot(x, planar_counts, marker="o", markersize=7, linewidth=0, color="#1f77b4", label="Planar TEW corrected counts")
    ax.plot(fit_x, fit_y, linestyle="--", color="#1f77b4", linewidth=2.4, label="Biexponential fit")
    ax.set_xlabel("Time after first acquisition (days)")
    ax.set_ylabel("TEW-corrected counts")
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.3)
    ax.legend(frameon=False, fontsize=10)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    PLANAR_COUNTS_POWERPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLANAR_COUNTS_POWERPOINT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(PLANAR_COUNTS_POWERPOINT_PATH.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    print(f"Saved planar counts PowerPoint figure: {PLANAR_COUNTS_POWERPOINT_PATH}")
    print(f"Saved planar counts PowerPoint figure: {PLANAR_COUNTS_POWERPOINT_PATH.with_suffix('.svg')}")
    show_figure()


def parse_dicom_datetime(scan_dir: Path) -> Optional[datetime]:
    for dicom_path in sorted(scan_dir.glob("*.dcm")):
        try:
            import pydicom

            ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        except Exception:
            continue

        date_text = str(getattr(ds, "AcquisitionDate", None) or getattr(ds, "StudyDate", "")).strip()
        time_text = str(getattr(ds, "AcquisitionTime", "000000")).strip()
        if not date_text:
            continue
        time_text = time_text.split(".")[0].ljust(6, "0")[:6]
        try:
            return datetime.strptime(date_text + time_text, "%Y%m%d%H%M%S")
        except ValueError:
            continue
    return None


def parse_day_from_legacy_scan_name(scan_dir: Path) -> Optional[float]:
    for dicom_path in sorted(scan_dir.glob("*.dcm")):
        match = re.search(r"D(\d+)", dicom_path.name)
        if match:
            return float(match.group(1))
    return None


def legacy_1j92g763_planar_tew_decay_data(patient_planar_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for scan_dir in sorted(path for path in patient_planar_dir.glob("scan*") if path.is_dir()):
        images = dicom_loader.load_scan_directory(scan_dir)
        if not images:
            continue

        remapped_images = []
        for item in images:
            mapped = item.copy()
            view = str(mapped.get("view", "")).lower()
            if view == "ant":
                mapped["view"] = "AP"
            elif view == "post":
                mapped["view"] = "PA"
            remapped_images.append(mapped)

        geometric_images = planar_processing.geometric_mean_images(remapped_images)
        correction = planar_processing.apply_tew_correction(geometric_images)
        duration_seconds = planar_processing.acquisition_duration_seconds(remapped_images)
        correction = planar_processing.apply_dead_time_to_tew_correction(geometric_images, correction, duration_seconds)
        scan_datetime = parse_dicom_datetime(scan_dir)
        fallback_day = parse_day_from_legacy_scan_name(scan_dir)
        dead_time_info = correction["dead_time"] or {}
        rows.append(
            {
                "label": scan_dir.name,
                "datetime": scan_datetime,
                "fallback_day": fallback_day,
                "tew_corrected_counts": correction["corrected_counts"],
                "tew_corrected_counts_before_dead_time": correction["corrected_counts_before_dead_time"],
                "tew_corrected_cps": correction["corrected_counts"] / duration_seconds,
                "tew_corrected_cps_before_dead_time": correction["corrected_counts_before_dead_time"] / duration_seconds,
                "duration_seconds": duration_seconds,
                "planar_activity_mbq": planar_processing.counts_to_activity_mbq(correction["corrected_counts"], duration_seconds),
                "dead_time_dtcf": dead_time_info.get("dtcf", 1.0),
                "dead_time_loss_percent": dead_time_info.get("count_loss_percent", 0.0),
                "wide_spectrum_cps": dead_time_info.get("rwo_cps"),
            }
        )

    rows.sort(key=lambda row: (row["datetime"] is None, row["datetime"], row["fallback_day"] or 0.0, row["label"]))
    first_datetime = next((row["datetime"] for row in rows if row["datetime"] is not None), None)
    for row in rows:
        if first_datetime is not None and row["datetime"] is not None:
            row["day_offset"] = (row["datetime"] - first_datetime).total_seconds() / 86400.0
        elif row["fallback_day"] is not None:
            row["day_offset"] = row["fallback_day"]
        else:
            row["day_offset"] = float(len(rows))
    return rows


def plot_powerpoint_patient_4v6_planar_counts_bi_fit(root_dir: Path = PATIENT_4V6_ROOT) -> None:
    patient_dirs = {
        "Patient 4": root_dir / "patient_4" / "PlanarWholeBodyScans",
        "Patient 6": root_dir / "patient_6" / "PlanarWholeBodyScans",
    }
    if not all(path.exists() for path in patient_dirs.values()):
        print(f"Skipping patient 4v6 figure; missing dataset root: {root_dir}")
        return

    colors = {"Patient 4": "#1f77b4", "Patient 6": "#ff7f0e"}
    markers = {"Patient 4": "o", "Patient 6": "s"}

    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor="white")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for label, patient_dir in patient_dirs.items():
        rows = legacy_1j92g763_planar_tew_decay_data(patient_dir)
        x = [row["day_offset"] for row in rows]
        y = [row["tew_corrected_counts"] for row in rows]
        fit = fit_biexponential(x, y)
        fit_x = np.linspace(min(x), max(x), 300)
        fit_y = biexponential(
            fit_x,
            fit["a1"],
            fit["decay_constant_1"],
            fit["a2"],
            fit["decay_constant_2"],
        )
        ax.plot(x, y, marker=markers[label], markersize=7, linewidth=0, color=colors[label], label=label)
        ax.plot(fit_x, fit_y, linestyle="--", color=colors[label], linewidth=2.4, label=f"{label} biexponential fit")
        print(f"{label} 1j92g763 planar TEW counts:")
        for row in rows:
            print(
                f"  day={row['day_offset']:.2f} | "
                f"before={row['tew_corrected_counts_before_dead_time']:.1f} | "
                f"after={row['tew_corrected_counts']:.1f} | "
                f"DTCF={row['dead_time_dtcf']:.4f} | "
                f"loss={row['dead_time_loss_percent']:.2f}% | "
                f"label={row['label']}"
            )

    ax.set_xlabel("Time after first acquisition (days)")
    ax.set_ylabel("TEW-corrected counts")
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.3)
    ax.legend(frameon=False, fontsize=10)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    PATIENT_4V6_COUNTS_POWERPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PATIENT_4V6_COUNTS_POWERPOINT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(PATIENT_4V6_COUNTS_POWERPOINT_PATH.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    print(f"Saved patient 4v6 planar counts figure: {PATIENT_4V6_COUNTS_POWERPOINT_PATH}")
    print(f"Saved patient 4v6 planar counts figure: {PATIENT_4V6_COUNTS_POWERPOINT_PATH.with_suffix('.svg')}")
    show_figure()


def print_half_life_fits(pairs: List[Dict[str, Any]]) -> None:
    x = [pair["day_offset"] for pair in pairs]
    planar_rates = [pair["planar_tew_cps"] for pair in pairs]
    photopeak_raw_activity = [pair["planar_photopeak_raw_activity_mbq"] for pair in pairs]
    planar_velocity_activity = [pair["planar_tew_scan_velocity_activity_mbq"] for pair in pairs]
    planar_local_activity = [pair["planar_tew_local_dwell_activity_mbq"] for pair in pairs]
    planar_total_activity = [pair["planar_total_window_activity_mbq"] for pair in pairs]
    qspect_values = [pair["qspect_activity_mbq"] for pair in pairs]
    print("Fitting retention models...", flush=True)
    planar_mono_fit = fit_monoexponential(x, planar_rates)
    planar_bi_fit = fit_biexponential(x, planar_rates)
    photopeak_raw_mono_fit = fit_monoexponential(x, photopeak_raw_activity)
    photopeak_raw_bi_fit = fit_biexponential(x, photopeak_raw_activity)
    planar_velocity_mono_fit = fit_monoexponential(x, planar_velocity_activity)
    planar_velocity_bi_fit = fit_biexponential(x, planar_velocity_activity)
    planar_local_mono_fit = fit_monoexponential(x, planar_local_activity)
    planar_local_bi_fit = fit_biexponential(x, planar_local_activity)
    planar_total_mono_fit = fit_monoexponential(x, planar_total_activity)
    planar_total_bi_fit = fit_biexponential(x, planar_total_activity)
    qspect_mono_fit = fit_monoexponential(x, qspect_values)
    qspect_bi_fit = fit_biexponential(x, qspect_values)
    print("Retention time-activity fits:", flush=True)
    print(describe_mono_fit("Planar raw photopeak activity", photopeak_raw_mono_fit))
    print(describe_bi_fit("Planar raw photopeak activity", photopeak_raw_bi_fit))
    print(describe_mono_fit("Planar TEW count rate", planar_mono_fit))
    print(describe_bi_fit("Planar TEW count rate", planar_bi_fit))
    print(describe_mono_fit("Planar TEW scan-velocity activity", planar_velocity_mono_fit))
    print(describe_bi_fit("Planar TEW scan-velocity activity", planar_velocity_bi_fit))
    print(describe_mono_fit("Planar TEW local-dwell activity", planar_local_mono_fit))
    print(describe_bi_fit("Planar TEW local-dwell activity", planar_local_bi_fit))
    print(describe_mono_fit("Planar total-window activity", planar_total_mono_fit))
    print(describe_bi_fit("Planar total-window activity", planar_total_bi_fit))
    print(describe_mono_fit("Q/SPECT activity", qspect_mono_fit))
    print(describe_bi_fit("Q/SPECT activity", qspect_bi_fit))


def run_comparison(planar_dir: Path = default_planar_dir(), qspect_dir: Path = default_qspect_dir()) -> None:
    planar_rows = planar_processing.planar_tew_decay_data(planar_dir)
    qspect_rows = qspect_processing.qspect_decay_data(qspect_dir)
    pairs = paired_decay_data(planar_rows, qspect_rows)
    print_pairs(pairs)
    write_activity_fit_values_report(pairs)
    print_half_life_fits(pairs)
    plot_activity_fit_comparison(pairs)
    plot_selected_activity_bi_fit_comparison(pairs)
    plot_selected_activity_ratios(pairs)
    plot_all_activity_ratios(pairs)
    plot_total_window_activity_fit_comparison(pairs)
    plot_powerpoint_activity_bi_fit(pairs)
    plot_powerpoint_planar_counts_bi_fit(pairs)
    if RUN_PATIENT_4V6_FIGURE:
        plot_powerpoint_patient_4v6_planar_counts_bi_fit()
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
