"""Exploratory planar/excretion closure using independent bi-exponential fits.

All activities are first placed on an injection-time-equivalent scale using
only the Lu-177 physical half-life.  Two fits are then performed independently:

* planar whole-body activity: decreasing bi-exponential constrained to the
  administered activity at t=0;
* cumulative urine/diaper activity: increasing saturating bi-exponential.

The module evaluates the excretion fit at every planar acquisition, performs
the reciprocal comparison by evaluating the planar fit at every collection
time, and finally adds both fitted curves on a common time grid.  The fit is a
time-alignment experiment, not a calibration of either measurement.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.optimize import least_squares

import sang_urine


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "fig" / "planar_excretion_fit_balance"
VALUES_PATH = OUTPUT_DIR / "planar_excretion_fit_balance_values.csv"
REPORT_PATH = OUTPUT_DIR / "planar_excretion_fit_balance_report.txt"


def planar_biexponential(
    elapsed_h: np.ndarray | float,
    amplitude_fast_mbq: float,
    amplitude_slow_mbq: float,
    rate_fast_h_inv: float,
    rate_slow_h_inv: float,
) -> np.ndarray:
    """Decreasing bi-exponential for injection-equivalent body activity."""
    time_h = np.asarray(elapsed_h, dtype=float)
    return amplitude_fast_mbq * np.exp(-rate_fast_h_inv * time_h) + (
        amplitude_slow_mbq * np.exp(-rate_slow_h_inv * time_h)
    )


def cumulative_excretion_biexponential(
    elapsed_h: np.ndarray | float,
    asymptote_mbq: float,
    fast_fraction: float,
    rate_fast_h_inv: float,
    rate_slow_h_inv: float,
) -> np.ndarray:
    """Increasing bi-exponential constrained to zero cumulative activity at t=0."""
    time_h = np.asarray(elapsed_h, dtype=float)
    return asymptote_mbq * (
        fast_fraction * (1.0 - np.exp(-rate_fast_h_inv * time_h))
        + (1.0 - fast_fraction) * (1.0 - np.exp(-rate_slow_h_inv * time_h))
    )


def anchored_body_biexponential(
    elapsed_h: np.ndarray | float,
    injected_mbq: float,
    fast_fraction: float,
    rate_fast_h_inv: float,
    rate_slow_h_inv: float,
) -> np.ndarray:
    """Decreasing bi-exponential constrained to equal ``injected_mbq`` at t=0."""
    return planar_biexponential(
        elapsed_h,
        injected_mbq * fast_fraction,
        injected_mbq * (1.0 - fast_fraction),
        rate_fast_h_inv,
        rate_slow_h_inv,
    )


def _best_bounded_fit(
    model: Any,
    times_h: np.ndarray,
    values_mbq: np.ndarray,
    starts: Iterable[Sequence[float]],
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
) -> np.ndarray:
    """Run several bounded nonlinear least-squares starts and keep the best fit."""
    best_result = None
    for start in starts:
        result = least_squares(
            lambda parameters: model(times_h, *parameters) - values_mbq,
            np.asarray(start, dtype=float),
            bounds=(np.asarray(lower_bounds), np.asarray(upper_bounds)),
            max_nfev=100_000,
        )
        if not result.success:
            continue
        if best_result is None or np.sum(result.fun**2) < np.sum(best_result.fun**2):
            best_result = result
    if best_result is None:
        raise RuntimeError("Bi-exponential fit did not converge")
    return np.asarray(best_result.x, dtype=float)


def _order_planar_components(parameters: np.ndarray) -> np.ndarray:
    amplitude_fast, amplitude_slow, rate_fast, rate_slow = parameters
    if rate_fast < rate_slow:
        return np.asarray(
            [amplitude_slow, amplitude_fast, rate_slow, rate_fast], dtype=float
        )
    return parameters


def _order_excretion_components(parameters: np.ndarray) -> np.ndarray:
    asymptote, fast_fraction, rate_fast, rate_slow = parameters
    if rate_fast < rate_slow:
        return np.asarray(
            [asymptote, 1.0 - fast_fraction, rate_slow, rate_fast], dtype=float
        )
    return parameters


def fit_planar_and_excretion(
    data: Dict[str, Any],
    balance: Dict[str, np.ndarray],
    imaging: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Fit both independent curves after physical-decay compensation."""
    injected_mbq = float(data["injected_activity_mbq"])
    half_life_h = float(data["half_life_h"])
    planar_times_h = np.asarray(imaging["planar_times_h"], dtype=float)
    planar_measured_mbq = np.asarray(imaging["planar_ctac_mbq"], dtype=float)
    planar_equivalent_mbq = planar_measured_mbq / sang_urine.decay_factor(
        planar_times_h, half_life_h
    )
    excretion_times_h = np.asarray(balance["times_h"], dtype=float)
    excretion_equivalent_mbq = np.asarray(
        balance["cumulative_injection_equivalent_mbq"], dtype=float
    )

    planar_starts = [
        (0.80, 0.10, 0.004),
        (0.60, 0.03, 0.003),
        (0.90, 0.50, 0.001),
        (0.50, 0.20, 0.010),
    ]
    constrained_planar_parameters = _best_bounded_fit(
        lambda time_h, fraction, rate_fast, rate_slow: (
            anchored_body_biexponential(
                time_h,
                injected_mbq,
                fraction,
                rate_fast,
                rate_slow,
            )
        ),
        planar_times_h,
        planar_equivalent_mbq,
        planar_starts,
        (0.0, 1.0e-6, 1.0e-6),
        (1.0, 5.0, 5.0),
    )
    fast_fraction, rate_fast, rate_slow = constrained_planar_parameters
    if rate_fast < rate_slow:
        fast_fraction = 1.0 - fast_fraction
        rate_fast, rate_slow = rate_slow, rate_fast
    planar_parameters = np.asarray(
        [
            injected_mbq * fast_fraction,
            injected_mbq * (1.0 - fast_fraction),
            rate_fast,
            rate_slow,
        ],
        dtype=float,
    )

    maximum_measured_excretion = float(excretion_equivalent_mbq.max())
    excretion_starts = [
        (0.90 * injected_mbq, 0.80, 0.10, 0.005),
        (0.95 * injected_mbq, 0.50, 0.30, 0.020),
        (0.88 * injected_mbq, 0.90, 1.00, 0.010),
        (injected_mbq, 0.30, 0.05, 0.003),
    ]
    excretion_parameters = _order_excretion_components(
        _best_bounded_fit(
            cumulative_excretion_biexponential,
            excretion_times_h,
            excretion_equivalent_mbq,
            excretion_starts,
            (maximum_measured_excretion, 0.0, 1.0e-6, 1.0e-6),
            (injected_mbq, 1.0, 5.0, 5.0),
        )
    )

    planar_fit_at_planar_mbq = planar_biexponential(
        planar_times_h, *planar_parameters
    )
    excretion_fit_at_excretion_mbq = cumulative_excretion_biexponential(
        excretion_times_h, *excretion_parameters
    )
    planar_residuals_mbq = planar_fit_at_planar_mbq - planar_equivalent_mbq
    excretion_residuals_mbq = (
        excretion_fit_at_excretion_mbq - excretion_equivalent_mbq
    )

    return {
        "injected_mbq": injected_mbq,
        "half_life_h": half_life_h,
        "labels": np.asarray(imaging["labels"]),
        "planar_times_h": planar_times_h,
        "planar_measured_mbq": planar_measured_mbq,
        "planar_equivalent_mbq": planar_equivalent_mbq,
        "excretion_times_h": excretion_times_h,
        "excretion_equivalent_mbq": excretion_equivalent_mbq,
        "planar_parameters": planar_parameters,
        "planar_initial_activity_forced_mbq": injected_mbq,
        "excretion_parameters": excretion_parameters,
        "planar_fit_at_planar_mbq": planar_fit_at_planar_mbq,
        "excretion_fit_at_excretion_mbq": excretion_fit_at_excretion_mbq,
        "planar_rmse_mbq": float(np.sqrt(np.mean(planar_residuals_mbq**2))),
        "excretion_rmse_mbq": float(
            np.sqrt(np.mean(excretion_residuals_mbq**2))
        ),
    }


def calculate_cross_time_balances(fits: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate each fit at the other measurement's time points."""
    injected_mbq = float(fits["injected_mbq"])
    planar_times_h = np.asarray(fits["planar_times_h"], dtype=float)
    excretion_times_h = np.asarray(fits["excretion_times_h"], dtype=float)
    planar_equivalent_mbq = np.asarray(fits["planar_equivalent_mbq"], dtype=float)
    excretion_equivalent_mbq = np.asarray(
        fits["excretion_equivalent_mbq"], dtype=float
    )
    planar_parameters = np.asarray(fits["planar_parameters"], dtype=float)
    excretion_parameters = np.asarray(fits["excretion_parameters"], dtype=float)

    excretion_fit_at_planar_mbq = cumulative_excretion_biexponential(
        planar_times_h, *excretion_parameters
    )
    sum_at_planar_mbq = planar_equivalent_mbq + excretion_fit_at_planar_mbq
    planar_fit_at_excretion_mbq = planar_biexponential(
        excretion_times_h, *planar_parameters
    )
    sum_at_excretion_mbq = planar_fit_at_excretion_mbq + excretion_equivalent_mbq

    last_collection_h = float(excretion_times_h.max())
    planar_rows: List[Dict[str, Any]] = []
    for index, label in enumerate(fits["labels"]):
        planar_rows.append(
            {
                "comparison": "measured_planar_plus_fitted_excretion",
                "label": str(label),
                "elapsed_h": float(planar_times_h[index]),
                "measured_component_injection_equivalent_mbq": float(
                    planar_equivalent_mbq[index]
                ),
                "fitted_component_injection_equivalent_mbq": float(
                    excretion_fit_at_planar_mbq[index]
                ),
                "sum_injection_equivalent_mbq": float(sum_at_planar_mbq[index]),
                "closure_ratio": float(sum_at_planar_mbq[index] / injected_mbq),
                "closure_error_percent": float(
                    100.0 * (sum_at_planar_mbq[index] / injected_mbq - 1.0)
                ),
                "fit_is_extrapolated_after_last_collection": bool(
                    planar_times_h[index] > last_collection_h
                ),
            }
        )

    excretion_rows: List[Dict[str, Any]] = []
    for index, time_h in enumerate(excretion_times_h):
        excretion_rows.append(
            {
                "comparison": "measured_excretion_plus_fitted_planar",
                "label": f"Collecte {index + 1}",
                "elapsed_h": float(time_h),
                "measured_component_injection_equivalent_mbq": float(
                    excretion_equivalent_mbq[index]
                ),
                "fitted_component_injection_equivalent_mbq": float(
                    planar_fit_at_excretion_mbq[index]
                ),
                "sum_injection_equivalent_mbq": float(sum_at_excretion_mbq[index]),
                "closure_ratio": float(sum_at_excretion_mbq[index] / injected_mbq),
                "closure_error_percent": float(
                    100.0 * (sum_at_excretion_mbq[index] / injected_mbq - 1.0)
                ),
                "fit_is_extrapolated_after_last_collection": False,
            }
        )

    return {
        "excretion_fit_at_planar_mbq": excretion_fit_at_planar_mbq,
        "sum_at_planar_mbq": sum_at_planar_mbq,
        "planar_fit_at_excretion_mbq": planar_fit_at_excretion_mbq,
        "sum_at_excretion_mbq": sum_at_excretion_mbq,
        "planar_rows": planar_rows,
        "excretion_rows": excretion_rows,
    }


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(True, axis="y", linestyle="--", alpha=0.3)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _stacked_balance_figure(
    rows: List[Dict[str, Any]],
    measured_label: str,
    fitted_label: str,
    title: str,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    positions = np.arange(len(rows))
    measured = np.asarray(
        [row["measured_component_injection_equivalent_mbq"] for row in rows]
    )
    fitted = np.asarray(
        [row["fitted_component_injection_equivalent_mbq"] for row in rows]
    )
    totals = measured + fitted
    injected_mbq = totals / np.asarray([row["closure_ratio"] for row in rows])
    reference = float(np.mean(injected_mbq))

    fig, axis = plt.subplots(figsize=(9.2, 5.8), facecolor="white", layout="constrained")
    axis.bar(positions, measured, color="#1f77b4", label=measured_label)
    fitted_bars = axis.bar(
        positions,
        fitted,
        bottom=measured,
        color="#ff7f0e",
        label=fitted_label,
    )
    has_extrapolated_fit = False
    for bar, row in zip(fitted_bars, rows):
        if row["fit_is_extrapolated_after_last_collection"]:
            bar.set_hatch("//")
            bar.set_edgecolor("#9a4f00")
            has_extrapolated_fit = True
    axis.axhline(
        reference,
        color="black",
        linestyle="--",
        linewidth=1.7,
        label=f"Activité injectée ({reference / 1000.0:.1f} GBq)",
    )
    for index, row in enumerate(rows):
        axis.text(
            index,
            totals[index] + 0.018 * reference,
            f"{totals[index] / 1000.0:.2f} GBq\n{row['closure_error_percent']:+.1f} %",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.set_xticks(
        positions,
        [f"{row['label']}\n{row['elapsed_h']:.2f} h" for row in rows],
    )
    axis.set_ylabel("Activité équivalente au temps d'injection (MBq)")
    axis.set_title(title)
    axis.set_ylim(0.0, max(reference, float(totals.max())) * 1.18)
    handles, labels = axis.get_legend_handles_labels()
    if has_extrapolated_fit:
        handles.append(
            Patch(
                facecolor="#ff7f0e",
                edgecolor="#9a4f00",
                hatch="//",
                label="Partie hachurée : extrapolation après la dernière collecte",
            )
        )
        labels.append("Partie hachurée : extrapolation après la dernière collecte")
    axis.legend(
        handles,
        labels,
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
    )
    _style_axis(axis)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_measured_planar_plus_fitted_excretion(
    cross_balances: Dict[str, Any], output_path: Path
) -> Path:
    """Plot closure at each planar acquisition time."""
    return plot_measured_imaging_plus_fitted_excretion(
        cross_balances, output_path, modality_label="Planaire"
    )


def plot_measured_imaging_plus_fitted_excretion(
    cross_balances: Dict[str, Any],
    output_path: Path,
    modality_label: str,
) -> Path:
    """Plot closure at each acquisition time for an imaging modality."""
    time_label = "planaires" if modality_label == "Planaire" else modality_label
    subject_label = "planaire" if modality_label == "Planaire" else modality_label
    return _stacked_balance_figure(
        cross_balances["planar_rows"],
        f"{modality_label} mesuré, corrigé de la décroissance physique",
        "Excrétion cumulative — fit bi-exponentiel",
        (
            f"Bilan aux temps {time_label} : "
            f"{subject_label} mesuré + excrétion ajustée"
        ),
        output_path,
    )


def plot_measured_excretion_plus_fitted_planar(
    cross_balances: Dict[str, Any], output_path: Path
) -> Path:
    """Plot reciprocal closure at each urine collection time."""
    return plot_measured_excretion_plus_fitted_imaging(
        cross_balances, output_path, modality_label="Planaire"
    )


def plot_measured_excretion_plus_fitted_imaging(
    cross_balances: Dict[str, Any],
    output_path: Path,
    modality_label: str,
) -> Path:
    """Plot reciprocal closure at collection times for an imaging modality."""
    subject_label = "planaire" if modality_label == "Planaire" else modality_label
    return _stacked_balance_figure(
        cross_balances["excretion_rows"],
        "Excrétion cumulative mesurée, corrigée de la décroissance physique",
        f"Activité corporelle {modality_label} — fit bi-exponentiel",
        (
            "Bilan aux temps d'excrétion : excrétion mesurée + "
            f"{subject_label} ajusté"
        ),
        output_path,
    )


def plot_sum_of_biexponential_fits(
    fits: Dict[str, Any],
    output_path: Path,
    modality_label: str = "Planaire",
) -> Path:
    """Plot both independently fitted curves and their continuous sum."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    maximum_time_h = float(np.max(fits["planar_times_h"]))
    last_collection_h = float(np.max(fits["excretion_times_h"]))
    curve_times_h = np.linspace(0.0, maximum_time_h, 1200)
    planar_curve_mbq = planar_biexponential(
        curve_times_h, *fits["planar_parameters"]
    )
    excretion_curve_mbq = cumulative_excretion_biexponential(
        curve_times_h, *fits["excretion_parameters"]
    )
    total_curve_mbq = planar_curve_mbq + excretion_curve_mbq
    injected_mbq = float(fits["injected_mbq"])

    fig, axis = plt.subplots(figsize=(9.2, 5.8), facecolor="white", layout="constrained")
    axis.axvspan(
        last_collection_h,
        maximum_time_h,
        color="#7f7f7f",
        alpha=0.10,
        label="Extrapolation du fit d'excrétion après la dernière collecte",
    )
    axis.plot(
        curve_times_h,
        planar_curve_mbq,
        color="#1f77b4",
        linewidth=2.1,
        label=f"Fit bi-exponentiel {modality_label}",
    )
    axis.scatter(
        fits["planar_times_h"],
        fits["planar_equivalent_mbq"],
        color="#1f77b4",
        marker="s",
        s=42,
        zorder=4,
        label=f"Mesures {modality_label}",
    )
    axis.plot(
        curve_times_h,
        excretion_curve_mbq,
        color="#ff7f0e",
        linewidth=2.1,
        label="Fit bi-exponentiel de l'excrétion cumulative",
    )
    axis.scatter(
        fits["excretion_times_h"],
        fits["excretion_equivalent_mbq"],
        color="#ff7f0e",
        marker="o",
        s=42,
        zorder=4,
        label="Mesures d'excrétion cumulative",
    )
    axis.plot(
        curve_times_h,
        total_curve_mbq,
        color="#2ca02c",
        linewidth=2.8,
        label="Somme des deux fits",
    )
    axis.axhline(
        injected_mbq,
        color="black",
        linestyle="--",
        linewidth=1.7,
        label=f"Activité injectée ({injected_mbq / 1000.0:.1f} GBq)",
    )
    axis.set_xlabel("Temps après l'injection (h)")
    axis.set_ylabel("Activité équivalente au temps d'injection (MBq)")
    axis.set_title(
        f"Somme des fits : {modality_label} corporel + excrétion "
        "(ancrage à t=0)"
    )
    axis.legend(
        frameon=False,
        fontsize=8,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
    )
    _style_axis(axis)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def write_values(
    cross_balances: Dict[str, Any], output_path: Path = VALUES_PATH
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = cross_balances["planar_rows"] + cross_balances["excretion_rows"]
    fieldnames = list(rows[0])
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: round(float(value), 6)
                    if isinstance(value, (float, np.floating))
                    else value
                    for key, value in row.items()
                }
            )
    return output_path


def write_report(
    fits: Dict[str, Any],
    cross_balances: Dict[str, Any],
    output_path: Path = REPORT_PATH,
    modality_label: str = "Planar",
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    planar_parameters = fits["planar_parameters"]
    excretion_parameters = fits["excretion_parameters"]
    lines = [
        f"{modality_label}/excretion bi-exponential mass-balance experiment",
        "===============================================================",
        "",
        "Method:",
        "  1. Every measured activity is corrected only for Lu-177 physical decay",
        "     to obtain an injection-time-equivalent activity.",
        f"  2. {modality_label} body activity and cumulative excretion are fitted",
        "     separately by bounded nonlinear least squares.",
        "  3. The body fit is constrained to P(0) = 7200 MBq.",
        "  4. One fit is evaluated at the other measurement's time points.",
        "  5. The two fitted curves are added and compared with 7200 MBq.",
        "",
        f"{modality_label} fit:",
        "  P(t) = A_fast exp(-k_fast t) + A_slow exp(-k_slow t)",
        "  Constraint: A_fast + A_slow = P(0) = 7200 MBq",
        f"  A_fast = {planar_parameters[0]:.3f} MBq",
        f"  A_slow = {planar_parameters[1]:.3f} MBq",
        f"  k_fast = {planar_parameters[2]:.8f} h^-1",
        f"  k_slow = {planar_parameters[3]:.8f} h^-1",
        f"  RMSE = {fits['planar_rmse_mbq']:.3f} MBq",
        "",
        "Cumulative-excretion fit:",
        "  E(t) = E_inf [f(1-exp(-k_fast t)) + (1-f)(1-exp(-k_slow t))]",
        f"  E_inf = {excretion_parameters[0]:.3f} MBq",
        f"  f = {excretion_parameters[1]:.8f}",
        f"  k_fast = {excretion_parameters[2]:.8f} h^-1",
        f"  k_slow = {excretion_parameters[3]:.8f} h^-1",
        f"  RMSE = {fits['excretion_rmse_mbq']:.3f} MBq",
        "  Note: E_inf reaches the physical upper bound of 7200 MBq; the slow",
        "  excretion tail is therefore weakly identifiable from only 46.62 h of",
        "  collection data.",
        "",
        f"Measured {modality_label} + fitted excretion:",
        " label | time h | recovered MBq | error % | excretion extrapolated",
        "------ | ------ | ------------- | ------- | ----------------------",
    ]
    for row in cross_balances["planar_rows"]:
        lines.append(
            f"{row['label']:6s} | {row['elapsed_h']:6.2f} | "
            f"{row['sum_injection_equivalent_mbq']:13.2f} | "
            f"{row['closure_error_percent']:+7.2f} | "
            f"{str(row['fit_is_extrapolated_after_last_collection']):>22s}"
        )
    lines.extend(
        [
            "",
            f"Measured excretion + fitted {modality_label}:",
            " label          | time h | recovered MBq | error %",
            "-------------- | ------ | ------------- | -------",
        ]
    )
    for row in cross_balances["excretion_rows"]:
        lines.append(
            f"{row['label']:14s} | {row['elapsed_h']:6.2f} | "
            f"{row['sum_injection_equivalent_mbq']:13.2f} | "
            f"{row['closure_error_percent']:+7.2f}"
        )
    lines.extend(
        [
            "",
            "Interpretation limits:",
            "  The fits are used only for temporal interpolation/extrapolation.",
            "  The body fit is explicitly forced to 7200 MBq at t=0; consequently,",
            "  the fitted-sum curve cannot independently validate the injected",
            "  activity at t=0.",
            "  The body fit has 3 free parameters for only 5 measurements; the excretion",
            "  fit has 4 parameters for 6 measurements. This is exploratory and can",
            "  overfit one patient.",
            "  Excretion after the final 46.62 h collection is model extrapolation,",
            f"  especially for the Day3 and Day6 {modality_label} comparisons.",
            "  Before the first 2.28 h collection endpoint, the fit assumes smooth",
            "  cumulative excretion; the exact time of a discrete void is unknown.",
            "  Blood concentrations are not added to the whole-body mass balance.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def run_planar_excretion_fit_balance(
    output_dir: Path = OUTPUT_DIR,
) -> Dict[str, Any]:
    """Run the full comparison and generate three 300-dpi PNG figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = sang_urine.load_sang_urine_csv()
    balance = sang_urine.calculate_urine_mass_balance(data)
    imaging = sang_urine.load_imaging_comparison(data["injection_datetime"])
    if imaging is None:
        raise FileNotFoundError("The fixed-J0 planar activity table is unavailable")
    fits = fit_planar_and_excretion(data, balance, imaging)
    cross_balances = calculate_cross_time_balances(fits)
    paths = {
        "planar_times": plot_measured_planar_plus_fitted_excretion(
            cross_balances,
            output_dir / "01_planar_measured_plus_excretion_fit.png",
        ),
        "excretion_times": plot_measured_excretion_plus_fitted_planar(
            cross_balances,
            output_dir / "02_excretion_measured_plus_planar_fit.png",
        ),
        "continuous_sum": plot_sum_of_biexponential_fits(
            fits,
            output_dir / "03_sum_of_planar_and_excretion_fits.png",
        ),
        "values": write_values(
            cross_balances, output_dir / VALUES_PATH.name
        ),
        "report": write_report(
            fits, cross_balances, output_dir / REPORT_PATH.name
        ),
    }
    return {
        "data": data,
        "balance": balance,
        "imaging": imaging,
        "fits": fits,
        "cross_balances": cross_balances,
        "paths": paths,
    }


def main() -> None:
    result = run_planar_excretion_fit_balance()
    print("Measured planar + fitted excretion:")
    for row in result["cross_balances"]["planar_rows"]:
        print(
            f"  {row['label']}: {row['sum_injection_equivalent_mbq']:.2f} MBq "
            f"({row['closure_error_percent']:+.2f} %)"
        )
    print("Measured excretion + fitted planar:")
    for row in result["cross_balances"]["excretion_rows"]:
        print(
            f"  {row['elapsed_h']:.2f} h: "
            f"{row['sum_injection_equivalent_mbq']:.2f} MBq "
            f"({row['closure_error_percent']:+.2f} %)"
        )
    for key, path in result["paths"].items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
