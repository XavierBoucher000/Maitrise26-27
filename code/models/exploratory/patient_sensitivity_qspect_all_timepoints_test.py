"""Tester les trois contraintes de pente avec tous les timepoints Q/SPECT.

Mode fixe: Q/SPECT crop, photopeak brut et moyenne geometrique globale.
"""

from __future__ import annotations

import csv
from pathlib import Path

from core.project_paths import PROJECT_ROOT
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import models.patient_sensitivity_qspect_slopes as slopes


PROJECT_DIR = PROJECT_ROOT
OUTPUT_DIR = (
    PROJECT_DIR
    / "fig/models/patient_sensitivity_qspect_slopes/shared/test_tous_timepoints"
)
DATASET_ORDER = (
    "p11_2026_05_studies",
    "p11_2026_07_studies",
    "p8_2026_06_studies",
    "p17_2026_09_studies",
)
METHODS = (
    ("origin", "Origine imposée"),
    ("free_intercept", "Intercept libre"),
    ("anchored_j0", "Ancrage à J0"),
)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return path
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _group_all(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for source in rows:
        grouped.setdefault(str(source["dataset_key"]), []).append(dict(source))
    return {
        key: sorted(cycle_rows, key=lambda row: float(row["planar_day"]))
        for key, cycle_rows in grouped.items()
        if key in DATASET_ORDER
    }


def _cycle_label(row: Mapping[str, Any]) -> str:
    patient = str(row["patient_id"]).upper()
    month = str(row["cycle_id"])[5:7]
    month_label = {
        "05": "mai", "06": "juin", "07": "juillet", "09": "sept."
    }.get(month, month)
    return f"{patient}\n{month_label}"


def _activity_rate(cycle_rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    activity = np.asarray(
        [float(row["reference_activity_mbq"]) for row in cycle_rows], dtype=float
    )
    rate = np.asarray(
        [slopes.corrected_rate(row, "photopeak", "gm_global") for row in cycle_rows],
        dtype=float,
    )
    return activity, rate


def _mare(activity: np.ndarray, rate: np.ndarray, fit: Mapping[str, Any]) -> float:
    predicted = (
        rate - float(fit["intercept_cps"])
    ) / float(fit["slope_cps_per_mbq"])
    return float(np.mean(np.abs(100.0 * (predicted / activity - 1.0))))


def build_results(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for dataset_key in DATASET_ORDER:
        cycle_rows = list(grouped[dataset_key])
        first_three = cycle_rows[:3]
        activity_three, rate_three = _activity_rate(first_three)
        activity_all, rate_all = _activity_rate(cycle_rows)
        example = cycle_rows[0]
        for method, method_label in METHODS:
            fit_three = slopes.linear_sensitivity_fit(
                activity_three, rate_three, method
            )
            fit_all = slopes.linear_sensitivity_fit(activity_all, rate_all, method)
            results.append(
                {
                    "dataset_key": dataset_key,
                    "patient_id": example["patient_id"],
                    "cycle_id": example["cycle_id"],
                    "fit_method": method,
                    "fit_method_label": method_label,
                    "number_of_first_points": len(first_three),
                    "number_of_all_points": len(cycle_rows),
                    "slope_first3_cps_per_mbq": fit_three["slope_cps_per_mbq"],
                    "intercept_first3_cps": fit_three["intercept_cps"],
                    "mare_first3_fit_on_first3_percent": _mare(
                        activity_three, rate_three, fit_three
                    ),
                    "mare_first3_fit_on_all_percent": _mare(
                        activity_all, rate_all, fit_three
                    ),
                    "slope_all_cps_per_mbq": fit_all["slope_cps_per_mbq"],
                    "intercept_all_cps": fit_all["intercept_cps"],
                    "mare_all_fit_on_all_percent": _mare(
                        activity_all, rate_all, fit_all
                    ),
                    "delta_mare_all_fit_minus_first3_fit_on_all_percent_points": (
                        _mare(activity_all, rate_all, fit_all)
                        - _mare(activity_all, rate_all, fit_three)
                    ),
                }
            )
    return results


def plot_all_point_fits(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    method: str,
    output_path: Path,
) -> Path:
    method_label = next(label for key, label in METHODS if key == method)
    figure, axes = plt.subplots(
        1, 4, figsize=(18.0, 5.8), facecolor="white", layout="constrained"
    )
    for axis, dataset_key in zip(axes, DATASET_ORDER):
        cycle_rows = list(grouped[dataset_key])
        activity, rate = _activity_rate(cycle_rows)
        fit = slopes.linear_sensitivity_fit(activity, rate, method)
        activity_gbq = activity / 1000.0
        rate_kcps = rate / 1000.0
        color = slopes.cycle_color(cycle_rows[0])
        x_line = np.linspace(0.0, float(np.max(activity_gbq)) * 1.04, 180)
        y_line = (
            float(fit["slope_cps_per_mbq"]) * x_line
            + float(fit["intercept_cps"]) / 1000.0
        )
        axis.plot(x_line, y_line, color=color, linewidth=2.4)
        first_count = min(3, len(cycle_rows))
        axis.scatter(
            activity_gbq[:first_count], rate_kcps[:first_count], s=78,
            color=color, edgecolor="white", linewidth=0.9, zorder=3,
            label="J0–J2",
        )
        if len(cycle_rows) > first_count:
            axis.scatter(
                activity_gbq[first_count:], rate_kcps[first_count:], s=86,
                facecolor="white", edgecolor=color, linewidth=2.0, zorder=3,
                label="Points ajoutés",
            )
        for row, x_value, y_value in zip(cycle_rows, activity_gbq, rate_kcps):
            axis.annotate(
                str(row["timepoint"]).replace("Day", "J"),
                (x_value, y_value), xytext=(5, 5), textcoords="offset points",
                fontsize=11,
            )
        intercept_text = (
            "" if method == "origin"
            else f"\nb = {float(fit['intercept_cps']) / 1000.0:.3f} kcps"
        )
        axis.text(
            0.96, 0.94,
            f"n = {len(cycle_rows)}\n"
            f"S = {float(fit['slope_cps_per_mbq']):.3f} cps/MBq\n"
            f"R² = {float(fit['r_squared']):.3f}"
            f"{intercept_text}\n"
            f"MARE = {float(fit['mare_activity_percent']):.1f} %",
            transform=axis.transAxes, ha="right", va="top", fontsize=11,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84},
        )
        axis.set_title(_cycle_label(cycle_rows[0]), fontsize=16, fontweight="bold")
        axis.set_xlabel("Activité Q/SPECT dans le crop (GBq)", fontsize=12)
        axis.set_ylabel("Taux de comptage du photopeak (kcps)", fontsize=12)
        axis.tick_params(axis="both", labelsize=11)
        axis.invert_xaxis()
        axis.grid(True, linestyle="--", alpha=0.24)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].legend(frameon=False, fontsize=10, loc="center right")
    figure.suptitle(
        f"Fit du photopeak avec tous les timepoints — {method_label}",
        fontsize=20, fontweight="bold",
    )
    return _save(figure, output_path)


def plot_mare_comparison(
    results: Sequence[Mapping[str, Any]], output_path: Path
) -> Path:
    figure, axes = plt.subplots(
        1, 3, figsize=(16.2, 6.2), facecolor="white", sharey=True,
        layout="constrained",
    )
    x = np.arange(len(DATASET_ORDER), dtype=float)
    width = 0.27
    for axis, (method, method_label) in zip(axes, METHODS):
        rows = [
            next(
                row for row in results
                if row["dataset_key"] == key and row["fit_method"] == method
            )
            for key in DATASET_ORDER
        ]
        three_on_all = [float(row["mare_first3_fit_on_all_percent"]) for row in rows]
        all_on_all = [float(row["mare_all_fit_on_all_percent"]) for row in rows]
        bars_three = axis.bar(
            x - width / 2.0, three_on_all, width,
            color="#9AA0A6", edgecolor="#222222", linewidth=0.8,
            label="Pente ajustée sur 3 points",
        )
        bars_all = axis.bar(
            x + width / 2.0, all_on_all, width,
            color=[slopes.cycle_color(row) for row in rows],
            edgecolor="#222222", linewidth=0.8,
            label="Pente ajustée sur tous les points",
        )
        axis.bar_label(bars_three, fmt="%.1f", padding=3, fontsize=10)
        axis.bar_label(bars_all, fmt="%.1f", padding=3, fontsize=10,
                       fontweight="bold")
        labels = [
            _cycle_label({"patient_id": row["patient_id"], "cycle_id": row["cycle_id"]})
            + f"\nn={int(row['number_of_all_points'])}"
            for row in rows
        ]
        axis.set_xticks(x, labels)
        axis.set_title(method_label, fontsize=16, fontweight="bold")
        axis.tick_params(axis="both", labelsize=10)
        axis.grid(axis="y", linestyle="--", alpha=0.24)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("MARE évaluée sur tous les points (%)", fontsize=14)
    axes[-1].legend(frameon=False, fontsize=10, loc="upper right")
    figure.suptitle(
        "Effet de l’ajout des timepoints sur la relation comptes–activité",
        fontsize=20, fontweight="bold",
    )
    return _save(figure, output_path)


def plot_fit_domain_comparison(
    results: Sequence[Mapping[str, Any]], output_path: Path
) -> Path:
    figure, axes = plt.subplots(
        1, 3, figsize=(16.2, 6.2), facecolor="white", sharey=True,
        layout="constrained",
    )
    x = np.arange(len(DATASET_ORDER), dtype=float)
    width = 0.27
    for axis, (method, method_label) in zip(axes, METHODS):
        rows = [
            next(
                row for row in results
                if row["dataset_key"] == key and row["fit_method"] == method
            )
            for key in DATASET_ORDER
        ]
        mare_three = [
            float(row["mare_first3_fit_on_first3_percent"]) for row in rows
        ]
        mare_all = [float(row["mare_all_fit_on_all_percent"]) for row in rows]
        bars_three = axis.bar(
            x - width / 2.0, mare_three, width, color="#9AA0A6",
            edgecolor="#222222", linewidth=0.8, label="Fit sur J0–J2",
        )
        bars_all = axis.bar(
            x + width / 2.0, mare_all, width,
            color=[slopes.cycle_color(row) for row in rows],
            edgecolor="#222222", linewidth=0.8, label="Fit sur tous les points",
        )
        axis.bar_label(bars_three, fmt="%.1f", padding=3, fontsize=10)
        axis.bar_label(bars_all, fmt="%.1f", padding=3, fontsize=10,
                       fontweight="bold")
        labels = [
            _cycle_label({"patient_id": row["patient_id"], "cycle_id": row["cycle_id"]})
            + f"\nn={int(row['number_of_all_points'])}"
            for row in rows
        ]
        axis.set_xticks(x, labels)
        axis.set_title(method_label, fontsize=16, fontweight="bold")
        axis.tick_params(axis="both", labelsize=10)
        axis.grid(axis="y", linestyle="--", alpha=0.24)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("MARE dans le domaine du fit (%)", fontsize=14)
    axes[-1].legend(frameon=False, fontsize=10, loc="upper right")
    figure.suptitle(
        "Qualité descriptive avec trois points ou tous les timepoints",
        fontsize=20, fontweight="bold",
    )
    return _save(figure, output_path)


def run(output_dir: Path = OUTPUT_DIR) -> dict[str, Path]:
    rows = slopes.prepare_qspect_reference_rows(
        slopes.load_master_table(slopes.DEFAULT_MASTER_TABLE)
    )
    grouped = _group_all(slopes.select_count_support(rows, "crop"))
    missing = [key for key in DATASET_ORDER if key not in grouped]
    if missing:
        raise ValueError("Cycles absents: " + ", ".join(missing))
    results = build_results(grouped)
    output_dir = Path(output_dir)
    outputs = {
        "origin": plot_all_point_fits(
            grouped, "origin", output_dir / "01_fit_origine_imposee.png"
        ),
        "free": plot_all_point_fits(
            grouped, "free_intercept", output_dir / "02_fit_intercept_libre.png"
        ),
        "anchor": plot_all_point_fits(
            grouped, "anchored_j0", output_dir / "03_fit_ancrage_J0.png"
        ),
        "same_domain": plot_fit_domain_comparison(
            results, output_dir / "04_qualite_fit_3_vs_tous_points.png"
        ),
        "late_points": plot_mare_comparison(
            results, output_dir / "05_prediction_tous_points_3_vs_tous.png"
        ),
        "results": _write_csv(output_dir / "resultats_tous_timepoints.csv", results),
    }
    return outputs


if __name__ == "__main__":
    for name, path in run().items():
        print(f"{name}: {path}")
