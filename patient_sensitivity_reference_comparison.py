"""Compare urine- and Q/SPECT-derived sensitivity fits for presentation.

The comparison uses the physically matched planar support for each target:
the complete planar image for whole-body urine retention and the Q/SPECT-
matched crop for reconstructed Q/SPECT activity. Both use global AP/PA
geometric mean, raw photopeak and the three J0--J2 points. The reported MARE is
descriptive because the same observations estimate and evaluate each slope.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import figure_layout
import patient_sensitivity_qspect_slopes as slopes


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = figure_layout.shared_dir(__file__)
DEFAULT_URINE_DIR = (
    PROJECT_DIR
    / "fig/models/patient_sensitivity_urine_target/2_sensibilite_J0_J2/complet"
)
DEFAULT_QSPECT_DIR = (
    PROJECT_DIR
    / "fig/models/patient_sensitivity_qspect_slopes/shared/2_sensibilite_J0_J2/crop"
)
COMBINATION = "GM_globale"
WINDOW = "photopeak"
TIMEPOINT_MARKERS = {"Day0": "o", "Day1": "s", "Day2": "^"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    if not rows:
        raise ValueError("No rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _selected_fit_rows(directory: Path) -> list[dict[str, str]]:
    return [
        row
        for row in _read_csv(directory / "pentes_fenetres_J0_J2.csv")
        if row["combination"] == COMBINATION and row["window"] == WINDOW
    ]


def _selected_prediction_rows(directory: Path) -> list[dict[str, str]]:
    return [
        row
        for row in _read_csv(directory / "predictions_intracycle.csv")
        if row["combination"] == COMBINATION and row["window"] == WINDOW
    ]


def _cycle_name(row: Mapping[str, Any]) -> str:
    patient = str(row["patient_id"]).upper()
    cycle = str(row["cycle_id"])
    month = cycle[5:7]
    month_names = {"05": "mai", "06": "juin", "09": "sept."}
    return f"{patient} — {month_names.get(month, month)}"


def prepare_comparison(
    urine_dir: Path = DEFAULT_URINE_DIR,
    qspect_dir: Path = DEFAULT_QSPECT_DIR,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    targets: dict[str, dict[str, Any]] = {}
    for key, label, support, directory in (
        ("urine", "Référence urinaire — image complète", "full", Path(urine_dir)),
        ("qspect", "Référence Q/SPECT — crop", "crop", Path(qspect_dir)),
    ):
        fits = _selected_fit_rows(directory)
        predictions = _selected_prediction_rows(directory)
        targets[key] = {
            "label": label,
            "support": support,
            "fits": fits,
            "predictions": predictions,
        }

    urine_keys = [row["dataset_key"] for row in targets["urine"]["fits"]]
    qspect_keys = {row["dataset_key"] for row in targets["qspect"]["fits"]}
    common_keys = [key for key in urine_keys if key in qspect_keys]
    if not common_keys:
        raise ValueError("No common patient-cycle between urine and Q/SPECT")

    for target in targets.values():
        target["fits"] = [
            row for row in target["fits"] if row["dataset_key"] in common_keys
        ]
        target["predictions"] = [
            row
            for row in target["predictions"]
            if row["dataset_key"] in common_keys
        ]
        target["mean_mare_percent"] = float(
            np.mean(
                [float(row["mare_activity_percent"]) for row in target["fits"]]
            )
        )
    return targets, common_keys


def plot_reference_comparison(
    targets: Mapping[str, Mapping[str, Any]],
    common_keys: Sequence[str],
    output_path: Path,
) -> Path:
    all_predictions = [
        row
        for target in targets.values()
        for row in target["predictions"]
    ]
    max_activity_gbq = max(
        float(row["reference_qspect_activity_mbq"]) / 1000.0
        for row in all_predictions
    )
    max_rate_kcps = max(
        float(row["dead_time_corrected_rate_cps"]) / 1000.0
        for row in all_predictions
    )
    x_limit = max_activity_gbq * 1.08
    y_limit = max_rate_kcps * 1.14

    fig, axes = plt.subplots(
        1, 2, figsize=(18.0, 8.6), sharex=True, sharey=True, facecolor="white"
    )
    for axis, target_key in zip(axes, ("urine", "qspect")):
        target = targets[target_key]
        fit_lookup = {row["dataset_key"]: row for row in target["fits"]}
        for dataset_key in common_keys:
            rows = [
                row
                for row in target["predictions"]
                if row["dataset_key"] == dataset_key
            ]
            rows.sort(key=lambda row: float(row["planar_day"]))
            if not rows:
                continue
            fit = fit_lookup[dataset_key]
            color = slopes.cycle_color(rows[0])
            x = np.asarray(
                [float(row["reference_qspect_activity_mbq"]) / 1000.0 for row in rows]
            )
            y = np.asarray(
                [float(row["dead_time_corrected_rate_cps"]) / 1000.0 for row in rows]
            )
            slope = float(fit["slope_cps_per_mbq"])
            line_x = np.linspace(0.0, x_limit, 250)
            axis.plot(line_x, slope * line_x, color=color, linewidth=2.4)
            for point_index, (row, x_value, y_value) in enumerate(
                zip(rows, x, y)
            ):
                axis.scatter(
                    [x_value],
                    [y_value],
                    marker=TIMEPOINT_MARKERS.get(str(row["timepoint"]), "o"),
                    s=96,
                    color=color,
                    edgecolor="white",
                    linewidth=1.1,
                    zorder=3,
                    label=(
                        f"{_cycle_name(rows[0])} — S = {slope:.3f}"
                        if point_index == 0
                        else None
                    ),
                )
        axis.scatter(
            [0.0], [0.0], marker="X", s=68, color="#222222", zorder=4
        )
        axis.set_xlim(x_limit, 0.0)
        axis.set_ylim(0.0, y_limit)
        axis.set_title(str(target["label"]), fontsize=21, fontweight="bold")
        axis.set_xlabel(
            "Activité de référence (GBq) — J0 à gauche", fontsize=15
        )
        axis.tick_params(axis="both", labelsize=13.5)
        axis.grid(True, linestyle="--", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
        cycle_legend = axis.legend(
            frameon=False, fontsize=11.5, loc="lower left"
        )
        axis.add_artist(cycle_legend)
        timepoint_handles = [
            Line2D(
                [0], [0], marker=marker, linestyle="none", color="#333333",
                markerfacecolor="#333333", markersize=8.5, label=label,
            )
            for label, marker in TIMEPOINT_MARKERS.items()
        ]
        axis.legend(
            handles=timepoint_handles,
            frameon=False,
            fontsize=11,
            loc="lower right",
            title="Timepoint",
            title_fontsize=11,
            ncol=3,
            handletextpad=0.3,
            columnspacing=0.8,
        )
        axis.text(
            0.04,
            0.95,
            f"MARE moyenne = {float(target['mean_mare_percent']):.1f} %",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=15,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.92},
        )
    axes[0].set_ylabel("Taux de comptage photopeak (kcps)", fontsize=15)
    fig.suptitle(
        "Comptes planaires et activité de référence\n"
        "Urine : image complète · Q/SPECT : crop · ajustement R = S·A sur J0–J2",
        fontsize=23,
        y=0.985,
    )
    fig.text(
        0.5,
        0.015,
        "MARE descriptive : les trois points servent à la fois à ajuster et à évaluer chaque pente.",
        ha="center",
        va="bottom",
        fontsize=12.5,
        color="#444444",
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.14,
        top=0.80,
        wspace=0.08,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_reference_bar_comparison(
    targets: Mapping[str, Mapping[str, Any]],
    common_keys: Sequence[str],
    output_path: Path,
) -> Path:
    """Compare slopes and descriptive errors with presentation-ready bars."""
    fit_lookups = {
        target_key: {
            row["dataset_key"]: row for row in targets[target_key]["fits"]
        }
        for target_key in ("urine", "qspect")
    }
    first_rows = [fit_lookups["urine"][key] for key in common_keys]
    cycle_labels = [_cycle_name(row) for row in first_rows]
    cycle_colors = [slopes.cycle_color(row) for row in first_rows]

    reference_keys = ("urine", "qspect")
    reference_labels = ("Urine\nimage complète", "Q/SPECT\ncrop")
    hatches = ("", "///")
    x_centers = np.arange(len(reference_keys), dtype=float)
    width = 0.22
    offsets = (
        np.arange(len(common_keys), dtype=float)
        - (len(common_keys) - 1) / 2.0
    ) * width

    fig, axes = plt.subplots(
        1, 2, figsize=(16.5, 7.8), facecolor="white",
        gridspec_kw={"width_ratios": (1.08, 1.0)},
    )

    sensitivity_axis, error_axis = axes
    for cycle_index, (dataset_key, cycle_label, color, offset) in enumerate(
        zip(common_keys, cycle_labels, cycle_colors, offsets)
    ):
        sensitivity_values = [
            float(fit_lookups[target_key][dataset_key]["slope_cps_per_mbq"])
            for target_key in reference_keys
        ]
        sensitivity_errors = [
            float(
                fit_lookups[target_key][dataset_key][
                    "slope_standard_error_cps_per_mbq"
                ]
            )
            for target_key in reference_keys
        ]
        bars = sensitivity_axis.bar(
            x_centers + offset,
            sensitivity_values,
            width=width * 0.92,
            yerr=sensitivity_errors,
            capsize=5,
            color=color,
            edgecolor="#222222",
            linewidth=0.9,
            error_kw={"ecolor": "#222222", "elinewidth": 1.4, "capthick": 1.4},
            label=cycle_label,
            zorder=3,
        )
        for target_index, (bar, value, standard_error) in enumerate(
            zip(bars, sensitivity_values, sensitivity_errors)
        ):
            bar.set_hatch(hatches[target_index])
            sensitivity_axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + standard_error + 0.018,
                slopes.format_value_with_uncertainty(value, standard_error),
                ha="center",
                va="bottom",
                fontsize=12.5,
                fontweight="bold",
            )

        mare_values = [
            float(fit_lookups[target_key][dataset_key]["mare_activity_percent"])
            for target_key in reference_keys
        ]
        bars = error_axis.bar(
            x_centers + offset,
            mare_values,
            width=width * 0.92,
            color=color,
            edgecolor="#222222",
            linewidth=0.9,
            zorder=3,
        )
        for target_index, (bar, value) in enumerate(zip(bars, mare_values)):
            bar.set_hatch(hatches[target_index])
            error_axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.55,
                f"{value:.1f} %",
                ha="center",
                va="bottom",
                fontsize=12.5,
                fontweight="bold",
            )

    for target_index, target_key in enumerate(reference_keys):
        error_axis.text(
            x_centers[target_index],
            27.2,
            f"Moyenne : {float(targets[target_key]['mean_mare_percent']):.1f} %",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color="#222222",
        )

    for axis in axes:
        axis.set_xticks(x_centers, reference_labels)
        axis.tick_params(axis="both", labelsize=14)
        axis.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(-0.58, 1.58)

    sensitivity_axis.set_ylim(0.0, 0.88)
    sensitivity_axis.set_ylabel("Sensibilité S (cps/MBq)", fontsize=16)
    sensitivity_axis.set_title(
        "Sensibilité estimée sur J0–J2 (S ± EC)",
        fontsize=19,
        fontweight="bold",
    )
    sensitivity_axis.legend(
        title="Patient / cycle",
        title_fontsize=13,
        fontsize=12.5,
        frameon=False,
        loc="lower left",
    )

    error_axis.set_ylim(0.0, 30.5)
    error_axis.set_ylabel("Erreur relative moyenne — MARE (%)", fontsize=16)
    error_axis.set_title(
        "Écart aux trois activités de référence", fontsize=19, fontweight="bold"
    )

    fig.suptitle(
        "Supports planaires adaptés aux deux activités de référence",
        fontsize=23,
        fontweight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.04,
        "Photopeak brut · urine : image complète · Q/SPECT : crop · moyenne géométrique globale\n"
        "Barres d’erreur = erreur standard de la pente; MARE descriptive sur les trois points ajustés",
        ha="center",
        fontsize=13.5,
        color="#333333",
    )
    fig.subplots_adjust(
        left=0.075, right=0.985, bottom=0.18, top=0.82, wspace=0.24
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _bar_comparison_data(
    targets: Mapping[str, Mapping[str, Any]],
    common_keys: Sequence[str],
) -> tuple[dict[str, dict[str, Mapping[str, Any]]], list[str], list[str]]:
    fit_lookups = {
        target_key: {
            str(row["dataset_key"]): row for row in targets[target_key]["fits"]
        }
        for target_key in ("urine", "qspect")
    }
    first_rows = [fit_lookups["urine"][key] for key in common_keys]
    cycle_labels = [_cycle_name(row) for row in first_rows]
    cycle_colors = [slopes.cycle_color(row) for row in first_rows]
    return fit_lookups, cycle_labels, cycle_colors


def plot_single_target_sensitivity_bars(
    targets: Mapping[str, Mapping[str, Any]],
    common_keys: Sequence[str],
    target_key: str,
    title: str,
    output_path: Path,
) -> Path:
    """Plot one target with the same visual specification as the other target."""
    fit_lookups, cycle_labels, cycle_colors = _bar_comparison_data(
        targets, common_keys
    )
    rows = [fit_lookups[target_key][dataset_key] for dataset_key in common_keys]
    values = np.asarray(
        [float(row["slope_cps_per_mbq"]) for row in rows], dtype=float
    )
    errors = np.asarray(
        [float(row["slope_standard_error_cps_per_mbq"]) for row in rows],
        dtype=float,
    )
    positions = np.arange(len(rows), dtype=float)

    fig, axis = plt.subplots(figsize=(10.0, 7.2), facecolor="white")
    bars = axis.bar(
        positions,
        values,
        width=0.62,
        yerr=errors,
        capsize=6,
        color=cycle_colors,
        edgecolor="#222222",
        linewidth=0.9,
        error_kw={"ecolor": "#222222", "elinewidth": 1.5, "capthick": 1.5},
        zorder=3,
    )
    for bar, value, standard_error in zip(bars, values, errors):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + standard_error + 0.018,
            slopes.format_value_with_uncertainty(value, standard_error),
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )

    axis.set_xticks(
        positions,
        [label.replace(" — ", "\n") for label in cycle_labels],
    )
    axis.set_xlim(-0.65, len(rows) - 0.35)
    axis.set_ylim(0.0, 0.88)
    axis.set_ylabel("Sensibilité S (cps/MBq)", fontsize=18)
    axis.set_title(title, fontsize=24, fontweight="bold", pad=20)
    axis.tick_params(axis="both", labelsize=16)
    axis.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.13, top=0.87)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_sensitivity_bar_comparison(
    targets: Mapping[str, Mapping[str, Any]],
    common_keys: Sequence[str],
    output_path: Path,
) -> Path:
    """Plot only the fitted sensitivities for a clean presentation slide."""
    fit_lookups, cycle_labels, cycle_colors = _bar_comparison_data(
        targets, common_keys
    )
    reference_keys = ("urine", "qspect")
    reference_labels = ("Urine\nimage complète", "Q/SPECT\ncrop")
    hatches = ("", "///")
    x_centers = np.arange(len(reference_keys), dtype=float)
    width = 0.22
    offsets = (
        np.arange(len(common_keys), dtype=float)
        - (len(common_keys) - 1) / 2.0
    ) * width

    fig, axis = plt.subplots(figsize=(10.8, 7.8), facecolor="white")
    for dataset_key, cycle_label, color, offset in zip(
        common_keys, cycle_labels, cycle_colors, offsets
    ):
        values = [
            float(fit_lookups[target_key][dataset_key]["slope_cps_per_mbq"])
            for target_key in reference_keys
        ]
        errors = [
            float(
                fit_lookups[target_key][dataset_key][
                    "slope_standard_error_cps_per_mbq"
                ]
            )
            for target_key in reference_keys
        ]
        bars = axis.bar(
            x_centers + offset,
            values,
            width=width * 0.92,
            yerr=errors,
            capsize=6,
            color=color,
            edgecolor="#222222",
            linewidth=0.9,
            error_kw={"ecolor": "#222222", "elinewidth": 1.5, "capthick": 1.5},
            label=cycle_label,
            zorder=3,
        )
        for target_index, (bar, value, standard_error) in enumerate(
            zip(bars, values, errors)
        ):
            bar.set_hatch(hatches[target_index])
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + standard_error + 0.018,
                slopes.format_value_with_uncertainty(value, standard_error),
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
            )

    axis.set_xticks(x_centers, reference_labels)
    axis.set_xlim(-0.58, 1.58)
    axis.set_ylim(0.0, 0.88)
    axis.set_ylabel("Sensibilité S (cps/MBq)", fontsize=17)
    axis.set_title(
        "Sensibilité estimée sur J0–J2 (S ± EC)",
        fontsize=22,
        fontweight="bold",
        pad=18,
    )
    axis.tick_params(axis="both", labelsize=16)
    axis.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        title="Patient / cycle",
        title_fontsize=14,
        fontsize=13.5,
        frameon=False,
        loc="lower left",
    )
    fig.text(
        0.5,
        0.035,
        "Photopeak brut · urine : image complète · Q/SPECT : crop · moyenne géométrique globale\n"
        "Barres d’erreur = erreur standard de la pente imposée à l’origine",
        ha="center",
        fontsize=13,
        color="#333333",
    )
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.18, top=0.88)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_mare_bar_comparison(
    targets: Mapping[str, Mapping[str, Any]],
    common_keys: Sequence[str],
    output_path: Path,
) -> Path:
    """Plot only descriptive MARE values for a clean presentation slide."""
    fit_lookups, cycle_labels, cycle_colors = _bar_comparison_data(
        targets, common_keys
    )
    reference_keys = ("urine", "qspect")
    reference_labels = ("Urine\nimage complète", "Q/SPECT\ncrop")
    hatches = ("", "///")
    x_centers = np.arange(len(reference_keys), dtype=float)
    width = 0.22
    offsets = (
        np.arange(len(common_keys), dtype=float)
        - (len(common_keys) - 1) / 2.0
    ) * width

    fig, axis = plt.subplots(figsize=(10.8, 7.8), facecolor="white")
    for dataset_key, cycle_label, color, offset in zip(
        common_keys, cycle_labels, cycle_colors, offsets
    ):
        values = [
            float(fit_lookups[target_key][dataset_key]["mare_activity_percent"])
            for target_key in reference_keys
        ]
        bars = axis.bar(
            x_centers + offset,
            values,
            width=width * 0.92,
            color=color,
            edgecolor="#222222",
            linewidth=0.9,
            label=cycle_label,
            zorder=3,
        )
        for target_index, (bar, value) in enumerate(zip(bars, values)):
            bar.set_hatch(hatches[target_index])
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.55,
                f"{value:.1f} %",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
            )

    for target_index, target_key in enumerate(reference_keys):
        axis.text(
            x_centers[target_index],
            27.2,
            f"Moyenne : {float(targets[target_key]['mean_mare_percent']):.1f} %",
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )

    axis.set_xticks(x_centers, reference_labels)
    axis.set_xlim(-0.58, 1.58)
    axis.set_ylim(0.0, 30.5)
    axis.set_ylabel("Erreur relative moyenne — MARE (%)", fontsize=17)
    axis.set_title(
        "Écart aux trois activités de référence",
        fontsize=22,
        fontweight="bold",
        pad=18,
    )
    axis.tick_params(axis="both", labelsize=16)
    axis.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        title="Patient / cycle",
        title_fontsize=14,
        fontsize=13.5,
        frameon=False,
        loc="center right",
    )
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.11, top=0.88)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def run_analysis(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    urine_dir: Path = DEFAULT_URINE_DIR,
    qspect_dir: Path = DEFAULT_QSPECT_DIR,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    targets, common_keys = prepare_comparison(urine_dir, qspect_dir)
    figure_path = plot_reference_comparison(
        targets,
        common_keys,
        output_dir / "urine_vs_qspect_photopeak_crop_gm.png",
    )
    sensitivity_bar_path = plot_sensitivity_bar_comparison(
        targets,
        common_keys,
        output_dir / "urine_vs_qspect_sensitivity_bars.png",
    )
    mare_bar_path = plot_mare_bar_comparison(
        targets,
        common_keys,
        output_dir / "urine_vs_qspect_mare_bars.png",
    )
    urine_bar_path = plot_single_target_sensitivity_bars(
        targets,
        common_keys,
        "urine",
        "Sensibilité planaire — urine (image complète)",
        output_dir / "urine_full_image_sensitivity_bars.png",
    )
    qspect_bar_path = plot_single_target_sensitivity_bars(
        targets,
        common_keys,
        "qspect",
        "Sensibilité planaire — Q/SPECT (crop)",
        output_dir / "qspect_crop_sensitivity_bars.png",
    )
    summary_rows: list[dict[str, Any]] = []
    for target_key in ("urine", "qspect"):
        target = targets[target_key]
        for fit in target["fits"]:
            summary_rows.append(
                {
                    "target": target_key,
                    "target_label": target["label"],
                    "dataset_key": fit["dataset_key"],
                    "dataset_label": fit["dataset_label"],
                    "support": target["support"],
                    "combination": COMBINATION,
                    "window": WINDOW,
                    "number_of_points": fit["number_of_points"],
                    "slope_cps_per_mbq": fit["slope_cps_per_mbq"],
                    "slope_standard_error_cps_per_mbq": fit[
                        "slope_standard_error_cps_per_mbq"
                    ],
                    "r_squared_zero_origin": fit["r_squared_zero_origin"],
                    "mare_activity_percent": fit["mare_activity_percent"],
                    "target_mean_mare_percent": target["mean_mare_percent"],
                }
            )
    csv_path = _write_csv(output_dir / "urine_vs_qspect_summary.csv", summary_rows)
    report_path = output_dir / "README.txt"
    report_path.write_text(
        """COMPARAISON URINE VERSUS Q/SPECT
=================================

Support planaire adapte a chaque activite de reference:
- urine: image planaire complete, correspondant a la retention corporelle;
- Q/SPECT: crop correspondant a la couverture reconstruite;
- moyenne geometrique globale AP/PA;
- photopeak brut;
- correction globale du temps mort;
- pente a l'origine R = S*A ajustee sur J0--J2.

La comparaison utilise seulement les cycles communs P11-mai, P8-juin et
P17-septembre. La MARE est descriptive: les memes trois points servent a
ajuster et a evaluer chaque pente.
""",
        encoding="utf-8",
    )
    return {
        "figure": figure_path,
        "sensitivity_bar_figure": sensitivity_bar_path,
        "mare_bar_figure": mare_bar_path,
        "urine_bar_figure": urine_bar_path,
        "qspect_bar_figure": qspect_bar_path,
        "summary": csv_path,
        "readme": report_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--urine-dir", type=Path, default=DEFAULT_URINE_DIR)
    parser.add_argument("--qspect-dir", type=Path, default=DEFAULT_QSPECT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = run_analysis(args.output_dir, args.urine_dir, args.qspect_dir)
    print(f"Saved comparison figure to {outputs['figure']}")
    print(
        "Saved sensitivity bar figure to "
        f"{outputs['sensitivity_bar_figure']}"
    )
    print(f"Saved MARE bar figure to {outputs['mare_bar_figure']}")
    print(f"Saved urine bar figure to {outputs['urine_bar_figure']}")
    print(f"Saved Q/SPECT bar figure to {outputs['qspect_bar_figure']}")


if __name__ == "__main__":
    main()
