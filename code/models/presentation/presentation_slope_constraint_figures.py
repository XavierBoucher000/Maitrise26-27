"""Figures de presentation pour comparer les contraintes de pente.

Toutes les figures utilisent le photopeak brut et la moyenne geometrique
globale. L'urine emploie l'image complete; le Q/SPECT emploie le crop apparie.
"""

from __future__ import annotations

import csv
from pathlib import Path

from core.project_paths import PROJECT_ROOT
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

import models.patient_sensitivity_qspect_slopes as slopes
import models.legacy.patient_sensitivity_urine_target as urine


PROJECT_DIR = PROJECT_ROOT
OUTPUT_DIR = PROJECT_DIR / "fig/models/presentation_summary"
SELECTED_OUTPUT_DIR = OUTPUT_DIR / "intercept_libre"
SELECTED_J0_OUTPUT_DIR = OUTPUT_DIR / "ancrage_J0"
URINE_DECAY_DETAIL_DIR = OUTPUT_DIR / "11_detail_decroissance_par_recolte"
COMBINATION = "GM_globale"
WINDOW = "photopeak"
COMMON_KEYS = (
    "p11_2026_05_studies",
    "p8_2026_06_studies",
    "p17_2026_09_studies",
)

REFERENCES = {
    "urine": {
        "label": "Urine — image complète",
        "activity_label": "Activité corporelle estimée par bilan urinaire (GBq)",
        "root": PROJECT_DIR / "fig/models/patient_sensitivity_urine_target",
        "support": "complet",
    },
    "qspect": {
        "label": "Q/SPECT — crop",
        "activity_label": "Activité Q/SPECT dans le crop (GBq)",
        "root": PROJECT_DIR / "fig/models/patient_sensitivity_qspect_slopes/shared",
        "support": "crop",
    },
}

METHODS = (
    ("origin", "Origine imposée", "2_sensibilite_J0_J2", "", 1.0),
    ("free_intercept", "Intercept libre", "5_pente_intercept_libre", "//", 0.72),
    ("anchored_j0", "Ancrage à J0", "6_pente_ancree_J0", "xx", 0.50),
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


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


def _method_dir(reference: str, section: str) -> Path:
    spec = REFERENCES[reference]
    return Path(spec["root"]) / section / str(spec["support"])


def _cycle_label(row: Mapping[str, Any]) -> str:
    patient = str(row["patient_id"]).upper()
    month = str(row["cycle_id"])[5:7]
    month_label = {"05": "mai", "06": "juin", "09": "sept."}.get(month, month)
    return f"{patient}\n{month_label}"


def _load_fit_rows(reference: str, section: str) -> list[dict[str, str]]:
    return [
        row
        for row in _read_csv(
            _method_dir(reference, section) / "pentes_fenetres_J0_J2.csv"
        )
        if row["combination"] == COMBINATION
        and row["window"] == WINDOW
        and row["dataset_key"] in COMMON_KEYS
    ]


def _load_prediction_rows(reference: str, section: str) -> list[dict[str, str]]:
    return [
        row
        for row in _read_csv(
            _method_dir(reference, section) / "predictions_intracycle.csv"
        )
        if row["combination"] == COMBINATION
        and row["window"] == WINDOW
        and row["dataset_key"] in COMMON_KEYS
    ]


def _load_j0_mare(reference: str) -> dict[str, float]:
    rows = [
        row
        for row in _read_csv(
            _method_dir(reference, "1_sensibilite_J0") / "predictions_post_J0.csv"
        )
        if row["combination"] == COMBINATION
        and row["window"] == WINDOW
        and row["dataset_key"] in COMMON_KEYS
    ]
    return {
        key: float(
            np.mean(
                [
                    abs(float(row["relative_error_percent"]))
                    for row in rows
                    if row["dataset_key"] == key
                ]
            )
        )
        for key in COMMON_KEYS
    }


def plot_urine_relation(
    method: str,
    output_path: Path,
    *,
    include_method_in_title: bool = True,
) -> Path:
    method_spec = next(item for item in METHODS if item[0] == method)
    _, method_label, section, _, _ = method_spec
    fits = _load_fit_rows("urine", section)
    predictions = _load_prediction_rows("urine", section)
    fit_lookup = {row["dataset_key"]: row for row in fits}

    figure, axes = plt.subplots(
        1, 3, figsize=(17.0, 6.1), facecolor="white", layout="constrained"
    )
    for axis, dataset_key in zip(axes, COMMON_KEYS):
        rows = [row for row in predictions if row["dataset_key"] == dataset_key]
        fit = fit_lookup[dataset_key]
        activity = np.asarray(
            [float(row["reference_activity_mbq"]) for row in rows]
        ) / 1000.0
        rate = np.asarray(
            [float(row["dead_time_corrected_rate_cps"]) for row in rows]
        ) / 1000.0
        slope = float(fit["slope_cps_per_mbq"])
        intercept_kcps = float(fit.get("intercept_cps", 0.0)) / 1000.0
        color = slopes.cycle_color(fit)
        x_line = np.linspace(0.0, 1.04 * float(np.max(activity)), 160)
        axis.plot(x_line, slope * x_line + intercept_kcps, color=color, linewidth=2.5)
        axis.scatter(
            activity, rate, s=86, color=color, edgecolor="white", linewidth=1.0,
            zorder=3,
        )
        if method == "origin":
            axis.scatter([0.0], [0.0], marker="X", s=68, color="#222222", zorder=4)
        elif method == "anchored_j0":
            axis.scatter(
                [activity[0]], [rate[0]], marker="D", s=122, facecolor="none",
                edgecolor="#222222", linewidth=1.6, zorder=4,
            )
        for row, x_value, y_value in zip(rows, activity, rate):
            axis.annotate(
                str(row["timepoint"]).replace("Day", "J"),
                (x_value, y_value), xytext=(6, 5), textcoords="offset points",
                fontsize=13,
            )
        intercept_text = "" if method == "origin" else f"\nb = {intercept_kcps:.3f} kcps"
        axis.text(
            0.96, 0.94,
            f"S = {slope:.3f} cps/MBq\n"
            f"R² = {float(fit['r_squared']):.3f}"
            f"{intercept_text}\n"
            f"MARE = {float(fit['mare_activity_percent']):.1f} %",
            transform=axis.transAxes, ha="right", va="top", fontsize=13,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84},
        )
        axis.set_title(_cycle_label(fit), fontsize=17, fontweight="bold")
        axis.set_xlabel(str(REFERENCES["urine"]["activity_label"]), fontsize=14)
        axis.set_ylabel("Taux de comptage du photopeak (kcps)", fontsize=14)
        axis.tick_params(axis="both", labelsize=13)
        axis.grid(True, linestyle="--", alpha=0.24)
        axis.spines[["top", "right"]].set_visible(False)
    title = "Relation entre l’activité urinaire et le taux de comptage du photopeak"
    if include_method_in_title:
        title += f" — {method_label}"
    figure.suptitle(
        title,
        fontsize=21, fontweight="bold",
    )
    return _save(figure, output_path)


def plot_sensitivity_comparison(reference: str, output_path: Path) -> Path:
    fit_by_method = {
        method: {row["dataset_key"]: row for row in _load_fit_rows(reference, section)}
        for method, _, section, _, _ in METHODS
    }
    x = np.arange(len(COMMON_KEYS), dtype=float)
    width = 0.23
    figure, axis = plt.subplots(figsize=(12.4, 7.0), facecolor="white")
    for method_index, (method, method_label, _, hatch, alpha) in enumerate(METHODS):
        offset = (method_index - 1) * width
        for patient_index, dataset_key in enumerate(COMMON_KEYS):
            row = fit_by_method[method][dataset_key]
            value = float(row["slope_cps_per_mbq"])
            error = float(row["slope_standard_error_cps_per_mbq"])
            bar = axis.bar(
                x[patient_index] + offset, value, width=width * 0.88,
                color=slopes.cycle_color(row), alpha=alpha, hatch=hatch,
                edgecolor="#222222", linewidth=1.0, yerr=error, capsize=5,
                error_kw={"elinewidth": 1.5, "capthick": 1.5}, zorder=3,
            )[0]
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + error + 0.018,
                f"{value:.2f}", ha="center", va="bottom", fontsize=12,
                fontweight="bold",
            )
    labels = [
        _cycle_label(fit_by_method["origin"][key]) for key in COMMON_KEYS
    ]
    axis.set_xticks(x, labels)
    axis.set_ylabel("Pente S (cps/MBq)", fontsize=16)
    axis.set_title(
        f"Sensibilité planaire selon la contrainte du fit — {REFERENCES[reference]['label']}",
        fontsize=20, fontweight="bold", pad=16,
    )
    axis.tick_params(axis="both", labelsize=14)
    axis.grid(axis="y", linestyle="--", alpha=0.24, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_ylim(0.0, 1.05)
    handles = [
        Patch(facecolor="#777777", edgecolor="#222222", hatch=hatch,
              alpha=alpha, label=label)
        for _, label, _, hatch, alpha in METHODS
    ]
    axis.legend(handles=handles, title="Contrainte", frameon=False,
                fontsize=12, title_fontsize=13, ncol=3, loc="upper right")
    figure.tight_layout()
    return _save(figure, output_path)


def plot_single_method_sensitivity(
    reference: str,
    method: str,
    output_path: Path,
) -> Path:
    method_spec = next(item for item in METHODS if item[0] == method)
    _, _, section, _, _ = method_spec
    rows = _load_fit_rows(reference, section)
    row_lookup = {row["dataset_key"]: row for row in rows}
    ordered_rows = [row_lookup[key] for key in COMMON_KEYS]
    values = np.asarray(
        [float(row["slope_cps_per_mbq"]) for row in ordered_rows]
    )
    errors = np.asarray(
        [float(row["slope_standard_error_cps_per_mbq"]) for row in ordered_rows]
    )
    x = np.arange(len(ordered_rows), dtype=float)
    figure, axis = plt.subplots(figsize=(10.8, 7.2), facecolor="white")
    bars = axis.bar(
        x, values, yerr=errors, capsize=6, width=0.58,
        color=[slopes.cycle_color(row) for row in ordered_rows],
        edgecolor="#222222", linewidth=1.0,
        error_kw={"elinewidth": 1.6, "capthick": 1.6}, zorder=3,
    )
    for bar, value, error in zip(bars, values, errors):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + error + 0.022,
            slopes.format_value_with_uncertainty(value, error),
            ha="center", va="bottom", fontsize=16, fontweight="bold",
        )
    axis.set_xticks(x, [_cycle_label(row) for row in ordered_rows])
    axis.set_ylabel("Sensibilité S (cps/MBq)", fontsize=17)
    axis.set_title(
        f"Sensibilité planaire — {REFERENCES[reference]['label']}",
        fontsize=22, fontweight="bold", pad=16,
    )
    axis.tick_params(axis="both", labelsize=15)
    axis.grid(axis="y", linestyle="--", alpha=0.24, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_ylim(0.0, max(1.0, float(np.max(values + errors)) * 1.22))
    figure.tight_layout()
    return _save(figure, output_path)


def _fit_mare_lookup(reference: str) -> dict[str, dict[str, float]]:
    return {
        method: {
            row["dataset_key"]: float(row["mare_activity_percent"])
            for row in _load_fit_rows(reference, section)
        }
        for method, _, section, _, _ in METHODS
    }


def _plot_error_panels(
    values_by_reference: Mapping[str, Mapping[str, Mapping[str, float]]],
    methods: Sequence[tuple[str, str]],
    output_path: Path,
    title: str,
    ylabel: str,
) -> Path:
    maximum_value = max(
        float(values_by_reference[reference][method][dataset_key])
        for reference in ("urine", "qspect")
        for method, _ in methods
        for dataset_key in COMMON_KEYS
    )
    y_max = max(10.0, maximum_value * 1.24)
    figure, axes = plt.subplots(
        1, 2, figsize=(15.5, 7.0), facecolor="white", sharey=True,
        layout="constrained",
    )
    width = 0.23
    for axis, reference in zip(axes, ("urine", "qspect")):
        axis.set_ylim(0.0, y_max)
        x = np.arange(len(methods), dtype=float)
        for patient_index, dataset_key in enumerate(COMMON_KEYS):
            offset = (patient_index - 1) * width
            values = [
                float(values_by_reference[reference][method][dataset_key])
                for method, _ in methods
            ]
            origin_row = next(
                row for row in _load_fit_rows(reference, "2_sensibilite_J0_J2")
                if row["dataset_key"] == dataset_key
            )
            bars = axis.bar(
                x + offset, values, width=width * 0.88,
                color=slopes.cycle_color(origin_row), edgecolor="#222222",
                linewidth=0.9, label=_cycle_label(origin_row).replace("\n", " — "),
                zorder=3,
            )
            axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=11,
                           fontweight="bold")
        for method_index, (method, _) in enumerate(methods):
            mean_value = float(
                np.mean(
                    [values_by_reference[reference][method][key] for key in COMMON_KEYS]
                )
            )
            axis.text(
                method_index, y_max * 0.96,
                f"Moy. {mean_value:.1f} %", ha="center", va="top",
                fontsize=12.5, fontweight="bold",
            )
        axis.set_xticks(x, [label for _, label in methods])
        axis.set_title(str(REFERENCES[reference]["label"]), fontsize=18,
                       fontweight="bold")
        axis.tick_params(axis="both", labelsize=13)
        axis.grid(axis="y", linestyle="--", alpha=0.24, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(ylabel, fontsize=16)
    axes[1].legend(title="Patient / cycle", frameon=False, fontsize=12,
                   title_fontsize=13, loc="center right")
    figure.suptitle(title, fontsize=21, fontweight="bold")
    return _save(figure, output_path)


def plot_intracycle_error(output_path: Path) -> Path:
    values = {reference: _fit_mare_lookup(reference) for reference in REFERENCES}
    methods = tuple((method, label) for method, label, _, _, _ in METHODS)
    return _plot_error_panels(
        values, methods, output_path,
        "Erreur descriptive selon la contrainte du fit",
        "MARE sur J0–J2 (%)",
    )


def plot_j0_and_fit_error(output_path: Path) -> Path:
    values: dict[str, dict[str, dict[str, float]]] = {}
    for reference in REFERENCES:
        values[reference] = {"j0_only": _load_j0_mare(reference)}
        values[reference].update(_fit_mare_lookup(reference))
    methods = (
        ("j0_only", "Calibration\nà J0"),
        ("origin", "Origine\nimposée"),
        ("free_intercept", "Intercept\nlibre"),
        ("anchored_j0", "Ancrage\nà J0"),
    )
    return _plot_error_panels(
        values, methods, output_path,
        "Erreur sur J0–J2 selon la méthode de calibration",
        "MARE sur J0–J2 (%)",
    )


def plot_single_method_reference_error(method: str, output_path: Path) -> Path:
    method_spec = next(item for item in METHODS if item[0] == method)
    method_key = method_spec[0]
    values = {
        reference: _fit_mare_lookup(reference)[method_key]
        for reference in REFERENCES
    }
    centers = np.asarray([0.0, 1.25])
    width = 0.22
    figure, axis = plt.subplots(figsize=(11.6, 7.2), facecolor="white")
    origin_lookup = {
        row["dataset_key"]: row
        for row in _load_fit_rows("urine", "2_sensibilite_J0_J2")
    }
    for patient_index, dataset_key in enumerate(COMMON_KEYS):
        offset = (patient_index - 1) * width
        patient_values = [
            values["urine"][dataset_key], values["qspect"][dataset_key]
        ]
        bars = axis.bar(
            centers + offset, patient_values, width=width * 0.92,
            color=slopes.cycle_color(origin_lookup[dataset_key]),
            edgecolor="#222222", linewidth=0.9,
            label=_cycle_label(origin_lookup[dataset_key]).replace("\n", " — "),
            zorder=3,
        )
        bars[1].set_hatch("///")
        axis.bar_label(
            bars, labels=[f"{value:.1f} %" for value in patient_values],
            padding=4, fontsize=14, fontweight="bold",
        )
    maximum = max(
        value
        for reference_values in values.values()
        for value in reference_values.values()
    )
    y_max = max(7.0, maximum * 1.42)
    axis.set_ylim(0.0, y_max)
    for center, reference in zip(centers, ("urine", "qspect")):
        mean_value = float(np.mean(list(values[reference].values())))
        axis.text(
            center, y_max * 0.93, f"Moyenne : {mean_value:.1f} %",
            ha="center", va="top", fontsize=16, fontweight="bold",
        )
    axis.set_xticks(
        centers,
        ["Urine\nimage complète", "Q/SPECT\ncrop"],
    )
    axis.set_ylabel("Erreur relative moyenne — MARE (%)", fontsize=17)
    axis.set_title(
        "Écart aux trois activités de référence",
        fontsize=22, fontweight="bold", pad=16,
    )
    axis.tick_params(axis="both", labelsize=15)
    axis.grid(axis="y", linestyle="--", alpha=0.24, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        title="Patient / cycle", frameon=False, fontsize=13,
        title_fontsize=14, loc="center right",
    )
    figure.tight_layout()
    return _save(figure, output_path)


def build_urine_decay_validation_rows() -> list[dict[str, Any]]:
    """Return the exact same-time urine balance used by the sensitivity fits.

    Each measured urine/diaper interval is supplied in the source spreadsheet
    at its sampling time.  The urine pipeline first converts every completed
    interval to its injection-time equivalent, then decays the cumulative
    amount forward to the planar acquisition time.  Only collections completed
    before an image are included in that image's balance.
    """
    output: list[dict[str, Any]] = []
    for spec in urine.URINE_DATASETS:
        dataset = urine.build_urine_dataset(spec, timepoint_count=3)
        interval_activity = np.asarray(
            dataset["urine_balance"]["interval_excreted_mbq"], dtype=float
        )
        collection_times = np.asarray(
            dataset["urine_balance"]["times_h"], dtype=float
        )
        for record in dataset["records"]:
            imaging_h = float(record["hours_after_injection"])
            physical = float(record["physical_available_activity_mbq"])
            excreted = float(record["cumulative_excreted_activity_mbq"])
            remaining = float(record["urine_remaining_activity_mbq"])
            completed_mask = collection_times <= imaging_h
            completed = int(np.count_nonzero(completed_mask))
            activity_at_collection = float(interval_activity[completed_mask].sum())
            if completed:
                activity_at_imaging = float(
                    np.sum(
                        interval_activity[completed_mask]
                        * urine.sang_decay_factor(
                            imaging_h - collection_times[completed_mask],
                            float(record["physical_half_life_h"]),
                        )
                    )
                )
                collection_time_range = (
                    f"{collection_times[completed_mask][0]:.1f}–"
                    f"{collection_times[completed_mask][-1]:.1f}"
                )
            else:
                activity_at_imaging = 0.0
                collection_time_range = "—"
            decay_difference = activity_at_collection - activity_at_imaging
            if not np.isclose(activity_at_imaging, excreted, rtol=1e-10, atol=1e-8):
                raise ValueError(
                    f"Incohérence de correction temporelle pour {spec.label} "
                    f"{record['timepoint']}: {activity_at_imaging} vs {excreted} MBq"
                )
            future = collection_times[collection_times > imaging_h]
            output.append(
                {
                    "dataset_key": f"{spec.patient_id}_{spec.cycle_id.lower()}",
                    "patient_id": spec.patient_id,
                    "cycle_id": spec.cycle_id,
                    "dataset_label": spec.label,
                    "timepoint": str(record["timepoint"]),
                    "planar_datetime": record["planar_datetime"].isoformat(
                        sep=" "
                    ),
                    "hours_after_injection": imaging_h,
                    "physical_half_life_h": float(
                        record["physical_half_life_h"]
                    ),
                    "completed_collections": completed,
                    "total_collections": int(collection_times.size),
                    "included_collection_time_range_h": collection_time_range,
                    "next_collection_h": (
                        float(future[0]) if future.size else float("nan")
                    ),
                    "summed_activity_at_collection_times_mbq": activity_at_collection,
                    "same_activity_at_imaging_time_mbq": activity_at_imaging,
                    "physical_decay_difference_mbq": decay_difference,
                    "physical_decay_difference_percent": (
                        100.0 * decay_difference / activity_at_collection
                        if activity_at_collection > 0.0
                        else float("nan")
                    ),
                    "injected_activity_after_decay_mbq": physical,
                    "cumulative_excreted_at_imaging_mbq": excreted,
                    "remaining_body_activity_mbq": remaining,
                    "closure_error_mbq": remaining + excreted - physical,
                    "closure_relative_error_percent": (
                        100.0 * (remaining + excreted - physical) / physical
                    ),
                }
            )
    return output


def plot_urine_decay_validation(
    rows: Sequence[Mapping[str, Any]], output_path: Path
) -> Path:
    """Tabulate the decay applied between collection and planar imaging."""
    ordered_labels = [
        "P11 — mai 2026",
        "P8 — juin 2026",
        "P17 — septembre 2026",
    ]
    figure, axes = plt.subplots(
        3, 1, figsize=(17.0, 10.2), facecolor="white", layout="constrained"
    )
    headers = (
        "Imagerie",
        "Temps image\n(h)",
        "Collectes\nincluses",
        "Plage des récoltes incluses\n(h après injection)",
        "Activité cumulée aux heures\nde récolte (MBq)*",
        "Même activité à l’heure\nde l’imagerie (MBq)",
        "Écart dû à la\ndécroissance (MBq)",
        "Écart\n(%)",
    )
    for axis, dataset_label in zip(axes, ordered_labels):
        selected = [
            dict(row) for row in rows if row["dataset_label"] == dataset_label
        ]
        selected.sort(key=lambda row: float(row["hours_after_injection"]))
        if not selected:
            raise ValueError(f"Aucune donnée de décroissance pour {dataset_label}")

        color = slopes.cycle_color(selected[0])
        table_rows = []
        for row in selected:
            difference_percent = float(row["physical_decay_difference_percent"])
            table_rows.append(
                (
                    str(row["timepoint"]).replace("Day", "J"),
                    f"{float(row['hours_after_injection']):.1f}",
                    f"{int(row['completed_collections'])}/{int(row['total_collections'])}",
                    str(row["included_collection_time_range_h"]),
                    f"{float(row['summed_activity_at_collection_times_mbq']):,.0f}".replace(",", " "),
                    f"{float(row['same_activity_at_imaging_time_mbq']):,.0f}".replace(",", " "),
                    f"{float(row['physical_decay_difference_mbq']):,.0f}".replace(",", " "),
                    f"{difference_percent:.1f}" if np.isfinite(difference_percent) else "—",
                )
            )

        axis.axis("off")
        axis.set_title(dataset_label, fontsize=18, fontweight="bold", pad=8)
        table = axis.table(
            cellText=table_rows,
            colLabels=headers,
            cellLoc="center",
            colLoc="center",
            loc="center",
            colWidths=(0.07, 0.09, 0.09, 0.15, 0.16, 0.17, 0.15, 0.07),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12.5)
        table.scale(1.0, 1.68)
        for (row_index, _), cell in table.get_celld().items():
            cell.set_edgecolor("#D0D0D0")
            cell.set_linewidth(0.8)
            if row_index == 0:
                cell.set_height(cell.get_height() * 1.45)
                cell.set_facecolor(color)
                cell.set_text_props(color="white", fontweight="bold", fontsize=11.5)
            elif row_index % 2:
                cell.set_facecolor("#F4F4F4")

    figure.suptitle(
        "Effet de la décroissance entre les récoltes et l’imagerie",
        fontsize=22,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.006,
        "* Somme des activités d’intervalle mesurées à leurs propres heures de récolte; "
        "elle sert uniquement à quantifier l’effet de la correction temporelle.",
        ha="center",
        va="bottom",
        fontsize=12,
        color="#444444",
    )
    return _save(figure, output_path)


def build_urine_decay_detail_rows() -> list[dict[str, Any]]:
    """Return every completed collection propagated to each imaging time."""
    output: list[dict[str, Any]] = []
    for spec in urine.URINE_DATASETS:
        dataset = urine.build_urine_dataset(spec, timepoint_count=3)
        urine_items = list(dataset["urine_data"]["urine"])
        half_life_h = float(dataset["urine_data"]["half_life_h"])
        for record in dataset["records"]:
            imaging_h = float(record["hours_after_injection"])
            timepoint = str(record["timepoint"]).replace("Day", "J")
            completed_items = [
                item for item in urine_items if float(item["elapsed_h"]) <= imaging_h
            ]
            contribution_sum = 0.0
            if not completed_items:
                output.append(
                    {
                        "dataset_label": spec.label,
                        "patient_id": spec.patient_id,
                        "cycle_id": spec.cycle_id,
                        "timepoint": timepoint,
                        "row_type": "no_collection",
                        "collection_label": "Aucune collecte terminée",
                        "collection_time_h": float("nan"),
                        "imaging_time_h": imaging_h,
                        "elapsed_collection_to_imaging_h": float("nan"),
                        "activity_at_collection_mbq": 0.0,
                        "physical_decay_factor": float("nan"),
                        "activity_contribution_at_imaging_mbq": 0.0,
                        "physical_half_life_h": half_life_h,
                    }
                )
                continue

            for collection_index, item in enumerate(completed_items, start=1):
                collection_h = float(item["elapsed_h"])
                elapsed_h = imaging_h - collection_h
                activity_at_collection = float(item["total_excreted_activity_mbq"])
                factor = float(urine.sang_decay_factor(elapsed_h, half_life_h))
                contribution = activity_at_collection * factor
                contribution_sum += contribution
                output.append(
                    {
                        "dataset_label": spec.label,
                        "patient_id": spec.patient_id,
                        "cycle_id": spec.cycle_id,
                        "timepoint": timepoint,
                        "row_type": "collection",
                        "collection_label": f"R{collection_index}",
                        "collection_time_h": collection_h,
                        "imaging_time_h": imaging_h,
                        "elapsed_collection_to_imaging_h": elapsed_h,
                        "activity_at_collection_mbq": activity_at_collection,
                        "physical_decay_factor": factor,
                        "activity_contribution_at_imaging_mbq": contribution,
                        "physical_half_life_h": half_life_h,
                    }
                )

            expected = float(record["cumulative_excreted_activity_mbq"])
            if not np.isclose(contribution_sum, expected, rtol=1e-10, atol=1e-8):
                raise ValueError(
                    f"Somme détaillée incohérente pour {spec.label} {timepoint}: "
                    f"{contribution_sum} vs {expected} MBq"
                )
            output.append(
                {
                    "dataset_label": spec.label,
                    "patient_id": spec.patient_id,
                    "cycle_id": spec.cycle_id,
                    "timepoint": timepoint,
                    "row_type": "total",
                    "collection_label": f"Total {timepoint}",
                    "collection_time_h": float("nan"),
                    "imaging_time_h": imaging_h,
                    "elapsed_collection_to_imaging_h": float("nan"),
                    "activity_at_collection_mbq": float(
                        sum(
                            float(item["total_excreted_activity_mbq"])
                            for item in completed_items
                        )
                    ),
                    "physical_decay_factor": float("nan"),
                    "activity_contribution_at_imaging_mbq": contribution_sum,
                    "physical_half_life_h": half_life_h,
                }
            )
    return output


def plot_urine_decay_detail_table(
    rows: Sequence[Mapping[str, Any]],
    dataset_label: str,
    output_path: Path,
) -> Path:
    """Create one auditable contribution table for a patient/cycle."""
    selected = [dict(row) for row in rows if row["dataset_label"] == dataset_label]
    if not selected:
        raise ValueError(f"Aucun détail de décroissance pour {dataset_label}")
    color = slopes.cycle_color(selected[0])
    headers = (
        "Image",
        "Récolte",
        "t récolte\n(h)",
        "t image\n(h)",
        "Δt\n(h)",
        "Activité mesurée à la\nrécolte (MBq)",
        "Facteur de\ndécroissance",
        "Contribution à l’heure\nde l’image (MBq)",
    )
    table_rows: list[tuple[str, ...]] = []
    row_types: list[str] = []
    previous_timepoint = ""
    for row in selected:
        row_type = str(row["row_type"])
        timepoint = str(row["timepoint"])
        image_label = timepoint if timepoint != previous_timepoint else ""
        previous_timepoint = timepoint
        if row_type == "no_collection":
            table_rows.append(
                (
                    image_label,
                    "Aucune collecte terminée",
                    "—",
                    f"{float(row['imaging_time_h']):.1f}",
                    "—",
                    "0",
                    "—",
                    "0",
                )
            )
        elif row_type == "total":
            table_rows.append(
                (
                    "",
                    str(row["collection_label"]),
                    "—",
                    f"{float(row['imaging_time_h']):.1f}",
                    "—",
                    f"{float(row['activity_at_collection_mbq']):,.0f}".replace(",", " "),
                    "—",
                    f"{float(row['activity_contribution_at_imaging_mbq']):,.0f}".replace(",", " "),
                )
            )
        else:
            table_rows.append(
                (
                    image_label,
                    str(row["collection_label"]),
                    f"{float(row['collection_time_h']):.1f}",
                    f"{float(row['imaging_time_h']):.1f}",
                    f"{float(row['elapsed_collection_to_imaging_h']):.1f}",
                    f"{float(row['activity_at_collection_mbq']):,.1f}".replace(",", " "),
                    f"{float(row['physical_decay_factor']):.4f}",
                    f"{float(row['activity_contribution_at_imaging_mbq']):,.1f}".replace(",", " "),
                )
            )
        row_types.append(row_type)

    figure_height = max(8.8, 0.48 * len(table_rows) + 3.4)
    figure, axis = plt.subplots(
        figsize=(17.0, figure_height), facecolor="white", layout="constrained"
    )
    axis.axis("off")
    table = axis.table(
        cellText=table_rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=(0.06, 0.13, 0.09, 0.09, 0.08, 0.18, 0.14, 0.18),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)
    table.scale(1.0, 1.48)
    for (row_index, _), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D0D0")
        cell.set_linewidth(0.8)
        if row_index == 0:
            cell.set_height(cell.get_height() * 1.40)
            cell.set_facecolor(color)
            cell.set_text_props(color="white", fontweight="bold", fontsize=11)
            continue
        row_type = row_types[row_index - 1]
        if row_type == "total":
            cell.set_facecolor("#E6E6E6")
            cell.set_text_props(fontweight="bold")
        elif row_index % 2:
            cell.set_facecolor("#F7F7F7")

    half_life_h = float(selected[0]["physical_half_life_h"])
    half_life_label = f"{half_life_h:.1f}".replace(".", ",")
    figure.suptitle(
        f"Contribution de chaque récolte à l’heure de l’imagerie — {dataset_label}",
        fontsize=21,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.025,
        "Contribution à l’image = activité mesurée à la récolte × "
        f"2^(−Δt / {half_life_label} h). Les récoltes sont recalculées séparément "
        "pour J1 et J2.",
        ha="center",
        va="bottom",
        fontsize=12.5,
        color="#333333",
    )
    return _save(figure, output_path)


def plot_urine_decay_method_table(output_path: Path) -> Path:
    """Document the two-step same-time correction used by the pipeline."""
    rows = (
        (
            "1",
            "Mesure de chaque récolte",
            "Aᵢ(tᵢ)",
            "Activité urine + couche mesurée à l’heure tᵢ de la récolte",
        ),
        (
            "2",
            "Retour au temps d’injection",
            "Aᵢ(0) = Aᵢ(tᵢ) / D(tᵢ)",
            "Toutes les récoltes sont placées sur une même référence temporelle",
        ),
        (
            "3",
            "Sélection temporelle",
            "tᵢ ≤ t_image",
            "Seules les récoltes terminées avant l’image sont incluses",
        ),
        (
            "4",
            "Somme à l’injection",
            "E(0) = Σ Aᵢ(0)",
            "Somme des activités excrétées équivalentes au temps d’injection",
        ),
        (
            "5",
            "Propagation à l’imagerie",
            "E(t_image) = E(0) × D(t_image)",
            "Activité excrétée cumulée à l’heure exacte de l’imagerie",
        ),
        (
            "6",
            "Activité corporelle restante",
            "A_corps(t_image) = A_inj × D(t_image) − E(t_image)",
            "Cible urinaire utilisée dans les analyses de sensibilité",
        ),
    )
    headers = ("Étape", "Opération", "Calcul", "Interprétation")
    figure, axis = plt.subplots(
        figsize=(17.0, 7.2), facecolor="white", layout="constrained"
    )
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="left",
        colLoc="center",
        loc="center",
        colWidths=(0.07, 0.22, 0.30, 0.39),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12.2)
    table.scale(1.0, 1.85)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D0D0")
        cell.set_linewidth(0.8)
        if row_index == 0:
            cell.set_height(cell.get_height() * 1.30)
            cell.set_facecolor("#333333")
            cell.set_text_props(color="white", fontweight="bold", ha="center")
        elif row_index % 2:
            cell.set_facecolor("#F3F3F3")
        if row_index > 0 and column_index == 0:
            cell.set_text_props(fontweight="bold", ha="center")
    figure.suptitle(
        "Correction temporelle des récoltes à l’heure de l’imagerie",
        fontsize=22,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.045,
        "D(t) = 2^(−t / 159,5 h). La même correction est appliquée séparément "
        "à chaque heure d’imagerie.",
        ha="center",
        fontsize=13,
        color="#333333",
    )
    return _save(figure, output_path)


def build_urine_decay_method_comparison_rows() -> list[dict[str, Any]]:
    """Compare the historical two-step formula with the direct formula."""
    output: list[dict[str, Any]] = []
    for spec in urine.URINE_DATASETS:
        dataset = urine.build_urine_dataset(spec, timepoint_count=3)
        balance = dataset["urine_balance"]
        collection_times = np.asarray(balance["times_h"], dtype=float)
        interval_activity = np.asarray(balance["interval_excreted_mbq"], dtype=float)
        interval_equivalent = np.asarray(
            balance["interval_injection_equivalent_mbq"], dtype=float
        )
        injected = float(dataset["urine_data"]["injected_activity_mbq"])
        half_life_h = float(dataset["urine_data"]["half_life_h"])
        for record in dataset["records"]:
            imaging_h = float(record["hours_after_injection"])
            mask = collection_times <= imaging_h
            image_decay = float(urine.sang_decay_factor(imaging_h, half_life_h))
            excreted_two_step = float(interval_equivalent[mask].sum() * image_decay)
            excreted_direct = float(
                np.sum(
                    interval_activity[mask]
                    * urine.sang_decay_factor(
                        imaging_h - collection_times[mask], half_life_h
                    )
                )
            )
            body_two_step = injected * image_decay - excreted_two_step
            body_direct = injected * image_decay - excreted_direct
            expected_excreted = float(record["cumulative_excreted_activity_mbq"])
            expected_body = float(record["urine_remaining_activity_mbq"])
            if not np.isclose(excreted_two_step, expected_excreted, rtol=1e-10, atol=1e-8):
                raise ValueError("La méthode historique ne reproduit pas le pipeline")
            if not np.isclose(body_two_step, expected_body, rtol=1e-10, atol=1e-8):
                raise ValueError("La cible corporelle historique ne reproduit pas le pipeline")
            output.append(
                {
                    "dataset_label": spec.label,
                    "patient_id": spec.patient_id,
                    "cycle_id": spec.cycle_id,
                    "timepoint": str(record["timepoint"]).replace("Day", "J"),
                    "imaging_time_h": imaging_h,
                    "completed_collections": int(np.count_nonzero(mask)),
                    "excreted_two_step_mbq": excreted_two_step,
                    "excreted_direct_mbq": excreted_direct,
                    "excreted_difference_mbq": excreted_direct - excreted_two_step,
                    "body_two_step_mbq": body_two_step,
                    "body_direct_mbq": body_direct,
                    "body_difference_mbq": body_direct - body_two_step,
                }
            )
    return output


def plot_urine_decay_method_comparison(
    rows: Sequence[Mapping[str, Any]], output_path: Path
) -> Path:
    """Show numerical equality of the two temporal formulations."""
    headers = (
        "Patient / cycle",
        "Image",
        "t image\n(h)",
        "Récoltes\nincluses",
        "Excrétée — méthode\n2 étapes (MBq)",
        "Excrétée — méthode\ndirecte (MBq)",
        "Écart excrété\n(MBq)",
        "Corps — méthode\n2 étapes (MBq)",
        "Corps — méthode\ndirecte (MBq)",
        "Écart corps\n(MBq)",
    )
    table_rows = []
    for row in rows:
        table_rows.append(
            (
                str(row["dataset_label"]),
                str(row["timepoint"]),
                f"{float(row['imaging_time_h']):.1f}",
                str(int(row["completed_collections"])),
                f"{float(row['excreted_two_step_mbq']):,.3f}".replace(",", " "),
                f"{float(row['excreted_direct_mbq']):,.3f}".replace(",", " "),
                f"{float(row['excreted_difference_mbq']):.2e}",
                f"{float(row['body_two_step_mbq']):,.3f}".replace(",", " "),
                f"{float(row['body_direct_mbq']):,.3f}".replace(",", " "),
                f"{float(row['body_difference_mbq']):.2e}",
            )
        )
    figure, axis = plt.subplots(
        figsize=(19.0, 8.8), facecolor="white", layout="constrained"
    )
    axis.axis("off")
    table = axis.table(
        cellText=table_rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=(0.13, 0.06, 0.07, 0.08, 0.12, 0.12, 0.09, 0.12, 0.12, 0.09),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.4)
    table.scale(1.0, 1.55)
    previous_dataset = None
    patient_colors = {
        "P11 — mai 2026": "#1F77B4",
        "P8 — juin 2026": "#D62728",
        "P17 — septembre 2026": "#2CA02C",
    }
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D0D0")
        cell.set_linewidth(0.8)
        if row_index == 0:
            cell.set_height(cell.get_height() * 1.40)
            cell.set_facecolor("#333333")
            cell.set_text_props(color="white", fontweight="bold", fontsize=9.8)
            continue
        dataset_label = str(rows[row_index - 1]["dataset_label"])
        if dataset_label != previous_dataset:
            previous_dataset = dataset_label
        if row_index % 2:
            cell.set_facecolor("#F5F5F5")
        if column_index == 0:
            cell.set_text_props(
                color=patient_colors.get(dataset_label, "#222222"),
                fontweight="bold",
            )
    figure.suptitle(
        "Vérification de l’équivalence des deux méthodes temporelles",
        fontsize=22,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.035,
        "Les écarts résiduels sont uniquement des arrondis numériques. "
        "La colonne “Corps” correspond à l’activité urinaire de référence du pipeline.",
        ha="center",
        fontsize=12.5,
        color="#333333",
    )
    return _save(figure, output_path)


def _transfer_mare_lookup(reference: str) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for method, _, section, _, _ in METHODS:
        rows = [
            row
            for row in _read_csv(
                _method_dir(reference, section) / "transferts_intercycle_metriques.csv"
            )
            if row["combination"] == COMBINATION
            and row["window"] == WINDOW
            and row["training_dataset"] in COMMON_KEYS
            and row["test_dataset"] in COMMON_KEYS
        ]
        output[method] = {
            key: float(
                np.mean(
                    [
                        float(row["mare_percent"])
                        for row in rows
                        if row["training_dataset"] == key
                    ]
                )
            )
            for key in COMMON_KEYS
        }
    return output


def _load_transfer_rows(reference: str, section: str) -> list[dict[str, str]]:
    return [
        row
        for row in _read_csv(
            _method_dir(reference, section) / "transferts_intercycle_metriques.csv"
        )
        if row["combination"] == COMBINATION
        and row["window"] == WINDOW
        and row["training_dataset"] in COMMON_KEYS
        and row["test_dataset"] in COMMON_KEYS
    ]


def plot_transfer_error(output_path: Path) -> Path:
    values = {
        reference: _transfer_mare_lookup(reference) for reference in REFERENCES
    }
    methods = tuple((method, label) for method, label, _, _, _ in METHODS)
    return _plot_error_panels(
        values, methods, output_path,
        "Erreur lors du transfert de la calibration vers un autre cycle",
        "MARE intercycle (%)",
    )


def plot_transfer_matrix(method: str, output_path: Path) -> Path:
    """Montrer chaque transfert source-vers-cible et sa moyenne de ligne."""
    method_spec = next(item for item in METHODS if item[0] == method)
    _, method_label, section, _, _ = method_spec
    matrices: dict[str, np.ndarray] = {}
    for reference in REFERENCES:
        rows = _load_transfer_rows(reference, section)
        matrix = np.full((len(COMMON_KEYS), len(COMMON_KEYS) + 1), np.nan)
        for source_index, source_key in enumerate(COMMON_KEYS):
            source_values: list[float] = []
            for test_index, test_key in enumerate(COMMON_KEYS):
                if source_key == test_key:
                    continue
                row = next(
                    item
                    for item in rows
                    if item["training_dataset"] == source_key
                    and item["test_dataset"] == test_key
                )
                value = float(row["mare_percent"])
                matrix[source_index, test_index] = value
                source_values.append(value)
            matrix[source_index, -1] = float(np.mean(source_values))
        matrices[reference] = matrix

    maximum = max(
        float(np.nanmax(matrix[:, :-1])) for matrix in matrices.values()
    )
    color_map = plt.get_cmap("YlOrRd").copy()
    color_map.set_bad(color="#EEEEEE")
    figure, axes = plt.subplots(
        1, 2, figsize=(14.7, 6.6), facecolor="white", layout="constrained"
    )
    cycle_labels = []
    origin_rows = _load_fit_rows("urine", "2_sensibilite_J0_J2")
    origin_lookup = {row["dataset_key"]: row for row in origin_rows}
    for key in COMMON_KEYS:
        cycle_labels.append(_cycle_label(origin_lookup[key]).replace("\n", " — "))

    image_handle = None
    for axis, reference in zip(axes, ("urine", "qspect")):
        matrix = matrices[reference]
        image_handle = axis.imshow(
            matrix, cmap=color_map, vmin=0.0, vmax=maximum, aspect="auto"
        )
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                if np.isnan(value):
                    text_value = "—"
                    color = "#666666"
                else:
                    text_value = f"{value:.1f} %"
                    color = "white" if value > 0.52 * maximum else "#222222"
                axis.text(
                    column_index, row_index, text_value,
                    ha="center", va="center", fontsize=14,
                    fontweight="bold", color=color,
                )
        axis.axvline(2.5, color="#222222", linewidth=2.0)
        axis.set_xticks(
            np.arange(len(COMMON_KEYS) + 1), cycle_labels + ["Moyenne"]
        )
        axis.set_yticks(np.arange(len(COMMON_KEYS)), cycle_labels)
        axis.set_xlabel("Cycle testé sans réajustement", fontsize=14)
        axis.set_ylabel("Cycle utilisé pour ajuster la pente", fontsize=14)
        axis.set_title(str(REFERENCES[reference]["label"]), fontsize=18,
                       fontweight="bold")
        axis.tick_params(axis="both", labelsize=12)
        for label in axis.get_xticklabels():
            label.set_rotation(25)
            label.set_ha("right")
        for spine in axis.spines.values():
            spine.set_visible(False)
    if image_handle is not None:
        color_bar = figure.colorbar(image_handle, ax=axes, shrink=0.88, pad=0.025)
        color_bar.set_label("MARE sur les trois timepoints du cycle testé (%)",
                            fontsize=13)
        color_bar.ax.tick_params(labelsize=11)
    figure.suptitle(
        f"Transfert de la calibration entre cycles — {method_label}",
        fontsize=21, fontweight="bold",
    )
    return _save(figure, output_path)


def build_summary_table() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for reference in REFERENCES:
        fit_values = _fit_mare_lookup(reference)
        transfer_values = _transfer_mare_lookup(reference)
        for method, method_label, _, _, _ in METHODS:
            rows.append(
                {
                    "reference": reference,
                    "reference_label": REFERENCES[reference]["label"],
                    "fit_method": method,
                    "fit_method_label": method_label,
                    "mean_intracycle_mare_percent": float(
                        np.mean(list(fit_values[method].values()))
                    ),
                    "mean_intercycle_transfer_mare_percent": float(
                        np.mean(list(transfer_values[method].values()))
                    ),
                    "number_of_cycles": len(COMMON_KEYS),
                }
            )
    return rows


def run(output_dir: Path = OUTPUT_DIR) -> dict[str, Path]:
    output_dir = Path(output_dir)
    urine_decay_rows = build_urine_decay_validation_rows()
    urine_decay_detail_rows = build_urine_decay_detail_rows()
    urine_decay_comparison_rows = build_urine_decay_method_comparison_rows()
    outputs = {
        "urine_origin": plot_urine_relation(
            "origin", output_dir / "01a_relation_urine_origine_imposee.png"
        ),
        "urine_free": plot_urine_relation(
            "free_intercept", output_dir / "01b_relation_urine_intercept_libre.png"
        ),
        "urine_anchor": plot_urine_relation(
            "anchored_j0", output_dir / "01c_relation_urine_ancree_J0.png"
        ),
        "urine_anchor_stable": plot_urine_relation(
            "anchored_j0", output_dir / "01_relation_simple_comptes_urine.png"
        ),
        "urine_sensitivity": plot_sensitivity_comparison(
            "urine", output_dir / "04_sensibilite_urine_image_complete.png"
        ),
        "qspect_sensitivity": plot_sensitivity_comparison(
            "qspect", output_dir / "05_sensibilite_qspect_crop.png"
        ),
        "intracycle_error": plot_intracycle_error(
            output_dir / "06_comparaison_erreur_urine_qspect.png"
        ),
        "j0_and_fit_error": plot_j0_and_fit_error(
            output_dir / "07_erreur_J0_force_vs_pente_J0_J2.png"
        ),
        "transfer_error": plot_transfer_error(
            output_dir / "08_transfert_intercycle_contraintes.png"
        ),
        "transfer_matrix_origin": plot_transfer_matrix(
            "origin", output_dir / "09a_matrice_transfert_origine_imposee.png"
        ),
        "transfer_matrix_free": plot_transfer_matrix(
            "free_intercept", output_dir / "09b_matrice_transfert_intercept_libre.png"
        ),
        "transfer_matrix_anchor": plot_transfer_matrix(
            "anchored_j0", output_dir / "09c_matrice_transfert_ancrage_J0.png"
        ),
        "transfer_matrix_stable": plot_transfer_matrix(
            "anchored_j0", output_dir / "09_matrice_transfert_intercycle.png"
        ),
        "selected_urine_sensitivity": plot_single_method_sensitivity(
            "urine", "free_intercept",
            SELECTED_OUTPUT_DIR / "01_sensibilite_urine_intercept_libre.png",
        ),
        "selected_qspect_sensitivity": plot_single_method_sensitivity(
            "qspect", "free_intercept",
            SELECTED_OUTPUT_DIR / "02_sensibilite_qspect_intercept_libre.png",
        ),
        "selected_urine_fit": plot_urine_relation(
            "free_intercept",
            SELECTED_OUTPUT_DIR / "03_fit_photopeak_urine_intercept_libre.png",
            include_method_in_title=False,
        ),
        "selected_reference_error": plot_single_method_reference_error(
            "free_intercept",
            SELECTED_OUTPUT_DIR
            / "04_erreur_urine_vs_qspect_intercept_libre.png",
        ),
        "selected_j0_urine_sensitivity": plot_single_method_sensitivity(
            "urine", "anchored_j0",
            SELECTED_J0_OUTPUT_DIR / "01_sensibilite_urine_ancrage_J0.png",
        ),
        "selected_j0_qspect_sensitivity": plot_single_method_sensitivity(
            "qspect", "anchored_j0",
            SELECTED_J0_OUTPUT_DIR / "02_sensibilite_qspect_ancrage_J0.png",
        ),
        "selected_j0_urine_fit": plot_urine_relation(
            "anchored_j0",
            SELECTED_J0_OUTPUT_DIR / "03_fit_photopeak_urine_ancrage_J0.png",
            include_method_in_title=False,
        ),
        "selected_j0_reference_error": plot_single_method_reference_error(
            "anchored_j0",
            SELECTED_J0_OUTPUT_DIR
            / "04_erreur_urine_vs_qspect_ancrage_J0.png",
        ),
        "urine_decay_validation": plot_urine_decay_validation(
            urine_decay_rows,
            output_dir / "10_validation_decroissance_urine_imagerie.png",
        ),
        "urine_decay_detail_p11": plot_urine_decay_detail_table(
            urine_decay_detail_rows,
            "P11 — mai 2026",
            URINE_DECAY_DETAIL_DIR / "P11_detail_decroissance_par_recolte.png",
        ),
        "urine_decay_detail_p8": plot_urine_decay_detail_table(
            urine_decay_detail_rows,
            "P8 — juin 2026",
            URINE_DECAY_DETAIL_DIR / "P8_detail_decroissance_par_recolte.png",
        ),
        "urine_decay_detail_p17": plot_urine_decay_detail_table(
            urine_decay_detail_rows,
            "P17 — septembre 2026",
            URINE_DECAY_DETAIL_DIR / "P17_detail_decroissance_par_recolte.png",
        ),
        "urine_decay_method": plot_urine_decay_method_table(
            URINE_DECAY_DETAIL_DIR / "methode_correction_temporelle.png",
        ),
        "urine_decay_method_comparison": plot_urine_decay_method_comparison(
            urine_decay_comparison_rows,
            URINE_DECAY_DETAIL_DIR / "comparaison_methodes_temporelles.png",
        ),
    }
    outputs["summary_csv"] = _write_csv(
        output_dir / "contraintes_pente_resultats.csv", build_summary_table()
    )
    outputs["urine_decay_validation_csv"] = _write_csv(
        output_dir / "10_validation_decroissance_urine_imagerie.csv",
        urine_decay_rows,
    )
    outputs["urine_decay_detail_csv"] = _write_csv(
        URINE_DECAY_DETAIL_DIR / "detail_decroissance_par_recolte.csv",
        urine_decay_detail_rows,
    )
    outputs["urine_decay_method_comparison_csv"] = _write_csv(
        URINE_DECAY_DETAIL_DIR / "comparaison_methodes_temporelles.csv",
        urine_decay_comparison_rows,
    )
    return outputs


if __name__ == "__main__":
    for name, path in run().items():
        print(f"{name}: {path}")
