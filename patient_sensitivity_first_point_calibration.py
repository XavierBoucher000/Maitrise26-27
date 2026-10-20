"""Simple J0 calibration of whole-body planar count rates.

This module is the presentation-oriented entry point that precedes the
spectral models.  It uses the first planar acquisition of each patient as a
single sensitivity calibration, predicts the following acquisitions without
refitting, and compares the predictions only with urine-derived remaining
body activity at the planar acquisition time.

The injected activity corrected only for physical decay is also shown.  It is
the no-excretion reference, not a measurement of retained body activity.
Planar measurements use the complete image and no CT or crop. Raw photopeak
AP+PA is the primary measurement; pixelwise geometric mean and TEW are
secondary robustness checks.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figure_layout
import patient_sensitivity_multicycle_v3 as v3
import patient_sensitivity_urine_target as urine


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = (
    figure_layout.analysis_dir("patient_sensitivity_urine_target.py")
    / "1_sensibilite_J0"
    / "5_ancienne_analyse_p11_p8"
)
DEFAULT_TIMEPOINT_COUNT = 3

METHOD_AP_PA = urine.SIMPLE_PHOTO_AP_PA
METHOD_GM = urine.SIMPLE_PHOTO_GM_PIXEL
METHOD_TEW_AP_PA = v3.MODE_TEW_FULL_AP_PA
METHOD_TEW_GM = v3.MODE_TEW_FULL_GM
METHOD_ORDER = (METHOD_AP_PA, METHOD_GM)
TEW_METHOD_ORDER = (METHOD_TEW_AP_PA, METHOD_TEW_GM)
ALL_METHOD_ORDER = (*METHOD_ORDER, *TEW_METHOD_ORDER)
METHOD_LABELS = {
    METHOD_AP_PA: "Photopeak brut — image complète AP+PA",
    METHOD_GM: "Photopeak brut — image complète GM pixel",
    METHOD_TEW_AP_PA: "TEW — image complète AP+PA",
    METHOD_TEW_GM: "TEW — image complète GM pixel",
}
METHOD_FILE_LABELS = {
    METHOD_AP_PA: "photopeak_AP_PA",
    METHOD_GM: "photopeak_GM_pixel",
    METHOD_TEW_AP_PA: "TEW_AP_PA",
    METHOD_TEW_GM: "TEW_GM_pixel",
}
METHOD_SHORT_LABELS = {
    METHOD_AP_PA: "Somme AP+PA",
    METHOD_GM: "GM pixel par pixel",
    METHOD_TEW_AP_PA: "TEW — somme AP+PA",
    METHOD_TEW_GM: "TEW — GM pixel",
}

REFERENCE_URINE = "urine_remaining"
REFERENCE_ORDER = (REFERENCE_URINE,)
REFERENCE_LABELS = {
    REFERENCE_URINE: "Activité restante selon les urines",
}
REFERENCE_SHORT_LABELS = {
    REFERENCE_URINE: "Urines",
}
REFERENCE_COLORS = {
    REFERENCE_URINE: "#2CA02C",
}
REFERENCE_MARKERS = {
    REFERENCE_URINE: "s",
}

DATASET_CONFIG = (
    ("p11", "P11 — mai 2026", urine.TRAINING_DATASET),
    ("p8", "P8 — juin 2026", urine.TEST_DATASET),
)


def _validate_positive(values: Sequence[float], label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{label} must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{label} must contain finite positive values")
    return array


def calibrate_first_point(
    measurement_rate_cps: Sequence[float],
    first_activity_mbq: float,
) -> tuple[float, np.ndarray]:
    """Calibrate one sensitivity at J0 and predict every planar timepoint."""
    rates = _validate_positive(measurement_rate_cps, "measurement_rate_cps")
    activity = float(first_activity_mbq)
    if not np.isfinite(activity) or activity <= 0.0:
        raise ValueError("first_activity_mbq must be finite and positive")
    sensitivity = float(rates[0] / activity)
    return sensitivity, rates / sensitivity


def prediction_metrics(
    reference_mbq: Sequence[float],
    predicted_mbq: Sequence[float],
    start_index: int = 0,
) -> Dict[str, float]:
    """Return percentage metrics, optionally excluding the calibration point."""
    reference = _validate_positive(reference_mbq, "reference_mbq")
    predicted = _validate_positive(predicted_mbq, "predicted_mbq")
    if reference.shape != predicted.shape:
        raise ValueError("reference and predicted activities must have equal shape")
    if not 0 <= int(start_index) < reference.size:
        raise ValueError("start_index is outside the observation range")
    errors = 100.0 * (predicted[int(start_index) :] / reference[int(start_index) :] - 1.0)
    return {
        "n_evaluated": int(errors.size),
        "bias_percent": float(np.mean(errors)),
        "mare_percent": float(np.mean(np.abs(errors))),
        "rmse_percent": float(np.sqrt(np.mean(errors**2))),
        "maximum_absolute_error_percent": float(np.max(np.abs(errors))),
    }


def build_patient_dataset(
    patient_id: str,
    label: str,
    urine_spec: urine.UrineDatasetSpec,
    count: int = DEFAULT_TIMEPOINT_COUNT,
) -> Dict[str, Any]:
    """Load full-image planar observations and the urine-derived target."""
    urine_dataset = urine.build_urine_dataset(urine_spec, count)
    records = list(urine_dataset["records"][:count])
    physical = _validate_positive(
        [record["physical_available_activity_mbq"] for record in records],
        "physical-only activity",
    )
    urine_remaining = _validate_positive(
        [record["urine_remaining_activity_mbq"] for record in records],
        "urine-derived activity",
    )
    hours = np.asarray([record["hours_after_injection"] for record in records], dtype=float)
    half_life_h = float(records[0]["physical_half_life_h"])
    if not np.isclose(physical[0], urine_remaining[0], rtol=0.0, atol=1.0e-6):
        raise ValueError(
            f"{label}: J0 is not before the first completed urine collection"
        )
    return {
        "patient_id": patient_id,
        "label": label,
        "records": records,
        "timepoints": [str(record["timepoint"]) for record in records],
        "hours_after_injection": hours,
        "half_life_h": half_life_h,
        "injected_activity_mbq": float(records[0]["injected_activity_mbq"]),
        "calibration_activity_j0_mbq": float(physical[0]),
        "references": {
            REFERENCE_URINE: urine_remaining,
        },
        "rates": {
            METHOD_AP_PA: _validate_positive(
                urine_dataset["simple_rates"][METHOD_AP_PA],
                f"{label}/{METHOD_AP_PA} rates",
            ),
            METHOD_GM: _validate_positive(
                urine_dataset["simple_rates"][METHOD_GM],
                f"{label}/{METHOD_GM} rates",
            ),
            METHOD_TEW_AP_PA: _validate_positive(
                urine_dataset["mode_observations"][METHOD_TEW_AP_PA][
                    "measurement_rate"
                ],
                f"{label}/{METHOD_TEW_AP_PA} rates",
            ),
            METHOD_TEW_GM: _validate_positive(
                urine_dataset["mode_observations"][METHOD_TEW_GM][
                    "measurement_rate"
                ],
                f"{label}/{METHOD_TEW_GM} rates",
            ),
        },
    }


def build_all_datasets(count: int = DEFAULT_TIMEPOINT_COUNT) -> Dict[str, Dict[str, Any]]:
    return {
        patient_id: build_patient_dataset(
            patient_id, label, urine_spec, count
        )
        for patient_id, label, urine_spec in DATASET_CONFIG
    }


def _row_common(dataset: Mapping[str, Any], index: int) -> Dict[str, Any]:
    record = dataset["records"][index]
    return {
        "patient_id": dataset["patient_id"],
        "patient_label": dataset["label"],
        "timepoint": dataset["timepoints"][index],
        "hours_after_injection": float(dataset["hours_after_injection"][index]),
        "planar_datetime": record["planar_datetime"],
    }


def reference_rows(datasets: Mapping[str, Mapping[str, Any]]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for dataset in datasets.values():
        for index in range(len(dataset["timepoints"])):
            row = _row_common(dataset, index)
            row["calibration_activity_j0_mbq"] = float(
                dataset["calibration_activity_j0_mbq"]
            )
            for reference in REFERENCE_ORDER:
                value = float(dataset["references"][reference][index])
                row[f"{reference}_activity_at_planar_mbq"] = value
            rows.append(row)
    return rows


def j0_sensitivity_comparison_rows(
    datasets: Mapping[str, Mapping[str, Any]],
    methods: Sequence[str] = METHOD_ORDER,
) -> list[Dict[str, Any]]:
    """Compare patient-specific J0 sensitivities and their transfer error."""
    rows: list[Dict[str, Any]] = []
    for method in methods:
        sensitivities = {
            patient_id: float(dataset["rates"][method][0])
            / float(dataset["calibration_activity_j0_mbq"])
            for patient_id, dataset in datasets.items()
        }
        p11 = sensitivities["p11"]
        p8 = sensitivities["p8"]
        rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "p11_sensitivity_cps_per_mbq": p11,
                "p8_sensitivity_cps_per_mbq": p8,
                "p11_over_p8": p11 / p8,
                "p11_minus_p8_relative_to_p8_percent": 100.0 * (p11 / p8 - 1.0),
                "j0_error_p11_calibration_applied_to_p8_percent": 100.0
                * (p8 / p11 - 1.0),
                "j0_error_p8_calibration_applied_to_p11_percent": 100.0
                * (p11 / p8 - 1.0),
            }
        )
    return rows


def intrapatient_results(
    datasets: Mapping[str, Mapping[str, Any]],
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    predictions: list[Dict[str, Any]] = []
    metrics_rows: list[Dict[str, Any]] = []
    results: Dict[str, Dict[str, Any]] = {}
    for patient_id, dataset in datasets.items():
        results[patient_id] = {}
        calibration_activity = float(dataset["calibration_activity_j0_mbq"])
        for method in ALL_METHOD_ORDER:
            sensitivity, predicted = calibrate_first_point(
                dataset["rates"][method], calibration_activity
            )
            results[patient_id][method] = {
                "sensitivity_cps_per_mbq": sensitivity,
                "predicted_activity_mbq": predicted,
            }
            for index, value in enumerate(predicted):
                row = _row_common(dataset, index)
                row.update(
                    {
                        "analysis": "intrapatient_J0",
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "calibration_patient": patient_id,
                        "sensitivity_cps_per_mbq": sensitivity,
                        "used_for_calibration": index == 0,
                        "predicted_activity_mbq": float(value),
                    }
                )
                for reference in REFERENCE_ORDER:
                    target = float(dataset["references"][reference][index])
                    row[f"{reference}_activity_mbq"] = target
                    row[f"error_vs_{reference}_percent"] = 100.0 * (value / target - 1.0)
                predictions.append(row)
            for reference in REFERENCE_ORDER:
                metric = prediction_metrics(
                    dataset["references"][reference], predicted, start_index=1
                )
                metrics_rows.append(
                    {
                        "analysis": "intrapatient_J0",
                        "calibration_patient": patient_id,
                        "test_patient": patient_id,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "reference": reference,
                        "reference_label": REFERENCE_LABELS[reference],
                        "sensitivity_cps_per_mbq": sensitivity,
                        **metric,
                    }
                )
    return predictions, metrics_rows, results


def interpatient_results(
    datasets: Mapping[str, Mapping[str, Any]],
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    predictions: list[Dict[str, Any]] = []
    metrics_rows: list[Dict[str, Any]] = []
    results: Dict[str, Dict[str, Any]] = {}
    for source_id, target_id in (("p11", "p8"), ("p8", "p11")):
        source = datasets[source_id]
        target = datasets[target_id]
        direction = f"{source_id}_to_{target_id}"
        results[direction] = {}
        source_activity_j0 = float(source["calibration_activity_j0_mbq"])
        for method in ALL_METHOD_ORDER:
            sensitivity, _source_prediction = calibrate_first_point(
                source["rates"][method], source_activity_j0
            )
            predicted = np.asarray(target["rates"][method], dtype=float) / sensitivity
            results[direction][method] = {
                "source_patient": source_id,
                "target_patient": target_id,
                "sensitivity_cps_per_mbq": sensitivity,
                "predicted_activity_mbq": predicted,
            }
            for index, value in enumerate(predicted):
                row = _row_common(target, index)
                row.update(
                    {
                        "analysis": "interpatient_J0_transfer",
                        "direction": direction,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "calibration_patient": source_id,
                        "test_patient": target_id,
                        "sensitivity_cps_per_mbq": sensitivity,
                        "predicted_activity_mbq": float(value),
                    }
                )
                for reference in REFERENCE_ORDER:
                    target_value = float(target["references"][reference][index])
                    row[f"{reference}_activity_mbq"] = target_value
                    row[f"error_vs_{reference}_percent"] = 100.0 * (
                        value / target_value - 1.0
                    )
                predictions.append(row)
            for reference in REFERENCE_ORDER:
                metric = prediction_metrics(
                    target["references"][reference], predicted, start_index=1
                )
                metrics_rows.append(
                    {
                        "analysis": "interpatient_J0_transfer",
                        "direction": direction,
                        "calibration_patient": source_id,
                        "test_patient": target_id,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "reference": reference,
                        "reference_label": REFERENCE_LABELS[reference],
                        "sensitivity_cps_per_mbq": sensitivity,
                        **metric,
                    }
                )
    return predictions, metrics_rows, results


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_reference_activities(
    datasets: Mapping[str, Mapping[str, Any]], path: Path
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    for axis, dataset in zip(axes, datasets.values()):
        x = np.arange(len(dataset["timepoints"]))
        for reference in REFERENCE_ORDER:
            axis.plot(
                x,
                np.asarray(dataset["references"][reference]) / 1000.0,
                color=REFERENCE_COLORS[reference],
                marker=REFERENCE_MARKERS[reference],
                linewidth=2.4,
                markersize=7,
                label=REFERENCE_SHORT_LABELS[reference],
            )
        axis.set_title(dataset["label"], fontsize=15)
        axis.set_xticks(x, dataset["timepoints"])
        axis.set_xlabel("Timepoint")
        axis.set_ylabel("Activité au temps du planaire (GBq)")
        axis.grid(alpha=0.25, linestyle="--")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Activité restante selon les urines au moment de l’imagerie", fontsize=18, y=0.99)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.80))
    return _save_figure(fig, path)


def plot_reference_activity_balance(
    datasets: Mapping[str, Mapping[str, Any]], path: Path
) -> Path:
    """Show how retained activity and measured excretion form the reference.

    Every component is expressed at the planar acquisition time.  Therefore,
    retained body activity plus the activity-equivalent of completed urine
    collections reproduces the injected activity remaining after physical
    decay.  This is the J0/J1/J2 analogue of the older planar-plus-excretion
    balance figure, without fitting a planar activity curve.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.3), sharey=False)
    for axis, dataset in zip(axes, datasets.values()):
        hours = np.asarray(dataset["hours_after_injection"], dtype=float)
        remaining = np.asarray(
            dataset["references"][REFERENCE_URINE], dtype=float
        )
        physical_available = np.asarray(
            [
                record["physical_available_activity_mbq"]
                for record in dataset["records"]
            ],
            dtype=float,
        )
        excreted_equivalent = physical_available - remaining
        reconstructed_total = remaining + excreted_equivalent

        axis.plot(
            hours,
            remaining / 1000.0,
            color="#1F77B4",
            marker="s",
            linewidth=2.6,
            markersize=7,
            label="Activité corporelle restante",
        )
        axis.plot(
            hours,
            excreted_equivalent / 1000.0,
            color="#FF7F0E",
            marker="o",
            linewidth=2.4,
            markersize=7,
            label="Excrétion cumulée incluse",
        )
        axis.plot(
            hours,
            reconstructed_total / 1000.0,
            color="#2CA02C",
            marker="D",
            linewidth=2.8,
            markersize=6,
            label="Somme des deux composantes",
        )
        axis.plot(
            hours,
            physical_available / 1000.0,
            color="black",
            linestyle="--",
            linewidth=1.8,
            label="Activité injectée après décroissance physique",
        )
        for index, (x_value, timepoint) in enumerate(
            zip(hours, dataset["timepoints"])
        ):
            axis.annotate(
                str(timepoint),
                (x_value, float(reconstructed_total[index]) / 1000.0),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=10,
            )
        axis.set_title(dataset["label"], fontsize=15)
        axis.set_xlabel("Temps après l’injection (h)")
        axis.set_ylabel("Activité au temps du planaire (GBq)")
        axis.grid(alpha=0.25, linestyle="--")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(
        "Construction de la référence urinaire à J0, J1 et J2",
        fontsize=18,
        y=0.99,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
        fontsize=10.5,
    )
    fig.text(
        0.5,
        0.015,
        "Activité corporelle restante + excrétion cumulée = activité disponible après décroissance physique. "
        "Seules les collectes terminées avant chaque image sont incluses.",
        ha="center",
        fontsize=10.5,
        color="#333333",
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.82))
    return _save_figure(fig, path)


def plot_j0_sensitivity_comparison(
    rows: Sequence[Mapping[str, Any]], path: Path
) -> Path:
    """Show the two patient-specific J0 sensitivities and relative gap."""
    if not rows:
        raise ValueError("At least one J0 sensitivity row is required")
    column_count = 2
    row_count = int(np.ceil(len(rows) / column_count))
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(13.5, 5.2 * row_count),
        sharey=False,
        squeeze=False,
    )
    patient_colors = ("#1F77B4", "#D62728")
    flat_axes = axes.ravel()
    for axis, row in zip(flat_axes, rows):
        values = [
            float(row["p11_sensitivity_cps_per_mbq"]),
            float(row["p8_sensitivity_cps_per_mbq"]),
        ]
        bars = axis.bar((0, 1), values, color=patient_colors, width=0.58)
        axis.bar_label(bars, fmt="%.3f", fontsize=12, padding=4)
        gap = float(row["p11_minus_p8_relative_to_p8_percent"])
        axis.text(
            0.5,
            0.80,
            f"P11 est {gap:.1f} % plus élevée que P8",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
        )
        axis.set_xticks((0, 1), ("P11-J0", "P8-J0"))
        axis.set_ylabel("Sensibilité J0 (cps/MBq)")
        axis.set_title(METHOD_LABELS[str(row["method"])], fontsize=13)
        axis.grid(axis="y", alpha=0.25, linestyle="--")
    for axis in flat_axes[len(rows) :]:
        axis.set_visible(False)
    fig.suptitle("La sensibilité calibrée à J0 dépend du patient", fontsize=18, y=0.99)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return _save_figure(fig, path)


def _plot_prediction_panels(
    datasets: Mapping[str, Mapping[str, Any]],
    results: Sequence[tuple[str, str, np.ndarray]],
    method: str,
    title: str,
    path: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2))
    for axis, (panel_title, target_id, predicted) in zip(axes, results):
        dataset = datasets[target_id]
        x = np.arange(len(dataset["timepoints"]))
        for reference in REFERENCE_ORDER:
            axis.plot(
                x,
                np.asarray(dataset["references"][reference]) / 1000.0,
                color=REFERENCE_COLORS[reference],
                marker=REFERENCE_MARKERS[reference],
                linewidth=2.0,
                markersize=6,
                label=REFERENCE_SHORT_LABELS[reference],
            )
        axis.plot(
            x,
            np.asarray(predicted) / 1000.0,
            color="#D62728",
            marker="^",
            linewidth=2.8,
            markersize=8,
            label="Prédiction planaire",
        )
        axis.set_title(panel_title, fontsize=14)
        axis.set_xticks(x, dataset["timepoints"])
        axis.set_xlabel("Timepoint")
        axis.set_ylabel("Activité au temps du planaire (GBq)")
        axis.grid(alpha=0.25, linestyle="--")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(f"{title}\n{METHOD_LABELS[method]}", fontsize=17, y=0.995)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.82),
        ncol=4,
        frameon=False,
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.72))
    return _save_figure(fig, path)


def plot_intrapatient_predictions(
    datasets: Mapping[str, Mapping[str, Any]],
    results: Mapping[str, Mapping[str, Any]],
    method: str,
    path: Path,
) -> Path:
    panels = [
        (
            f"{datasets[patient_id]['label']} — calibration à son J0",
            patient_id,
            np.asarray(results[patient_id][method]["predicted_activity_mbq"]),
        )
        for patient_id in ("p11", "p8")
    ]
    return _plot_prediction_panels(
        datasets,
        panels,
        method,
        "Calibration intrapatient au premier point",
        path,
    )


def plot_interpatient_predictions(
    datasets: Mapping[str, Mapping[str, Any]],
    results: Mapping[str, Mapping[str, Any]],
    method: str,
    path: Path,
) -> Path:
    panels = []
    for direction in ("p11_to_p8", "p8_to_p11"):
        result = results[direction][method]
        source = str(result["source_patient"]).upper()
        target = str(result["target_patient"]).upper()
        panels.append(
            (
                f"Sensibilité J0 {source} appliquée à {target}",
                str(result["target_patient"]),
                np.asarray(result["predicted_activity_mbq"]),
            )
        )
    return _plot_prediction_panels(
        datasets,
        panels,
        method,
        "Transfert interpatient sans réajustement",
        path,
    )


def _plot_combined_prediction_grid(
    datasets: Mapping[str, Mapping[str, Any]],
    method_panels: Sequence[
        tuple[str, Sequence[tuple[str, str, np.ndarray]]]
    ],
    title: str,
    path: Path,
) -> Path:
    """Combine every requested raw/TEW measurement method in one grid."""
    row_count = len(method_panels)
    if row_count == 0:
        raise ValueError("At least one planar measurement method is required")
    fig, axes = plt.subplots(
        row_count,
        2,
        figsize=(14.5, 4.45 * row_count + 1.2),
        sharey=True,
        squeeze=False,
    )
    for row_index, (method, panels) in enumerate(method_panels):
        for column_index, (panel_title, target_id, predicted) in enumerate(panels):
            axis = axes[row_index, column_index]
            dataset = datasets[target_id]
            x = np.arange(len(dataset["timepoints"]))
            for reference in REFERENCE_ORDER:
                axis.plot(
                    x,
                    np.asarray(dataset["references"][reference]) / 1000.0,
                    color=REFERENCE_COLORS[reference],
                    marker=REFERENCE_MARKERS[reference],
                    linewidth=2.2,
                    markersize=7,
                    label=REFERENCE_SHORT_LABELS[reference],
                )
            axis.plot(
                x,
                np.asarray(predicted) / 1000.0,
                color="#D62728",
                marker="^",
                linewidth=2.8,
                markersize=8,
                label="Prédiction planaire",
            )
            axis.set_title(panel_title, fontsize=13)
            axis.set_xticks(x, dataset["timepoints"])
            axis.set_xlabel("Timepoint")
            if column_index == 0:
                axis.set_ylabel("Activité au temps du planaire (GBq)")
            axis.text(
                0.03,
                0.94,
                METHOD_SHORT_LABELS[method],
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=11.5,
                fontweight="bold",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "#B0B0B0",
                    "alpha": 0.90,
                    "pad": 3,
                },
            )
            axis.grid(alpha=0.25, linestyle="--")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(title, fontsize=18, y=0.985)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90), h_pad=2.8, w_pad=2.2)
    return _save_figure(fig, path)


def plot_combined_intrapatient_predictions(
    datasets: Mapping[str, Mapping[str, Any]],
    results: Mapping[str, Mapping[str, Any]],
    path: Path,
    methods: Sequence[str] = METHOD_ORDER,
) -> Path:
    method_panels = []
    for method in methods:
        panels = [
            (
                f"{datasets[patient_id]['label']} — calibration à son J0",
                patient_id,
                np.asarray(results[patient_id][method]["predicted_activity_mbq"]),
            )
            for patient_id in ("p11", "p8")
        ]
        method_panels.append((method, panels))
    return _plot_combined_prediction_grid(
        datasets,
        method_panels,
        "Calibration intrapatient au premier point",
        path,
    )


def plot_combined_interpatient_predictions(
    datasets: Mapping[str, Mapping[str, Any]],
    results: Mapping[str, Mapping[str, Any]],
    path: Path,
    methods: Sequence[str] = METHOD_ORDER,
) -> Path:
    method_panels = []
    for method in methods:
        panels = []
        for direction in ("p11_to_p8", "p8_to_p11"):
            result = results[direction][method]
            source = str(result["source_patient"]).upper()
            target = str(result["target_patient"]).upper()
            panels.append(
                (
                    f"Sensibilité J0 {source} appliquée à {target}",
                    str(result["target_patient"]),
                    np.asarray(result["predicted_activity_mbq"]),
                )
            )
        method_panels.append((method, panels))
    return _plot_combined_prediction_grid(
        datasets,
        method_panels,
        "Transfert interpatient sans réajustement",
        path,
    )


def plot_error_by_timepoint(
    prediction_rows: Sequence[Mapping[str, Any]],
    category_field: str,
    category_order: Sequence[str],
    category_labels: Mapping[str, str],
    title: str,
    path: Path,
    methods: Sequence[str] = METHOD_ORDER,
) -> Path:
    """Show absolute relative error separately at every follow-up timepoint."""
    timepoints: list[str] = []
    for row in prediction_rows:
        timepoint = str(row["timepoint"])
        if timepoint != "J0" and timepoint not in timepoints:
            timepoints.append(timepoint)
    if not timepoints:
        raise ValueError("No follow-up timepoint is available for the error plot")

    method_colors = {
        METHOD_AP_PA: "#1F77B4",
        METHOD_GM: "#FF7F0E",
        METHOD_TEW_AP_PA: "#2CA02C",
        METHOD_TEW_GM: "#9467BD",
    }
    x = np.arange(len(timepoints), dtype=float)
    fig, axes = plt.subplots(1, len(category_order), figsize=(13.8, 5.9), sharey=True)
    axes_array = np.atleast_1d(axes)
    all_errors: list[float] = []
    for axis, category in zip(axes_array, category_order):
        for method in methods:
            values: list[float] = []
            for timepoint in timepoints:
                matches = [
                    row
                    for row in prediction_rows
                    if str(row[category_field]) == category
                    and str(row["method"]) == method
                    and str(row["timepoint"]) == timepoint
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Missing prediction for {category}/{method}/{timepoint}"
                    )
                values.append(
                    abs(float(matches[0]["error_vs_urine_remaining_percent"]))
                )
            all_errors.extend(values)
            axis.plot(
                x,
                values,
                color=method_colors[method],
                marker="o",
                linewidth=2.6,
                markersize=8,
                label=METHOD_SHORT_LABELS[method],
            )
            for position, value in zip(x, values):
                axis.annotate(
                    f"{value:.1f}",
                    (position, value),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=10,
                    color=method_colors[method],
                    fontweight="bold",
                )
        axis.set_title(category_labels[category], fontsize=14)
        axis.set_xticks(x, timepoints)
        axis.set_xlabel("Jour évalué")
        axis.set_ylabel("Erreur absolue relative (%)")
        axis.grid(alpha=0.25, linestyle="--")
    axes_array[0].set_ylim(0.0, 1.16 * max(all_errors))
    handles, labels = axes_array[0].get_legend_handles_labels()
    fig.suptitle(title, fontsize=17, y=0.99)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=len(methods),
        frameon=False,
        fontsize=10.5,
    )
    fig.text(
        0.5,
        0.015,
        "J0 exclu. Pour un patient et un jour, cette valeur est la contribution individuelle à la MARE.",
        ha="center",
        fontsize=10.5,
        color="#4D4D4D",
    )
    fig.tight_layout(rect=(0.0, 0.055, 1.0, 0.82))
    return _save_figure(fig, path)


def plot_tew_error_effect(
    prediction_rows: Sequence[Mapping[str, Any]],
    category_field: str,
    category_order: Sequence[str],
    category_labels: Mapping[str, str],
    title: str,
    path: Path,
) -> Path:
    """Show how TEW changes absolute activity error relative to raw photopeak."""
    timepoints: list[str] = []
    for row in prediction_rows:
        timepoint = str(row["timepoint"])
        if timepoint != "J0" and timepoint not in timepoints:
            timepoints.append(timepoint)
    if not timepoints:
        raise ValueError("No follow-up timepoint is available for the TEW plot")

    method_pairs = (
        (METHOD_AP_PA, METHOD_TEW_AP_PA, "AP+PA"),
        (METHOD_GM, METHOD_TEW_GM, "GM pixel"),
    )
    colors = ("#1F77B4", "#FF7F0E")
    x = np.arange(len(timepoints), dtype=float)
    fig, axes = plt.subplots(
        1, len(category_order), figsize=(13.8, 5.9), sharey=True, squeeze=False
    )
    axes_array = axes[0]
    all_deltas: list[float] = []
    for axis, category in zip(axes_array, category_order):
        for pair_index, (raw_method, tew_method, pair_label) in enumerate(method_pairs):
            deltas: list[float] = []
            for timepoint in timepoints:
                raw_matches = [
                    row
                    for row in prediction_rows
                    if str(row[category_field]) == category
                    and str(row["method"]) == raw_method
                    and str(row["timepoint"]) == timepoint
                ]
                tew_matches = [
                    row
                    for row in prediction_rows
                    if str(row[category_field]) == category
                    and str(row["method"]) == tew_method
                    and str(row["timepoint"]) == timepoint
                ]
                if len(raw_matches) != 1 or len(tew_matches) != 1:
                    raise ValueError(
                        f"Missing raw/TEW prediction for {category}/{pair_label}/{timepoint}"
                    )
                raw_error = abs(
                    float(raw_matches[0]["error_vs_urine_remaining_percent"])
                )
                tew_error = abs(
                    float(tew_matches[0]["error_vs_urine_remaining_percent"])
                )
                deltas.append(tew_error - raw_error)
            all_deltas.extend(deltas)
            axis.plot(
                x,
                deltas,
                color=colors[pair_index],
                marker="o",
                linewidth=2.6,
                markersize=8,
                label=pair_label,
            )
            for position, value in zip(x, deltas):
                axis.annotate(
                    f"{value:+.1f}",
                    (position, value),
                    xytext=(0, 8 if value >= 0.0 else -14),
                    textcoords="offset points",
                    ha="center",
                    fontsize=10,
                    color=colors[pair_index],
                    fontweight="bold",
                )
        axis.axhline(0.0, color="#333333", linewidth=1.2)
        axis.set_title(category_labels[category], fontsize=14)
        axis.set_xticks(x, timepoints)
        axis.set_xlabel("Jour évalué")
        axis.set_ylabel("Variation de l’erreur absolue avec TEW (points de %)")
        axis.grid(alpha=0.25, linestyle="--")

    maximum = max((abs(value) for value in all_deltas), default=1.0)
    limit = max(2.0, 1.25 * maximum)
    axes_array[0].set_ylim(-limit, limit)
    handles, labels = axes_array[0].get_legend_handles_labels()
    fig.suptitle(title, fontsize=17, y=0.99)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=2,
        frameon=False,
        fontsize=10.5,
    )
    fig.text(
        0.5,
        0.015,
        "Valeur = |erreur TEW| − |erreur brute|. Négatif : TEW améliore; positif : TEW détériore. J0 exclu.",
        ha="center",
        fontsize=10.5,
        color="#4D4D4D",
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.82))
    return _save_figure(fig, path)


def _format_metrics_table(
    rows: Sequence[Mapping[str, Any]], category_field: str
) -> list[str]:
    lines = [
        " catégorie | mesure planaire | référence | MARE | biais",
        "---------- | ---------------- | --------- | ---- | -----",
    ]
    for row in rows:
        lines.append(
            f"{str(row[category_field]):>10} | {METHOD_LABELS[str(row['method'])]:<34} | "
            f"{REFERENCE_SHORT_LABELS[str(row['reference'])]:<18} | "
            f"{float(row['mare_percent']):6.2f}% | {float(row['bias_percent']):+7.2f}%"
        )
    return lines


def build_report(
    datasets: Mapping[str, Mapping[str, Any]],
    j0_rows: Sequence[Mapping[str, Any]],
    intrapatient_metrics: Sequence[Mapping[str, Any]],
    interpatient_metrics: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "CALIBRATION PLANAIRE AU PREMIER POINT — CHEMIN PRINCIPAL",
        "=" * 62,
        "",
        "But",
        "---",
        "Tester la méthode la plus simple avant les modèles spectraux: une seule",
        "sensibilité est définie au premier point, puis verrouillée.",
        "",
        "1. Activités au temps de l'imagerie",
        "------------------------------------",
        "A_urine(t) = [A_injectée - E_cumulée,équiv.t0(t)] × 2^(-t/T_phys).",
        "Seules les collectes terminées avant l'image sont soustraites.",
        "",
        "La décroissance physique sert uniquement à exprimer l'activité de",
        "calibration au temps exact de J0. Aucune courbe 'physique seulement'",
        "n'est utilisée comme référence pour juger J1 ou J2.",
        "",
        "À J0, aucune collecte n'est terminée. L'activité injectée corrigée au",
        "temps de J0 est donc identique à la cible urinaire de J0.",
        "",
        "2. Mesure planaire",
        "-------------------",
        "Analyse principale: photopeak brut, image complète, somme AP+PA.",
        "Analyse secondaire: photopeak brut, image complète, GM pixel par pixel",
        "puis somme. Les mêmes deux combinaisons sont recalculées avec TEW.",
        "Les figures comparatives regroupent donc brut AP+PA, brut GM, TEW",
        "AP+PA et TEW GM dans un même cadre.",
        "L'effet propre du TEW est Δ_TEW = |erreur_TEW| - |erreur_brute|;",
        "Δ_TEW < 0 indique une amélioration et Δ_TEW > 0 une détérioration.",
        "Aucun crop, CT, correction d'atténuation ou modèle M0-M9.",
        "",
        "3. Calibration intrapatient",
        "-----------------------------",
        "S_J0 = R_planaire(J0) / A_calibration(J0)",
        "A_prédite(Ji) = R_planaire(Ji) / S_J0",
        "J1 et J2 sont prédits sans réajustement. Les métriques excluent J0.",
        "",
        "Sensibilités mesurées à J0:",
    ]
    for row in j0_rows:
        lines.append(
            f"  {METHOD_LABELS[str(row['method'])]}: P11 = "
            f"{float(row['p11_sensitivity_cps_per_mbq']):.4f}, P8 = "
            f"{float(row['p8_sensitivity_cps_per_mbq']):.4f} cps/MBq; "
            f"écart relatif = {float(row['p11_minus_p8_relative_to_p8_percent']):.2f} %."
        )
    lines.extend(
        [
        "",
        *_format_metrics_table(intrapatient_metrics, "test_patient"),
        "",
        "4. Transfert interpatient",
        "---------------------------",
        "La sensibilité de P11-J0 est appliquée aux trois images de P8, puis la",
        "sensibilité de P8-J0 est appliquée aux trois images de P11. Aucun point",
        "du patient cible n'est utilisé pour réajuster le facteur. J0 sert à",
        "visualiser le décalage initial entre patients, mais les métriques et",
        "les figures d'erreur incluent uniquement J1 et J2 du patient cible.",
        "",
        *_format_metrics_table(interpatient_metrics, "direction"),
        "",
        "5. Interprétation",
        "-----------------",
        "Les prédictions sont évaluées uniquement contre la rétention dérivée",
        "des urines. La décroissance physique demeure une correction temporelle",
        "de J0, et non une cible clinique supplémentaire.",
        "",
        "L'erreur causée uniquement par un transfert de sensibilité vaut",
        "S_cible/S_source - 1. Si les sensibilités étaient réellement 2 et 1",
        "cps/MBq, le transfert 2→1 produirait -50 %, tandis que 1→2 produirait",
        "+100 %. Le MARE est donc directionnel lorsque les sensibilités diffèrent.",
        "",
        "Limites",
        "-------",
        "Deux patients et trois points par patient ne démontrent pas une",
        "généralisation. La cible urinaire dépend de la complétude et de l'heure",
        "des collectes. Une différence entre deux patients suggère une variation",
        "interpatient, mais ne permet pas encore d'en estimer la distribution.",
        "",
        "Organisation des résultats",
        "---------------------------",
        "1_activites_reference: cible urinaire et sensibilités J0.",
        "2_calibration_intrapatient: chaque patient calibré sur son propre J0,",
        "avec comparaison brut/TEW et effet du TEW sur l'erreur.",
        "3_transfert_interpatient: P11→P8 et P8→P11 sans réajustement,",
        "avec comparaison brut/TEW et effet du TEW sur l'erreur.",
        "4_methodologie: ce rapport et les tableaux récapitulatifs.",
    ]
    )
    for dataset in datasets.values():
        lines.extend(
            [
                "",
                f"{dataset['label']}: activité injectée = "
                f"{float(dataset['injected_activity_mbq']) / 1000.0:.3f} GBq; "
                f"T_phys = {float(dataset['half_life_h']):.2f} h.",
            ]
        )
    return "\n".join(lines) + "\n"


def build_readme() -> str:
    return """DOSSIER PRINCIPAL — CALIBRATION AU PREMIER POINT
================================================

Lire les dossiers dans l'ordre:

1_activites_reference
  Montre l'activité restante selon les urines, sa combinaison avec l'excrétion
  cumulée, puis compare les sensibilités J0 de P11 et P8. La décroissance
  physique sert seulement à positionner les activités au temps du planaire.

2_calibration_intrapatient
  Chaque patient est calibré sur son propre J0; J1 et J2 sont des prédictions.
  Les figures comparatives montrent ensemble le photopeak brut et le TEW,
  avec AP+PA et moyenne géométrique pixel par pixel.

3_transfert_interpatient
  Sensibilité P11-J0 appliquée à P8, puis sensibilité P8-J0 appliquée à P11,
  avec photopeak brut et TEW dans les mêmes figures comparatives.

Les figures « effet_TEW_sur_erreur » montrent la variation de l'erreur absolue
causée par le TEW: une valeur négative signifie une amélioration.

4_methodologie
  Formules, hypothèses, métriques et limites.

Les modèles M0-M9 et le fit urinaire continu sont volontairement absents de
ce dossier. Ils restent des analyses exploratoires séparées.
"""


def run_first_point_calibration(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    timepoint_count: int = DEFAULT_TIMEPOINT_COUNT,
) -> Dict[str, Any]:
    """Run the clean chronological J0-calibration analysis."""
    output_dir = Path(output_dir)
    references_dir = output_dir / "1_activites_reference"
    intrapatient_dir = output_dir / "2_calibration_intrapatient"
    interpatient_dir = output_dir / "3_transfert_interpatient"
    methodology_dir = output_dir / "4_methodologie"
    for directory in (
        output_dir,
        references_dir,
        intrapatient_dir,
        interpatient_dir,
        methodology_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    datasets = build_all_datasets(timepoint_count)
    refs = reference_rows(datasets)
    j0_rows = j0_sensitivity_comparison_rows(datasets)
    tew_j0_rows = j0_sensitivity_comparison_rows(datasets, TEW_METHOD_ORDER)
    all_j0_rows = [*j0_rows, *tew_j0_rows]
    intra_predictions, intra_metrics, intra_results = intrapatient_results(datasets)
    inter_predictions, inter_metrics, inter_results = interpatient_results(datasets)

    paths: Dict[str, Path] = {
        "readme": output_dir / "README.txt",
        "reference_csv": _write_csv(references_dir / "activites_reference.csv", refs),
        "reference_figure": plot_reference_activities(
            datasets, references_dir / "activites_au_temps_planaire.png"
        ),
        "reference_balance_figure": plot_reference_activity_balance(
            datasets,
            references_dir / "bilan_activite_reference_J0_J1_J2.png",
        ),
        "j0_sensitivity_csv": _write_csv(
            references_dir / "sensibilites_J0.csv", j0_rows
        ),
        "j0_sensitivity_figure": plot_j0_sensitivity_comparison(
            all_j0_rows, references_dir / "sensibilites_J0_par_patient.png"
        ),
        "tew_j0_sensitivity_csv": _write_csv(
            references_dir / "sensibilites_J0_TEW.csv", tew_j0_rows
        ),
        "tew_j0_sensitivity_figure": plot_j0_sensitivity_comparison(
            tew_j0_rows, references_dir / "sensibilites_J0_TEW_par_patient.png"
        ),
        "intrapatient_predictions_csv": _write_csv(
            intrapatient_dir / "predictions_intrapatient.csv", intra_predictions
        ),
        "intrapatient_metrics_csv": _write_csv(
            intrapatient_dir / "metriques_intrapatient.csv", intra_metrics
        ),
        "interpatient_predictions_csv": _write_csv(
            interpatient_dir / "predictions_interpatient.csv", inter_predictions
        ),
        "interpatient_metrics_csv": _write_csv(
            interpatient_dir / "metriques_interpatient.csv", inter_metrics
        ),
    }
    for method in ALL_METHOD_ORDER:
        file_label = METHOD_FILE_LABELS[method]
        paths[f"intrapatient_{method}"] = plot_intrapatient_predictions(
            datasets,
            intra_results,
            method,
            intrapatient_dir / f"predictions_intrapatient_{file_label}.png",
        )
        paths[f"interpatient_{method}"] = plot_interpatient_predictions(
            datasets,
            inter_results,
            method,
            interpatient_dir / f"predictions_interpatient_{file_label}.png",
        )
    paths["intrapatient_combined"] = plot_combined_intrapatient_predictions(
        datasets,
        intra_results,
        intrapatient_dir / "predictions_intrapatient_AP_PA_et_GM.png",
        ALL_METHOD_ORDER,
    )
    paths["interpatient_combined"] = plot_combined_interpatient_predictions(
        datasets,
        inter_results,
        interpatient_dir / "predictions_interpatient_AP_PA_et_GM.png",
        ALL_METHOD_ORDER,
    )
    paths["intrapatient_mare"] = plot_error_by_timepoint(
        intra_predictions,
        "patient_id",
        ("p11", "p8"),
        {"p11": "P11", "p8": "P8"},
        "Erreur par jour — photopeak brut et TEW, calibration intrapatient J0",
        intrapatient_dir / "MARE_intrapatient.png",
        ALL_METHOD_ORDER,
    )
    paths["interpatient_mare"] = plot_error_by_timepoint(
        inter_predictions,
        "direction",
        ("p11_to_p8", "p8_to_p11"),
        {"p11_to_p8": "P11→P8", "p8_to_p11": "P8→P11"},
        "Erreur par jour — photopeak brut et TEW, transfert interpatient",
        interpatient_dir / "MARE_interpatient.png",
        ALL_METHOD_ORDER,
    )
    paths["intrapatient_tew_error_effect"] = plot_tew_error_effect(
        intra_predictions,
        "patient_id",
        ("p11", "p8"),
        {"p11": "P11", "p8": "P8"},
        "Effet du TEW sur l’erreur — calibration intrapatient J0",
        intrapatient_dir / "effet_TEW_sur_erreur_intrapatient.png",
    )
    paths["interpatient_tew_error_effect"] = plot_tew_error_effect(
        inter_predictions,
        "direction",
        ("p11_to_p8", "p8_to_p11"),
        {"p11_to_p8": "P11→P8", "p8_to_p11": "P8→P11"},
        "Effet du TEW sur l’erreur — transfert interpatient",
        interpatient_dir / "effet_TEW_sur_erreur_interpatient.png",
    )

    paths["readme"].write_text(build_readme(), encoding="utf-8")
    report_path = methodology_dir / "methodologie_et_resultats.txt"
    report_path.write_text(
        build_report(datasets, all_j0_rows, intra_metrics, inter_metrics),
        encoding="utf-8",
    )
    paths["methodology_report"] = report_path
    _write_csv(methodology_dir / "resume_metriques.csv", [*intra_metrics, *inter_metrics])
    paths["metrics_summary_csv"] = methodology_dir / "resume_metriques.csv"

    return {
        "datasets": datasets,
        "j0_sensitivity_comparison": all_j0_rows,
        "intrapatient_metrics": intra_metrics,
        "interpatient_metrics": inter_metrics,
        "paths": paths,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timepoints", type=int, default=DEFAULT_TIMEPOINT_COUNT)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_first_point_calibration(args.output_dir, args.timepoints)
    print("Calibration au premier point terminée")
    for name, path in result["paths"].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
