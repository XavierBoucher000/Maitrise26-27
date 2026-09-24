"""Grouped multi-cycle validation of planar effective-sensitivity models.

V2 is intentionally left unchanged.  This module generalizes the locked
planar spectral model to explicit lists of training and test cycles.  A cycle
can never occur in both groups.  The default run evaluates:

* every directed single-cycle transfer;
* every train-two-cycles/test-one-cycle split;
* the reverse patient-level transfer, P8 -> both P11 cycles.

The Q/SPECT activity is used only to construct the training target and to
score held-out predictions.  No CT voxel value is a model input.  The current
profile crop is still Q/SPECT-guided and is therefore a development-stage
preprocessing step rather than a final planar-only deployment solution.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import attenuation_correction as ac
import attenuation_global_model_v2 as spectral_model
import figure_layout
import patient_effective_sensitivity as sensitivity
import patient_sensitivity_feature_test_v2 as v2


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = figure_layout.shared_dir(__file__)

LOCKED_FEATURE = spectral_model.BROAD_FEATURE_NAME
ASYMMETRY_FEATURE = "abs AP/PA photopeak asymmetry"
LOWER_FEATURE = "log lower/photopeak"
UPPER_FEATURE = "log upper/photopeak"
TEW_FRACTION_FEATURE = "TEW retained fraction"

MODEL_CONSTANT = "M0_constant"
MODEL_LOCKED = "M1_locked_spectral"
MODEL_ASYMMETRY = "M2_spectral_plus_ap_pa"
MODEL_ORDER = (MODEL_CONSTANT, MODEL_LOCKED, MODEL_ASYMMETRY)
MODEL_LABELS = {
    MODEL_CONSTANT: "M0 — sensibilité constante",
    MODEL_LOCKED: "M1 — ratio spectral verrouillé",
    MODEL_ASYMMETRY: "M2 — ratio spectral + asymétrie AP/PA",
}
MODEL_COLORS = {
    MODEL_CONSTANT: "#7A7A7A",
    MODEL_LOCKED: "#2E9F55",
    MODEL_ASYMMETRY: "#8C6BB1",
}
MODEL_MARKERS = {
    MODEL_CONSTANT: "o",
    MODEL_LOCKED: "D",
    MODEL_ASYMMETRY: "s",
}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    patient_id: str
    cycle_id: str
    label: str
    short_label: str
    planar_dir: Path
    qspect_dir: Path


@dataclass(frozen=True)
class SplitDefinition:
    split_id: str
    split_type: str
    training_datasets: tuple[str, ...]
    test_datasets: tuple[str, ...]
    label: str


def _configured_spec(
    key: str,
    patient_id: str,
    cycle_id: str,
    short_label: str,
) -> DatasetSpec:
    configured = v2.CYCLE_DATASETS[(patient_id, cycle_id)]
    return DatasetSpec(
        key=key,
        patient_id=patient_id,
        cycle_id=cycle_id,
        label=str(configured["patient_label"]),
        short_label=short_label,
        planar_dir=Path(configured["planar_dir"]),
        qspect_dir=Path(configured["qspect_dir"]),
    )


DATASETS: Dict[str, DatasetSpec] = {
    "p11_may": _configured_spec(
        "p11_may", "p11", "2026-05__Studies", "P11-mai"
    ),
    "p11_july": _configured_spec(
        "p11_july", "p11", "2026-07__Studies", "P11-juillet"
    ),
    "p8_june": _configured_spec(
        "p8_june", "p8", "2026-06__Studies", "P8-juin"
    ),
}


def default_split_definitions() -> list[SplitDefinition]:
    """Return the non-random grouped transfers used by the v3 audit."""
    keys = tuple(DATASETS)
    splits: list[SplitDefinition] = []
    for training_key in keys:
        for test_key in keys:
            if training_key == test_key:
                continue
            splits.append(
                SplitDefinition(
                    split_id=f"pair_{training_key}_to_{test_key}",
                    split_type="pairwise_cycle_transfer",
                    training_datasets=(training_key,),
                    test_datasets=(test_key,),
                    label=(
                        f"{DATASETS[training_key].short_label} → "
                        f"{DATASETS[test_key].short_label}"
                    ),
                )
            )

    for test_key in keys:
        training_keys = tuple(key for key in keys if key != test_key)
        split_type = "leave_one_cycle_out"
        if test_key == "p8_june":
            split_type += "+patient_holdout"
        splits.append(
            SplitDefinition(
                split_id=f"loco_holdout_{test_key}",
                split_type=split_type,
                training_datasets=training_keys,
                test_datasets=(test_key,),
                label=(
                    " + ".join(DATASETS[key].short_label for key in training_keys)
                    + f" → {DATASETS[test_key].short_label}"
                ),
            )
        )

    splits.append(
        SplitDefinition(
            split_id="patient_holdout_p8_to_p11_all",
            split_type="patient_holdout_reverse",
            training_datasets=("p8_june",),
            test_datasets=("p11_may", "p11_july"),
            label="P8-juin → P11-mai + P11-juillet",
        )
    )
    return splits


def _validate_dataset_keys(keys: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(key) for key in keys)
    if not normalized:
        raise ValueError("At least one dataset is required")
    unknown = sorted(set(normalized) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown dataset key(s): {', '.join(unknown)}")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Dataset lists must not contain duplicates")
    return normalized


def validate_split(
    training_datasets: Sequence[str], test_datasets: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate and return disjoint training and test dataset keys."""
    training = _validate_dataset_keys(training_datasets)
    test = _validate_dataset_keys(test_datasets)
    overlap = sorted(set(training) & set(test))
    if overlap:
        raise ValueError(
            "Training and test datasets must be disjoint; overlap: "
            + ", ".join(overlap)
        )
    return training, test


def build_dataset_observations(spec: DatasetSpec) -> Dict[str, Any]:
    """Build all planar-only features and Q/SPECT targets for one cycle."""
    rows = sensitivity.calculate_patient_effective_sensitivity(
        planar_dir=spec.planar_dir,
        qspect_dir=spec.qspect_dir,
        crop_strategy="profile_qspect",
        skip_incomplete_qspect=True,
        allow_unpaired_planar=True,
    )
    scans_by_name = {
        str(scan["scan_name"]): scan
        for scan in ac.sorted_planar_scans(spec.planar_dir)
    }
    scans = [scans_by_name[str(row["planar_scan_name"])] for row in rows]
    spectral_rows = [
        spectral_model.global_spectral_features(
            scan,
            (int(row["crop_top"]), int(row["crop_bottom"])),
            require_broad_low_energy=True,
        )
        for scan, row in zip(scans, rows)
    ]
    features = {
        name: np.asarray(
            [item["features"][name] for item in spectral_rows], dtype=np.float64
        )
        for name in spectral_model.ALL_FEATURE_NAMES
    }
    features[TEW_FRACTION_FEATURE] = np.asarray(
        [row["tew_retained_fraction"] for row in rows], dtype=np.float64
    )
    target = np.asarray(
        [row["tew_effective_sensitivity_cps_per_mbq"] for row in rows],
        dtype=np.float64,
    )
    tew_rate = np.asarray([row["tew_gm_cps"] for row in rows], dtype=np.float64)
    if target.size == 0:
        raise ValueError(f"No valid paired observations for {spec.key}")
    if np.any(~np.isfinite(target)) or np.any(target <= 0.0):
        raise ValueError(f"Invalid effective-sensitivity target in {spec.key}")

    records = []
    for row in rows:
        records.append(
            {
                "dataset_key": spec.key,
                "patient_id": spec.patient_id,
                "cycle_id": spec.cycle_id,
                "dataset_label": spec.label,
                "timepoint": str(row["label"]),
                "planar_day": float(row["planar_day"]),
                "planar_datetime": row["planar_datetime"],
                "qspect_datetime": row["qspect_datetime"],
                "crop_top": int(row["crop_top"]),
                "crop_bottom": int(row["crop_bottom"]),
                "profile_match_accepted": bool(row["profile_match_accepted"]),
            }
        )
    return {
        "spec": spec,
        "records": records,
        "features": features,
        "target_sensitivity": target,
        "tew_rate": tew_rate,
        "reference_activity": tew_rate / target,
    }


def load_observation_cache(
    dataset_keys: Iterable[str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    keys = tuple(DATASETS) if dataset_keys is None else _validate_dataset_keys(tuple(dataset_keys))
    return {key: build_dataset_observations(DATASETS[key]) for key in keys}


def _combine_observations(
    observations: Mapping[str, Mapping[str, Any]], dataset_keys: Sequence[str]
) -> Dict[str, Any]:
    keys = _validate_dataset_keys(dataset_keys)
    missing = sorted(set(keys) - set(observations))
    if missing:
        raise ValueError(f"Observation cache is missing: {', '.join(missing)}")
    feature_names = (
        *spectral_model.ALL_FEATURE_NAMES,
        TEW_FRACTION_FEATURE,
    )
    return {
        "dataset_keys": keys,
        "records": [
            record
            for key in keys
            for record in observations[key]["records"]
        ],
        "features": {
            name: np.concatenate(
                [np.asarray(observations[key]["features"][name]) for key in keys]
            )
            for name in feature_names
        },
        "target_sensitivity": np.concatenate(
            [np.asarray(observations[key]["target_sensitivity"]) for key in keys]
        ),
        "tew_rate": np.concatenate(
            [np.asarray(observations[key]["tew_rate"]) for key in keys]
        ),
        "reference_activity": np.concatenate(
            [np.asarray(observations[key]["reference_activity"]) for key in keys]
        ),
    }


def _model_features(model_name: str) -> tuple[str, ...]:
    if model_name == MODEL_CONSTANT:
        return ()
    if model_name == MODEL_LOCKED:
        return (LOCKED_FEATURE,)
    if model_name == MODEL_ASYMMETRY:
        return (LOCKED_FEATURE, ASYMMETRY_FEATURE)
    raise ValueError(f"Unknown model: {model_name}")


def fit_model(model_name: str, training: Mapping[str, Any]) -> Dict[str, Any]:
    """Fit one pre-specified model on training observations only."""
    target = np.asarray(training["target_sensitivity"], dtype=np.float64)
    feature_names = _model_features(model_name)
    if model_name == MODEL_CONSTANT:
        return {
            "model": model_name,
            "status": "fitted",
            "reason": "",
            "feature_names": feature_names,
            "coefficients": np.asarray([float(np.mean(target))]),
            "rank": 1,
            "condition_number": 1.0,
            "feature_ranges": {},
        }

    parameter_count = len(feature_names) + 1
    if target.size <= parameter_count:
        return {
            "model": model_name,
            "status": "not_estimable",
            "reason": (
                f"n_train={target.size} must exceed {parameter_count} fitted parameters"
            ),
            "feature_names": feature_names,
            "coefficients": np.full(parameter_count, np.nan),
            "rank": 0,
            "condition_number": np.nan,
            "feature_ranges": {},
        }

    columns = [
        np.asarray(training["features"][name], dtype=np.float64)
        for name in feature_names
    ]
    design = np.column_stack([np.ones(target.size), *columns])
    coefficients, _residuals, rank, singular_values = np.linalg.lstsq(
        design, np.log(target), rcond=None
    )
    if int(rank) < parameter_count:
        return {
            "model": model_name,
            "status": "not_estimable",
            "reason": "rank-deficient training design",
            "feature_names": feature_names,
            "coefficients": coefficients,
            "rank": int(rank),
            "condition_number": np.inf,
            "feature_ranges": {},
        }
    condition_number = float(singular_values[0] / singular_values[-1])
    return {
        "model": model_name,
        "status": "fitted",
        "reason": "",
        "feature_names": feature_names,
        "coefficients": coefficients,
        "rank": int(rank),
        "condition_number": condition_number,
        "feature_ranges": {
            name: (
                float(np.min(training["features"][name])),
                float(np.max(training["features"][name])),
            )
            for name in feature_names
        },
    }


def predict_model(fit: Mapping[str, Any], test: Mapping[str, Any]) -> np.ndarray:
    if fit["status"] != "fitted":
        raise ValueError(f"Cannot predict with model status {fit['status']}")
    coefficients = np.asarray(fit["coefficients"], dtype=np.float64)
    if fit["model"] == MODEL_CONSTANT:
        return np.full(
            np.asarray(test["target_sensitivity"]).shape,
            coefficients[0],
            dtype=np.float64,
        )
    columns = [
        np.asarray(test["features"][name], dtype=np.float64)
        for name in fit["feature_names"]
    ]
    design = np.column_stack(
        [np.ones(np.asarray(test["target_sensitivity"]).size), *columns]
    )
    return np.exp(design @ coefficients)


def _extrapolation_flags(
    fit: Mapping[str, Any], test: Mapping[str, Any]
) -> np.ndarray:
    count = np.asarray(test["target_sensitivity"]).size
    flags = np.zeros(count, dtype=bool)
    for name, bounds in fit.get("feature_ranges", {}).items():
        values = np.asarray(test["features"][name], dtype=np.float64)
        flags |= (values < bounds[0]) | (values > bounds[1])
    return flags


def _metrics(
    target_sensitivity: np.ndarray, predicted_sensitivity: np.ndarray
) -> Dict[str, float]:
    return v2._prediction_metrics(target_sensitivity, predicted_sensitivity)


def fit_and_evaluate(
    training_datasets: Sequence[str],
    test_datasets: Sequence[str],
    observations: Mapping[str, Mapping[str, Any]],
    *,
    split_id: str = "custom",
    split_type: str = "custom",
    split_label: str = "custom split",
) -> Dict[str, Any]:
    """Fit on X datasets and evaluate unchanged on disjoint Y datasets."""
    training_keys, test_keys = validate_split(training_datasets, test_datasets)
    training = _combine_observations(observations, training_keys)
    test = _combine_observations(observations, test_keys)
    metrics_rows: list[Dict[str, Any]] = []
    prediction_rows: list[Dict[str, Any]] = []
    fits: Dict[str, Dict[str, Any]] = {}

    for model_name in MODEL_ORDER:
        fit = fit_model(model_name, training)
        fits[model_name] = fit
        base_metric_row = {
            "split_id": split_id,
            "split_type": split_type,
            "split_label": split_label,
            "training_datasets": "+".join(training_keys),
            "test_datasets": "+".join(test_keys),
            "n_train": len(training["records"]),
            "n_test": len(test["records"]),
            "model": model_name,
            "model_label": MODEL_LABELS[model_name],
            "fit_status": fit["status"],
            "fit_reason": fit["reason"],
            "rank": fit["rank"],
            "condition_number": fit["condition_number"],
        }
        if fit["status"] != "fitted":
            metrics_rows.append(
                {
                    **base_metric_row,
                    "metric_scope": "combined_test",
                    "metric_dataset": "+".join(test_keys),
                    "activity_mare_percent": np.nan,
                    "activity_bias_percent": np.nan,
                    "activity_rmse_percent": np.nan,
                    "maximum_activity_error_percent": np.nan,
                    "sensitivity_mare_percent": np.nan,
                    "sensitivity_bias_percent": np.nan,
                }
            )
            continue

        predicted_sensitivity = predict_model(fit, test)
        extrapolated = _extrapolation_flags(fit, test)
        target_sensitivity = np.asarray(test["target_sensitivity"], dtype=np.float64)
        tew_rate = np.asarray(test["tew_rate"], dtype=np.float64)
        reference_activity = np.asarray(test["reference_activity"], dtype=np.float64)
        predicted_activity = tew_rate / predicted_sensitivity
        activity_ratio = predicted_activity / reference_activity

        combined_metrics = _metrics(target_sensitivity, predicted_sensitivity)
        metrics_rows.append(
            {
                **base_metric_row,
                "metric_scope": "combined_test",
                "metric_dataset": "+".join(test_keys),
                **combined_metrics,
            }
        )

        dataset_array = np.asarray(
            [record["dataset_key"] for record in test["records"]], dtype=object
        )
        for dataset_key in test_keys:
            selected = dataset_array == dataset_key
            scoped_metrics = _metrics(
                target_sensitivity[selected], predicted_sensitivity[selected]
            )
            metrics_rows.append(
                {
                    **base_metric_row,
                    "metric_scope": "test_dataset",
                    "metric_dataset": dataset_key,
                    **scoped_metrics,
                }
            )

        coefficient_values = np.asarray(fit["coefficients"], dtype=np.float64)
        coefficient_by_name = {"intercept_or_constant": coefficient_values[0]}
        coefficient_by_name.update(
            {
                f"coefficient_{name}": coefficient_values[index + 1]
                for index, name in enumerate(fit["feature_names"])
            }
        )
        for index, record in enumerate(test["records"]):
            prediction_rows.append(
                {
                    "split_id": split_id,
                    "split_type": split_type,
                    "split_label": split_label,
                    "training_datasets": "+".join(training_keys),
                    "test_datasets": "+".join(test_keys),
                    "model": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    **record,
                    LOCKED_FEATURE: float(test["features"][LOCKED_FEATURE][index]),
                    LOWER_FEATURE: float(test["features"][LOWER_FEATURE][index]),
                    UPPER_FEATURE: float(test["features"][UPPER_FEATURE][index]),
                    ASYMMETRY_FEATURE: float(
                        test["features"][ASYMMETRY_FEATURE][index]
                    ),
                    TEW_FRACTION_FEATURE: float(
                        test["features"][TEW_FRACTION_FEATURE][index]
                    ),
                    "tew_rate_cps": float(tew_rate[index]),
                    "observed_sensitivity_cps_per_mbq": float(
                        target_sensitivity[index]
                    ),
                    "predicted_sensitivity_cps_per_mbq": float(
                        predicted_sensitivity[index]
                    ),
                    "qspect_activity_mbq": float(reference_activity[index]),
                    "predicted_activity_mbq": float(predicted_activity[index]),
                    "predicted_over_qspect": float(activity_ratio[index]),
                    "activity_error_percent": float(
                        100.0 * (activity_ratio[index] - 1.0)
                    ),
                    "feature_extrapolation": bool(extrapolated[index]),
                    "fit_rank": fit["rank"],
                    "fit_condition_number": fit["condition_number"],
                    **coefficient_by_name,
                }
            )

    return {
        "split_id": split_id,
        "split_type": split_type,
        "split_label": split_label,
        "training": training,
        "test": test,
        "fits": fits,
        "metrics": metrics_rows,
        "predictions": prediction_rows,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _combined_metric_lookup(
    metrics_rows: Sequence[Mapping[str, Any]], split_id: str, model_name: str
) -> Mapping[str, Any] | None:
    return next(
        (
            row
            for row in metrics_rows
            if row["split_id"] == split_id
            and row["model"] == model_name
            and row["metric_scope"] == "combined_test"
        ),
        None,
    )


def plot_pairwise_transfer_matrix(
    metrics_rows: Sequence[Mapping[str, Any]], output_path: Path
) -> None:
    keys = tuple(DATASETS)
    finite_values = [
        float(row["activity_mare_percent"])
        for row in metrics_rows
        if row["split_type"] == "pairwise_cycle_transfer"
        and row["metric_scope"] == "combined_test"
        and np.isfinite(float(row["activity_mare_percent"]))
    ]
    vmax = max(10.0, max(finite_values, default=10.0))
    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(15.5, 5.0), layout="constrained")
    last_image = None
    for axis, model_name in zip(axes, MODEL_ORDER):
        matrix = np.full((len(keys), len(keys)), np.nan, dtype=np.float64)
        for row_index, training_key in enumerate(keys):
            for column_index, test_key in enumerate(keys):
                if training_key == test_key:
                    continue
                split_id = f"pair_{training_key}_to_{test_key}"
                row = _combined_metric_lookup(metrics_rows, split_id, model_name)
                if row is not None:
                    matrix[row_index, column_index] = float(
                        row["activity_mare_percent"]
                    )
        masked = np.ma.masked_invalid(matrix)
        last_image = axis.imshow(masked, cmap="YlOrRd", vmin=0.0, vmax=vmax)
        axis.set_xticks(range(len(keys)), [DATASETS[key].short_label for key in keys], rotation=30, ha="right")
        axis.set_yticks(range(len(keys)), [DATASETS[key].short_label for key in keys])
        axis.set_xlabel("Cycle de test")
        axis.set_ylabel("Cycle d’entraînement")
        axis.set_title(MODEL_LABELS[model_name])
        for row_index in range(len(keys)):
            for column_index in range(len(keys)):
                value = matrix[row_index, column_index]
                text = "—" if row_index == column_index else (
                    "N/E" if not np.isfinite(value) else f"{value:.1f} %"
                )
                axis.text(
                    column_index,
                    row_index,
                    text,
                    ha="center",
                    va="center",
                    color="white" if np.isfinite(value) and value > 0.55 * vmax else "black",
                    fontsize=10,
                    fontweight="bold",
                )
    if last_image is not None:
        colorbar = fig.colorbar(last_image, ax=axes, shrink=0.82)
        colorbar.set_label("MARE sur l’activité (%)")
    fig.suptitle("Transfert des modèles entre les trois cycles", fontsize=16)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_leave_one_cycle_out(
    metrics_rows: Sequence[Mapping[str, Any]], output_path: Path
) -> None:
    test_keys = tuple(DATASETS)
    x = np.arange(len(test_keys), dtype=np.float64)
    width = 0.24
    fig, axis = plt.subplots(figsize=(11.0, 6.2), layout="constrained")
    for model_index, model_name in enumerate(MODEL_ORDER):
        values = []
        for test_key in test_keys:
            row = _combined_metric_lookup(
                metrics_rows, f"loco_holdout_{test_key}", model_name
            )
            values.append(
                np.nan if row is None else float(row["activity_mare_percent"])
            )
        positions = x + (model_index - 1) * width
        bars = axis.bar(
            positions,
            values,
            width,
            color=MODEL_COLORS[model_name],
            label=MODEL_LABELS[model_name],
        )
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                axis.annotate(
                    f"{value:.1f}",
                    (bar.get_x() + bar.get_width() / 2.0, value),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )
    axis.set_xticks(x, [f"Test : {DATASETS[key].short_label}" for key in test_keys])
    axis.set_ylabel("MARE sur l’activité (%)")
    axis.set_title("Validation leave-one-cycle-out")
    axis.grid(axis="y", linestyle="--", alpha=0.28)
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_primary_patient_holdout(
    prediction_rows: Sequence[Mapping[str, Any]], output_path: Path
) -> None:
    split_id = "loco_holdout_p8_june"
    selected = [row for row in prediction_rows if row["split_id"] == split_id]
    if not selected:
        return
    timepoints = [
        str(row["timepoint"])
        for row in selected
        if row["model"] == MODEL_CONSTANT
    ]
    x = np.arange(len(timepoints), dtype=np.float64)
    fig, axis = plt.subplots(figsize=(10.2, 6.1), layout="constrained")
    axis.axhspan(0.90, 1.10, color="#2E9F55", alpha=0.10, label="Écart de ±10 %")
    axis.axhline(1.0, color="black", linewidth=1.4, label="Q/SPECT")
    for model_name in MODEL_ORDER:
        model_rows = [row for row in selected if row["model"] == model_name]
        model_rows.sort(key=lambda row: float(row["planar_day"]))
        if not model_rows:
            continue
        ratios = [float(row["predicted_over_qspect"]) for row in model_rows]
        axis.plot(
            x,
            ratios,
            marker=MODEL_MARKERS[model_name],
            linewidth=2.2,
            markersize=8,
            color=MODEL_COLORS[model_name],
            label=MODEL_LABELS[model_name],
        )
        for position, value in zip(x, ratios):
            axis.annotate(
                f"{value:.3f}",
                (position, value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=MODEL_COLORS[model_name],
            )
    axis.set_xticks(x, timepoints)
    axis.set_xlabel("Timepoint de P8 — cycle de juin")
    axis.set_ylabel("Activité prédite / activité Q/SPECT")
    axis.set_title("Entraînement sur les deux cycles de P11 — test externe sur P8")
    axis.grid(axis="y", linestyle="--", alpha=0.28)
    axis.legend(frameon=False, ncol=2)
    axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _format_metric(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/E"
    return "N/E" if not np.isfinite(numeric) else f"{numeric:.2f}"


def build_report(
    observations: Mapping[str, Mapping[str, Any]],
    metrics_rows: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "Multi-cycle planar effective-sensitivity model — v3",
        "====================================================",
        "",
        "Datasets:",
    ]
    for key, spec in DATASETS.items():
        lines.append(
            f"  {key}: {spec.label}, n={len(observations[key]['records'])}"
        )
    lines.extend(
        [
            "",
            "Models:",
            "  M0: arithmetic mean training sensitivity.",
            f"  M1: log(S_eff) = b0 + b1 * {LOCKED_FEATURE}.",
            "  M2: M1 plus absolute AP/PA photopeak asymmetry.",
            "  M2 is not estimated when n_train does not exceed its 3 fitted parameters.",
            "",
            "Primary grouped validations (MARE on activity, %):",
            "  held-out cycle                     |       M0 |       M1 |       M2",
            "  ---------------------------------- | -------- | -------- | --------",
        ]
    )
    for test_key in DATASETS:
        values = []
        for model_name in MODEL_ORDER:
            row = _combined_metric_lookup(
                metrics_rows, f"loco_holdout_{test_key}", model_name
            )
            values.append("N/E" if row is None else _format_metric(row["activity_mare_percent"]))
        lines.append(
            f"  {DATASETS[test_key].short_label:<34} | "
            + " | ".join(f"{value:>8}" for value in values)
        )

    primary_rows = {
        model: _combined_metric_lookup(metrics_rows, "loco_holdout_p8_june", model)
        for model in MODEL_ORDER
    }
    lines.extend(
        [
            "",
            "Primary inter-patient result:",
            "  Training = P11 May + P11 July; test = P8 June.",
        ]
    )
    for model_name in MODEL_ORDER:
        row = primary_rows[model_name]
        if row is None or row["fit_status"] != "fitted":
            lines.append(f"  {MODEL_LABELS[model_name]}: not estimable")
            continue
        lines.append(
            f"  {MODEL_LABELS[model_name]}: "
            f"MARE={float(row['activity_mare_percent']):.2f}%, "
            f"bias={float(row['activity_bias_percent']):+.2f}%, "
            f"RMSE={float(row['activity_rmse_percent']):.2f}%, "
            f"max={float(row['maximum_activity_error_percent']):.2f}%"
        )

    lines.extend(
        [
            "",
            "Interpretation limits:",
            "  - There are 11 timepoints but only two patients.",
            "  - P11 May and P11 July are two cycles from the same patient.",
            "  - The locked spectral feature was historically discovered on P11 May.",
            "    Splits that test P11 May are exploratory, not fully independent.",
            "  - The current crop is guided by Q/SPECT profile correspondence.",
            "    The model inputs are planar, but preprocessing is not yet deployable",
            "    without Q/SPECT.",
            "  - No random timepoint split is used; every split is grouped by cycle.",
            "  - M2 is exploratory and limited to two predictors.",
            "",
        ]
    )
    return "\n".join(lines)


def run_default_analysis(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    observations = load_observation_cache()
    results = []
    metrics_rows: list[Dict[str, Any]] = []
    prediction_rows: list[Dict[str, Any]] = []
    for split in default_split_definitions():
        result = fit_and_evaluate(
            split.training_datasets,
            split.test_datasets,
            observations,
            split_id=split.split_id,
            split_type=split.split_type,
            split_label=split.label,
        )
        results.append(result)
        metrics_rows.extend(result["metrics"])
        prediction_rows.extend(result["predictions"])

    inventory_rows = [
        {
            "dataset_key": key,
            "patient_id": DATASETS[key].patient_id,
            "cycle_id": DATASETS[key].cycle_id,
            "label": DATASETS[key].label,
            "n_observations": len(observations[key]["records"]),
            "planar_dir": str(DATASETS[key].planar_dir),
            "qspect_dir": str(DATASETS[key].qspect_dir),
        }
        for key in DATASETS
    ]
    _write_csv(output_dir / "v3_dataset_inventory.csv", inventory_rows)
    _write_csv(output_dir / "v3_split_metrics.csv", metrics_rows)
    _write_csv(output_dir / "v3_predictions.csv", prediction_rows)
    (output_dir / "v3_report.txt").write_text(
        build_report(observations, metrics_rows), encoding="utf-8"
    )
    plot_pairwise_transfer_matrix(
        metrics_rows, output_dir / "v3_pairwise_transfer_matrix.png"
    )
    plot_leave_one_cycle_out(
        metrics_rows, output_dir / "v3_leave_one_cycle_out.png"
    )
    plot_primary_patient_holdout(
        prediction_rows, output_dir / "v3_primary_p11_to_p8.png"
    )
    return {
        "observations": observations,
        "results": results,
        "metrics": metrics_rows,
        "predictions": prediction_rows,
        "output_dir": output_dir,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--train",
        action="append",
        choices=tuple(DATASETS),
        help="training dataset key; repeat to pool cycles",
    )
    parser.add_argument(
        "--test",
        action="append",
        choices=tuple(DATASETS),
        help="test dataset key; repeat to combine held-out cycles",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if (args.train is None) != (args.test is None):
        raise ValueError("--train and --test must be provided together")
    if args.train is None:
        result = run_default_analysis(args.output_dir)
        print(f"Saved v3 outputs to {result['output_dir']}")
        return

    training, test = validate_split(args.train, args.test)
    observations = load_observation_cache((*training, *test))
    result = fit_and_evaluate(
        training,
        test,
        observations,
        split_id="custom",
        split_type="custom",
        split_label=(
            " + ".join(DATASETS[key].short_label for key in training)
            + " → "
            + " + ".join(DATASETS[key].short_label for key in test)
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "v3_custom_metrics.csv", result["metrics"])
    _write_csv(args.output_dir / "v3_custom_predictions.csv", result["predictions"])
    print(f"Saved custom v3 outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
