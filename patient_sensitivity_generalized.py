"""Automatically discover and validate planar sensitivity datasets.

This is the scalable entry point for the Q/SPECT-target sensitivity models.
It leaves the historical v3 analysis unchanged, discovers every complete
patient/cycle under ``Data/p*/``, and evaluates grouped holdouts without a
random timepoint split.

Default validation levels:

* leave one complete cycle out;
* leave one patient out, pooling all cycles of that patient in the test set.

All v3 rate modes and their eligible M0--M9 models are evaluated.  Q/SPECT is
used only to build and score the target.  CT is not a model input.  The
Q/SPECT-guided crop modes remain development-only preprocessing branches.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figure_layout
import patient_sensitivity_multicycle_v3 as v3
import study_layout


DEFAULT_OUTPUT_DIR = figure_layout.shared_dir(__file__)


def dataset_key(patient: str, cycle: str) -> str:
    """Return a deterministic command-line-safe key for one patient cycle."""
    normalized = re.sub(r"[^a-z0-9]+", "_", f"{patient}_{cycle}".lower())
    return normalized.strip("_")


def _cycle_short_label(patient: str, cycle: str) -> str:
    month_match = re.match(r"(\d{4})-(\d{2})", cycle)
    cycle_label = cycle if month_match is None else f"{month_match.group(1)}-{month_match.group(2)}"
    return f"{patient.upper()} {cycle_label}"


def discover_dataset_specs(
    data_dir: Path = study_layout.DATA_DIR,
    minimum_pairs: int = 2,
) -> tuple[Dict[str, v3.DatasetSpec], list[Dict[str, Any]]]:
    """Discover complete datasets and return an auditable inventory."""
    specs: Dict[str, v3.DatasetSpec] = {}
    inventory_rows: list[Dict[str, Any]] = []
    for cycle in study_layout.discover_cycles(data_dir):
        inventory = study_layout.cycle_inventory(cycle)
        planar_count = int(inventory["planar_timepoints"])
        qspect_count = int(inventory["qspect_complete_timepoints"])
        incomplete_count = int(inventory["qspect_incomplete_timepoints"])
        included = bool(
            planar_count >= minimum_pairs
            and qspect_count >= minimum_pairs
            and incomplete_count == 0
            and planar_count == qspect_count
        )
        reasons = []
        if planar_count < minimum_pairs:
            reasons.append(f"fewer than {minimum_pairs} planar timepoints")
        if qspect_count < minimum_pairs:
            reasons.append(f"fewer than {minimum_pairs} complete Q/SPECT timepoints")
        if incomplete_count:
            reasons.append(f"{incomplete_count} incomplete Q/SPECT series")
        if planar_count != qspect_count:
            reasons.append("planar/QSPECT timepoint count mismatch")

        key = dataset_key(cycle.patient, cycle.cycle)
        inventory_rows.append(
            {
                "dataset_key": key,
                "patient_id": cycle.patient,
                "cycle_id": cycle.cycle,
                "planar_dir": str(cycle.planar_dir),
                "qspect_dir": str(cycle.qspect_dir),
                "planar_timepoints": planar_count,
                "qspect_complete_timepoints": qspect_count,
                "qspect_incomplete_timepoints": incomplete_count,
                "included": included,
                "exclusion_reason": "; ".join(reasons),
            }
        )
        if not included:
            continue
        short_label = _cycle_short_label(cycle.patient, cycle.cycle)
        specs[key] = v3.DatasetSpec(
            key=key,
            patient_id=cycle.patient,
            cycle_id=cycle.cycle,
            label=short_label,
            short_label=short_label,
            planar_dir=cycle.planar_dir,
            qspect_dir=cycle.qspect_dir,
        )
    return specs, inventory_rows


def grouped_splits(
    datasets: Mapping[str, v3.DatasetSpec],
    *,
    include_pairwise: bool = False,
) -> list[v3.SplitDefinition]:
    """Build scalable cycle- and patient-level holdout definitions."""
    keys = tuple(datasets)
    if len(keys) < 2:
        raise ValueError("At least two complete cycles are required")

    splits: list[v3.SplitDefinition] = []
    if include_pairwise:
        for training_key in keys:
            for test_key in keys:
                if training_key == test_key:
                    continue
                splits.append(
                    v3.SplitDefinition(
                        split_id=f"pair_{training_key}_to_{test_key}",
                        split_type="pairwise_cycle_transfer",
                        training_datasets=(training_key,),
                        test_datasets=(test_key,),
                        label=(
                            f"{datasets[training_key].short_label} -> "
                            f"{datasets[test_key].short_label}"
                        ),
                    )
                )

    for test_key in keys:
        training = tuple(key for key in keys if key != test_key)
        splits.append(
            v3.SplitDefinition(
                split_id=f"cycle_holdout_{test_key}",
                split_type="leave_one_cycle_out",
                training_datasets=training,
                test_datasets=(test_key,),
                label=(
                    " + ".join(datasets[key].short_label for key in training)
                    + f" -> {datasets[test_key].short_label}"
                ),
            )
        )

    patients = tuple(sorted({spec.patient_id for spec in datasets.values()}))
    if len(patients) >= 2:
        for patient in patients:
            test = tuple(
                key for key, spec in datasets.items() if spec.patient_id == patient
            )
            training = tuple(
                key for key, spec in datasets.items() if spec.patient_id != patient
            )
            if not training or not test:
                continue
            splits.append(
                v3.SplitDefinition(
                    split_id=f"patient_holdout_{patient}",
                    split_type="leave_one_patient_out",
                    training_datasets=training,
                    test_datasets=test,
                    label=f"Other patients -> {patient.upper()}",
                )
            )
    return splits


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _column_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def build_master_table_rows(
    observations: Mapping[str, Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    """Flatten every model input, target and QC value to one row/timepoint."""
    output: list[Dict[str, Any]] = []
    for dataset_key, observation in observations.items():
        records = observation["records"]
        source_rows = observation.get("source_rows", [{} for _ in records])
        matched_rows = observation.get(
            "matched_measurements", [{} for _ in records]
        )
        reference_activity = np.asarray(
            observation["reference_activity"], dtype=np.float64
        )
        for index, record in enumerate(records):
            source = source_rows[index]
            matched = matched_rows[index]
            row: Dict[str, Any] = {
                **record,
                "observation_index": index,
                "qspect_minus_planar_h": source.get("qspect_minus_planar_h"),
                "qspect_activity_measured_mbq": source.get(
                    "qspect_activity_measured_mbq"
                ),
                "qspect_activity_at_planar_time_mbq": float(
                    reference_activity[index]
                ),
                "physical_time_adjustment_percent": source.get(
                    "physical_time_adjustment_percent"
                ),
                "crop_height_cm": source.get("crop_height_cm"),
                "profile_match_correlation": source.get(
                    "profile_match_correlation"
                ),
                "actual_frame_duration_s": matched.get("duration_seconds"),
                "local_dwell_time_s": matched.get(
                    "local_dwell_time_seconds"
                ),
                "wide_spectrum_ap_counts": (
                    matched.get("wide_spectrum_counts_by_view") or {}
                ).get("AP"),
                "wide_spectrum_pa_counts": (
                    matched.get("wide_spectrum_counts_by_view") or {}
                ).get("PA"),
                "wide_spectrum_ap_cps": (
                    matched.get("wide_spectrum_cps_by_view") or {}
                ).get("AP"),
                "wide_spectrum_pa_cps": (
                    matched.get("wide_spectrum_cps_by_view") or {}
                ).get("PA"),
                "wide_spectrum_mean_cps": matched.get(
                    "wide_spectrum_mean_cps"
                ),
                "dead_time_dtcf": matched.get("dead_time_dtcf", 1.0),
                "dead_time_loss_percent": matched.get(
                    "dead_time_loss_percent", 0.0
                ),
                "dead_time_corrected_in_model": True,
                "ct_used_as_model_input": False,
            }

            for support, energy_values in matched.get(
                "window_counts", {}
            ).items():
                for energy, count_values in energy_values.items():
                    energy_slug = _column_slug(energy)
                    for count_name, value in count_values.items():
                        row[
                            f"counts_{support}_{energy_slug}_{count_name}"
                        ] = value

            for rate_mode, values in observation["rate_modes"].items():
                measurement_rate = np.asarray(
                    values["measurement_rate"], dtype=np.float64
                )
                target = np.asarray(
                    values["target_sensitivity"], dtype=np.float64
                )
                row[f"rate_{rate_mode}_cps_dt_corrected"] = float(
                    measurement_rate[index]
                )
                row[
                    f"effective_sensitivity_{rate_mode}_cps_per_mbq"
                ] = float(target[index])

            for geometry, feature_values in matched.get(
                "features_by_geometry", {}
            ).items():
                for feature_name, value in feature_values.items():
                    row[
                        f"feature_{geometry}_{_column_slug(feature_name)}"
                    ] = value
            output.append(row)
    return output


def build_master_table_report(
    datasets: Mapping[str, v3.DatasetSpec],
    master_rows: Sequence[Mapping[str, Any]],
) -> str:
    """Document the provenance and interpretation of the wide master table."""
    lines = [
        "Analysis master table",
        "=====================",
        "",
        f"Rows: {len(master_rows)} (one paired planar/Q/SPECT timepoint per row)",
        f"Patients: {len({row['patient_id'] for row in master_rows})}",
        f"Cycles: {len(datasets)}",
        "",
        "Included cycles:",
    ]
    for spec in datasets.values():
        count = sum(row["dataset_key"] == spec.key for row in master_rows)
        lines.append(f"- {spec.short_label}: {count} paired timepoints")
    lines.extend(
        [
            "",
            "Core definitions:",
            "- Counts are stored for each of the four energy windows, AP and PA,",
            "  both on the full image and inside the Q/SPECT-guided development crop.",
            "- AP+PA counts are arithmetic sums; GM counts are pixelwise geometric means",
            "  followed by spatial summation.",
            "- RWo is the arithmetic mean of the raw AP and PA 4-window detector rates.",
            "- DTCF is one global paralyzable-model factor per acquisition (tau=0.632 us).",
            "- Every rate_*_dt_corrected column includes this global DTCF.",
            "- Q/SPECT activity is physically decay-adjusted to the planar acquisition time.",
            "- Effective sensitivity = dead-time-corrected planar rate / Q/SPECT activity.",
            "- CT is not a model input.",
            "",
            "Primary-series selection:",
            "- Exact DICOM duplicates are removed before analysis.",
            "- If one Q/SPECT Day label has multiple reconstructions, the complete series",
            "  with the largest number of slices is retained (P11 Day2: 233 rather than 225).",
            "- Planar acquisitions shorter than 80% of the longest ScanLength in the cycle",
            "  are excluded (P11 July: local 390-mm repeat excluded; 2000-mm WB retained).",
            "",
            "Limits:",
            "- Crop-based columns are development-only because the crop is Q/SPECT-guided.",
            "- Full-image columns are the deployable planar-only comparison branches.",
            "- Three patients remain insufficient for definitive external validation.",
            "",
        ]
    )
    return "\n".join(lines)


def plot_patient_holdout_mare(
    metrics_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> list[Path]:
    """Plot one all-model MARE matrix per measurement-rate mode."""
    paths: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for rate_mode, mode_spec in v3.RATE_MODES.items():
        rows = [
            row
            for row in metrics_rows
            if row.get("rate_mode") == rate_mode
            and row.get("split_type") == "leave_one_patient_out"
            and row.get("metric_scope") == "combined_test"
        ]
        if not rows:
            continue
        split_ids = list(dict.fromkeys(str(row["split_id"]) for row in rows))
        model_names = [
            model for model in v3.MODEL_ORDER if model in mode_spec.model_names
        ]
        matrix = np.full((len(split_ids), len(model_names)), np.nan)
        for row in rows:
            if row.get("fit_status") != "fitted" or row["model"] not in model_names:
                continue
            i = split_ids.index(str(row["split_id"]))
            j = model_names.index(str(row["model"]))
            matrix[i, j] = float(row["activity_mare_percent"])

        width = max(9.0, 1.0 + 1.05 * len(model_names))
        height = max(4.4, 1.8 + 0.8 * len(split_ids))
        fig, ax = plt.subplots(figsize=(width, height), facecolor="white")
        finite = matrix[np.isfinite(matrix)]
        vmax = 30.0 if finite.size == 0 else max(10.0, float(np.nanmax(finite)))
        image = ax.imshow(matrix, cmap="YlOrRd", vmin=0.0, vmax=vmax, aspect="auto")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                ax.text(
                    j,
                    i,
                    "--" if not np.isfinite(value) else f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="black",
                )
        model_codes = [model.split("_", 1)[0] for model in model_names]
        ax.set_xticks(range(len(model_names)), model_codes, fontsize=11)
        patient_labels = [split_id.removeprefix("patient_holdout_").upper() for split_id in split_ids]
        ax.set_yticks(range(len(split_ids)), patient_labels, fontsize=10)
        ax.set_xlabel("Modèle")
        ax.set_ylabel("Patient entièrement exclu de l'entraînement")
        ax.set_title(f"Validation interpatient — {mode_spec.short_label}")
        colorbar = fig.colorbar(image, ax=ax, pad=0.02)
        colorbar.set_label("MARE sur l'activité (%)")
        fig.tight_layout()
        path = output_dir / f"patient_holdout_mare_{rate_mode}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_primary_interpatient_mare(
    metrics_rows: Sequence[Mapping[str, Any]], output_dir: Path
) -> Path:
    """Compare the main models across the three fully held-out patients."""
    modes = (
        v3.MODE_TEW_CROP_GM,
        v3.MODE_RAW_CROP_GM,
        v3.MODE_RAW_FULL_GM,
    )
    models = (
        v3.MODEL_CONSTANT,
        v3.MODEL_ORIGIN_SLOPE,
        v3.MODEL_LOCKED,
        v3.MODEL_ASYMMETRY,
        v3.MODEL_PHOTO_ALL,
        v3.MODEL_RAW_ASYMMETRY,
    )
    patient_ids = sorted(
        {
            str(row["split_id"]).removeprefix("patient_holdout_")
            for row in metrics_rows
            if row.get("split_type") == "leave_one_patient_out"
        }
    )
    patient_colors = {
        patient: color
        for patient, color in zip(
            patient_ids, ("#C41230", "#006699", "#5B6770", "#E87722")
        )
    }
    figure, axes = plt.subplots(
        1, len(modes), figsize=(16.0, 5.4), sharey=True, facecolor="white"
    )
    x = np.arange(len(models), dtype=np.float64)
    width = 0.22
    for axis, mode in zip(np.atleast_1d(axes), modes):
        for patient_index, patient in enumerate(patient_ids):
            values = []
            for model in models:
                match = next(
                    (
                        row
                        for row in metrics_rows
                        if row.get("rate_mode") == mode
                        and row.get("split_id") == f"patient_holdout_{patient}"
                        and row.get("metric_scope") == "combined_test"
                        and row.get("fit_status") == "fitted"
                        and row.get("model") == model
                    ),
                    None,
                )
                values.append(
                    np.nan
                    if match is None
                    else float(match["activity_mare_percent"])
                )
            offsets = x + (patient_index - (len(patient_ids) - 1) / 2.0) * width
            bars = axis.bar(
                offsets,
                values,
                width=width,
                color=patient_colors[patient],
                label=patient.upper(),
            )
            for bar, value in zip(bars, values):
                if np.isfinite(value):
                    axis.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height() + 0.7,
                        f"{value:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=90,
                    )
        axis.set_xticks(x, [model.split("_", 1)[0] for model in models])
        axis.set_title(v3.RATE_MODES[mode].short_label, fontsize=12, weight="bold")
        axis.set_xlabel("Modèle")
        axis.grid(axis="y", linestyle="--", alpha=0.30)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[0].set_ylabel("MARE sur l'activité du patient exclu (%)")
    axes[-1].legend(title="Patient test", frameon=False, loc="upper right")
    figure.suptitle(
        "Validation interpatient des principaux modèles",
        fontsize=17,
        weight="bold",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "primary_interpatient_model_mare.png"
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


def plot_primary_interpatient_predictions(
    prediction_rows: Sequence[Mapping[str, Any]], output_dir: Path
) -> Path:
    """Show held-out M2 activities against Q/SPECT for the core modes."""
    modes = (
        v3.MODE_TEW_CROP_GM,
        v3.MODE_RAW_CROP_GM,
        v3.MODE_RAW_FULL_GM,
    )
    selected = [
        row
        for row in prediction_rows
        if row.get("split_type") == "leave_one_patient_out"
        and row.get("model") == v3.MODEL_ASYMMETRY
        and row.get("rate_mode") in modes
    ]
    patient_ids = sorted({str(row["patient_id"]) for row in selected})
    patient_colors = {
        patient: color
        for patient, color in zip(
            patient_ids, ("#C41230", "#006699", "#5B6770", "#E87722")
        )
    }
    figure, axes = plt.subplots(
        1, len(modes), figsize=(15.5, 5.1), sharex=True, sharey=True, facecolor="white"
    )
    all_values = [
        float(row[field])
        for row in selected
        for field in ("qspect_activity_mbq", "predicted_activity_mbq")
    ]
    upper = 1.05 * max(all_values)
    for axis, mode in zip(np.atleast_1d(axes), modes):
        mode_rows = [row for row in selected if row["rate_mode"] == mode]
        for patient in patient_ids:
            patient_rows = [
                row for row in mode_rows if str(row["patient_id"]) == patient
            ]
            axis.scatter(
                [float(row["qspect_activity_mbq"]) for row in patient_rows],
                [float(row["predicted_activity_mbq"]) for row in patient_rows],
                s=48,
                color=patient_colors[patient],
                label=patient.upper(),
                alpha=0.90,
            )
        axis.plot([0.0, upper], [0.0, upper], color="black", linestyle="--", linewidth=1.3)
        errors = [abs(float(row["activity_error_percent"])) for row in mode_rows]
        axis.text(
            0.04,
            0.94,
            f"MARE global = {np.mean(errors):.1f} %",
            transform=axis.transAxes,
            va="top",
            fontsize=10,
        )
        axis.set_xlim(0.0, upper)
        axis.set_ylim(0.0, upper)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(v3.RATE_MODES[mode].short_label, fontsize=12, weight="bold")
        axis.set_xlabel("Activité Q/SPECT (MBq)")
        axis.grid(True, linestyle="--", alpha=0.25)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[0].set_ylabel("Activité planaire prédite (MBq)")
    axes[-1].legend(title="Patient test", frameon=False)
    figure.suptitle(
        "Prédictions interpatient sans réajustement — M2",
        fontsize=17,
        weight="bold",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "primary_interpatient_predictions_m2.png"
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


def build_report(
    datasets: Mapping[str, v3.DatasetSpec],
    inventory_rows: Sequence[Mapping[str, Any]],
    splits: Sequence[v3.SplitDefinition],
    metrics_rows: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "Generalized patient-sensitivity model audit",
        "===========================================",
        "",
        f"Complete included cycles: {len(datasets)}",
        f"Patients represented: {len({spec.patient_id for spec in datasets.values()})}",
        f"Grouped validation splits: {len(splits)}",
        "",
        "Included datasets:",
    ]
    for spec in datasets.values():
        lines.append(f"- {spec.key}: {spec.short_label}")
    excluded = [row for row in inventory_rows if not bool(row["included"])]
    lines.extend(["", "Excluded datasets:"])
    if excluded:
        for row in excluded:
            lines.append(
                f"- {row['dataset_key']}: {row['exclusion_reason']}"
            )
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "Leave-one-patient-out MARE:",
        ]
    )
    patient_rows = [
        row
        for row in metrics_rows
        if row.get("split_type") == "leave_one_patient_out"
        and row.get("metric_scope") == "combined_test"
        and row.get("fit_status") == "fitted"
    ]
    for rate_mode, mode_spec in v3.RATE_MODES.items():
        lines.append(f"- {mode_spec.short_label}")
        for row in patient_rows:
            if row.get("rate_mode") != rate_mode:
                continue
            patient = str(row["split_id"]).removeprefix("patient_holdout_").upper()
            lines.append(
                f"    {patient} | {row['model']} | "
                f"MARE={float(row['activity_mare_percent']):.2f}% | "
                f"n_test={int(row['n_test'])}"
            )
    lines.extend(["", "Primary observation-weighted interpatient MARE:"])
    primary_modes = (
        v3.MODE_TEW_CROP_GM,
        v3.MODE_RAW_CROP_GM,
        v3.MODE_RAW_FULL_GM,
    )
    primary_models = (
        v3.MODEL_CONSTANT,
        v3.MODEL_ORIGIN_SLOPE,
        v3.MODEL_LOCKED,
        v3.MODEL_ASYMMETRY,
    )
    for rate_mode in primary_modes:
        lines.append(f"- {v3.RATE_MODES[rate_mode].short_label}")
        for model in primary_models:
            selected = [
                row
                for row in patient_rows
                if row.get("rate_mode") == rate_mode
                and row.get("model") == model
            ]
            total_n = sum(int(row["n_test"]) for row in selected)
            if total_n == 0:
                continue
            weighted_mare = sum(
                float(row["activity_mare_percent"]) * int(row["n_test"])
                for row in selected
            ) / total_n
            lines.append(
                f"    {model} | MARE={weighted_mare:.2f}% | n={total_n}"
            )
    lines.extend(
        [
            "",
            "Interpretation limits:",
            "- All measurement rates include the global dead-time correction.",
            "- Timepoints from one patient are never split randomly across train/test.",
            "- Leave-one-patient-out is the main generalization estimate.",
            "- Leave-one-cycle-out is secondary when another cycle of the same patient remains in training.",
            "- Comparing many models on the same held-out patients is exploratory model selection, not final validation.",
            "- Crop modes still use Q/SPECT-guided preprocessing and are not deployable planar-only modes.",
            "",
        ]
    )
    return "\n".join(lines)


def run_generalized_analysis(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    data_dir: Path = study_layout.DATA_DIR,
    minimum_pairs: int = 2,
    include_pairwise: bool = False,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets, inventory_rows = discover_dataset_specs(data_dir, minimum_pairs)
    if len(datasets) < 2:
        raise ValueError("Fewer than two complete datasets were discovered")

    # v3's fitting helpers validate keys against this registry.  Replacing the
    # historical registry is local to this command's Python process.
    v3.DATASETS = dict(datasets)
    splits = grouped_splits(datasets, include_pairwise=include_pairwise)
    base_observations = {
        key: v3.build_dataset_observations(spec) for key, spec in datasets.items()
    }
    master_rows = build_master_table_rows(base_observations)

    all_metrics: list[Dict[str, Any]] = []
    all_predictions: list[Dict[str, Any]] = []
    for rate_mode, mode_spec in v3.RATE_MODES.items():
        observations = v3.select_rate_mode_observations(base_observations, rate_mode)
        for split in splits:
            result = v3.fit_and_evaluate(
                split.training_datasets,
                split.test_datasets,
                observations,
                split_id=split.split_id,
                split_type=split.split_type,
                split_label=split.label,
                rate_mode=rate_mode,
                model_names=mode_spec.model_names,
                include_j0_personalized=True,
                reference_name="activité Q/SPECT",
            )
            all_metrics.extend(result["metrics"])
            all_predictions.extend(result["predictions"])

    _write_csv(output_dir / "dataset_inventory.csv", inventory_rows)
    master_table_path = _write_csv(
        output_dir / "analysis_master_table.csv", master_rows
    )
    master_table_report_path = output_dir / "analysis_master_table_methodology.txt"
    master_table_report_path.write_text(
        build_master_table_report(datasets, master_rows), encoding="utf-8"
    )
    _write_csv(output_dir / "grouped_split_definitions.csv", [asdict(split) for split in splits])
    _write_csv(output_dir / "all_model_metrics.csv", all_metrics)
    _write_csv(output_dir / "all_model_predictions.csv", all_predictions)
    figure_paths = plot_patient_holdout_mare(all_metrics, output_dir / "patient_holdout")
    primary_dir = output_dir / "primary_validation"
    figure_paths.extend(
        [
            plot_primary_interpatient_mare(all_metrics, primary_dir),
            plot_primary_interpatient_predictions(
                all_predictions, primary_dir
            ),
        ]
    )
    post_j0_dir = output_dir / "post_J0_comparison"
    figure_paths.extend(
        v3.plot_post_j0_model_mare(
            all_metrics,
            rate_mode,
            post_j0_dir / f"post_J0_MARE_{rate_mode}.png",
            model_names=mode_spec.model_names,
            title_prefix="Cible Q/SPECT — comparaison après J0",
        )
        for rate_mode, mode_spec in v3.RATE_MODES.items()
    )
    report_path = output_dir / "generalized_model_report.txt"
    report_path.write_text(
        build_report(datasets, inventory_rows, splits, all_metrics),
        encoding="utf-8",
    )
    methodology_path = output_dir / "modeles_methodologie_commune.txt"
    methodology_path.write_text(
        v3.build_model_methodology_document(), encoding="utf-8"
    )
    return {
        "datasets": datasets,
        "inventory": inventory_rows,
        "splits": splits,
        "metrics": all_metrics,
        "predictions": all_predictions,
        "master_rows": master_rows,
        "master_table_path": master_table_path,
        "master_table_report_path": master_table_report_path,
        "figures": figure_paths,
        "report_path": report_path,
        "methodology_path": methodology_path,
        "output_dir": output_dir,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=study_layout.DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--minimum-pairs", type=int, default=2)
    parser.add_argument(
        "--include-pairwise",
        action="store_true",
        help="also evaluate every directed one-cycle-to-one-cycle transfer",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_generalized_analysis(
        args.output_dir,
        data_dir=args.data_dir,
        minimum_pairs=args.minimum_pairs,
        include_pairwise=args.include_pairwise,
    )
    print(
        f"Included {len(result['datasets'])} cycles; "
        f"saved generalized outputs to {result['output_dir']}"
    )


if __name__ == "__main__":
    main()
