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
    lines.extend(
        [
            "",
            "Interpretation limits:",
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
            )
            all_metrics.extend(result["metrics"])
            all_predictions.extend(result["predictions"])

    _write_csv(output_dir / "dataset_inventory.csv", inventory_rows)
    _write_csv(output_dir / "grouped_split_definitions.csv", [asdict(split) for split in splits])
    _write_csv(output_dir / "all_model_metrics.csv", all_metrics)
    _write_csv(output_dir / "all_model_predictions.csv", all_predictions)
    figure_paths = plot_patient_holdout_mare(all_metrics, output_dir / "patient_holdout")
    report_path = output_dir / "generalized_model_report.txt"
    report_path.write_text(
        build_report(datasets, inventory_rows, splits, all_metrics),
        encoding="utf-8",
    )
    return {
        "datasets": datasets,
        "inventory": inventory_rows,
        "splits": splits,
        "metrics": all_metrics,
        "predictions": all_predictions,
        "figures": figure_paths,
        "report_path": report_path,
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
