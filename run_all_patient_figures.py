"""Run patient-specific figure scripts over every complete dataset cycle.

Each child script writes to:
    fig/<script>/patient X/<cycle folder>/

Cycles with missing planar/Q/SPECT timepoints are reported and skipped instead
of being silently mixed with another exam.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import figure_layout
import study_layout


PROJECT_DIR = Path(__file__).resolve().parent
RUN_REPORT_DIR = figure_layout.FIG_ROOT / "_run_reports"


@dataclass(frozen=True)
class Analysis:
    name: str
    script: str
    minimum_pairs: int = 2


ANALYSES = (
    Analysis("acquisition_timeline", "acquisition_timeline.py"),
    Analysis("patient_effective_sensitivity", "patient_effective_sensitivity.py"),
    Analysis("planar_qspect_profile_crop", "planar_qspect_profile_crop.py"),
    Analysis("attenuation_correction", "attenuation_correction.py"),
    Analysis("compare_crop_methods", "compare_crop_methods.py"),
    Analysis("compare_ct_conversion_methods", "compare_ct_conversion_methods.py"),
    Analysis("compare_ct_planar_simpleitk", "compare_ct_planar_simpleitk.py"),
    Analysis("compare_tew_pixel_global", "compare_tew_pixel_global.py"),
    Analysis("crop_excluded_counts_test", "crop_excluded_counts_test.py"),
    Analysis("compare_ap_pa_alignment_options", "compare_ap_pa_alignment_options.py"),
    Analysis("plot_ap_pa_alignment_test", "plot_ap_pa_alignment_test.py"),
    Analysis("plot_attenuation_windows", "plots/plot_attenuation_windows.py"),
    Analysis(
        "generate_planar_dead_time_report",
        "reports/generate_planar_dead_time_report.py",
    ),
    Analysis("compare_planar_qspect", "compare_planar_qspect.py", 4),
    Analysis("attenuation_global_model_v2", "attenuation_global_model_v2.py", 4),
    Analysis("attenuation_window_model", "attenuation_window_model.py", 4),
    Analysis("patient_sensitivity_feature_test", "patient_sensitivity_feature_test.py", 4),
)


BOOTSTRAP = """
import runpy
import sys
from pathlib import Path
import planar_processing
import qspect_processing

planar_dir = Path(sys.argv[1])
qspect_dir = Path(sys.argv[2])
script = Path(sys.argv[3])
planar_processing.default_planar_study_dir = lambda: planar_dir
qspect_processing.default_qspect_dir = lambda: qspect_dir
sys.argv = [str(script)]
runpy.run_path(str(script), run_name="__main__")
"""


def _selected_analyses(names: Iterable[str] | None) -> List[Analysis]:
    if not names:
        return list(ANALYSES)
    requested = set(names)
    known = {analysis.name for analysis in ANALYSES}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"Unknown analyses: {', '.join(unknown)}")
    return [analysis for analysis in ANALYSES if analysis.name in requested]


def _run_analysis(
    analysis: Analysis,
    dataset: study_layout.CycleDataset,
    dry_run: bool,
) -> dict:
    inventory = study_layout.cycle_inventory(dataset)
    output_dir = figure_layout.patient_cycle_dir(
        analysis.script, dataset.patient, dataset.cycle
    )
    result = {
        "analysis": analysis.name,
        "patient": dataset.patient,
        "cycle": dataset.cycle,
        "planar_timepoints": inventory["planar_timepoints"],
        "complete_qspect_timepoints": inventory["qspect_complete_timepoints"],
        "incomplete_qspect_timepoints": inventory["qspect_incomplete_timepoints"],
        "status": "",
        "return_code": "",
        "output_dir": str(output_dir),
    }
    enough_pairs = (
        bool(inventory["complete_for_full_pipeline"])
        and int(inventory["qspect_complete_timepoints"]) >= analysis.minimum_pairs
    )
    if not enough_pairs:
        result["status"] = "skipped_incomplete_or_insufficient"
        return result
    if dry_run:
        log_path = output_dir / "run.log"
        if log_path.exists() and "return_code=0" in log_path.read_text(
            encoding="utf-8", errors="replace"
        ):
            result["status"] = "completed_existing"
            result["return_code"] = 0
        else:
            result["status"] = "ready"
        return result

    output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["FIG_PATIENT"] = dataset.patient
    environment["FIG_CYCLE"] = dataset.cycle
    environment.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig")
    environment.setdefault("PYTHONPYCACHEPREFIX", "/private/tmp/codex_pycache")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            BOOTSTRAP,
            str(dataset.planar_dir),
            str(dataset.qspect_dir),
            str(PROJECT_DIR / analysis.script),
        ],
        cwd=PROJECT_DIR,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    log = [
        f"analysis={analysis.name}",
        f"patient={dataset.patient}",
        f"cycle={dataset.cycle}",
        f"return_code={completed.returncode}",
        "",
        "STDOUT",
        completed.stdout,
        "STDERR",
        completed.stderr,
    ]
    (output_dir / "run.log").write_text("\n".join(log), encoding="utf-8")
    result["return_code"] = completed.returncode
    result["status"] = "completed" if completed.returncode == 0 else "failed"
    return result


def run_all(
    analysis_names: Iterable[str] | None = None,
    patients: Iterable[str] | None = None,
    cycles: Iterable[str] | None = None,
    dry_run: bool = False,
) -> List[dict]:
    analyses = _selected_analyses(analysis_names)
    patient_filter = None if not patients else {value.lower() for value in patients}
    cycle_filter = None if not cycles else set(cycles)
    datasets = [
        dataset
        for dataset in study_layout.discover_cycles()
        if (patient_filter is None or dataset.patient.lower() in patient_filter)
        and (cycle_filter is None or dataset.cycle in cycle_filter)
    ]
    rows = []
    for dataset in datasets:
        for analysis in analyses:
            print(f"[{dataset.patient}/{dataset.cycle}] {analysis.name}", flush=True)
            row = _run_analysis(analysis, dataset, dry_run=dry_run)
            print(f"  -> {row['status']}", flush=True)
            rows.append(row)

    RUN_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RUN_REPORT_DIR / "patient_cycle_figure_runs.csv"
    if rows:
        with report_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Run report: {report_path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", action="append", dest="analyses")
    parser.add_argument("--patient", action="append", dest="patients")
    parser.add_argument("--cycle", action="append", dest="cycles")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_all(args.analyses, args.patients, args.cycles, args.dry_run)


if __name__ == "__main__":
    main()
