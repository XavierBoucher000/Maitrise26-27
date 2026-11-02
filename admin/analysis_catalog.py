"""Canonical registry for project scripts, roles, and figure categories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from core.project_paths import PROJECT_ROOT


STATUS_ACTIVE = "active"
STATUS_SHARED = "shared_library"
STATUS_EXPLORATORY = "exploratory"
STATUS_LEGACY_DEPENDENCY = "legacy_dependency"
STATUS_ARCHIVE_CANDIDATE = "archive_candidate"
STATUS_PRESENTATION = "presentation"
STATUS_ORCHESTRATION = "orchestration"


@dataclass(frozen=True)
class AnalysisSpec:
    name: str
    path: str
    category: tuple[str, ...]
    figure_category: tuple[str, ...] | None
    status: str
    role: str
    minimum_pairs: int = 0
    run_per_complete_cycle: bool = False
    replacement: str = ""
    note: str = ""


def _spec(
    name: str,
    category: tuple[str, ...],
    status: str,
    role: str,
    *,
    path: str | None = None,
    minimum_pairs: int = 0,
    run_per_cycle: bool = False,
    replacement: str = "",
    note: str = "",
) -> AnalysisSpec:
    if category[0] in {"core", "admin"}:
        figure_category = None
    elif category[0] in {"models", "retention_excretion"}:
        # Logical subgroups do not change the established figure paths.
        figure_category = (category[0],)
    else:
        figure_category = category
    if path is None:
        directory = category
        if category[:2] == ("models", "legacy_dependencies"):
            directory = ("models", "legacy")
        elif category[:2] == ("models", "archive_candidates"):
            directory = ("models", "legacy")
        elif category[:2] == ("models", "tests_exploratoires"):
            directory = ("models", "exploratory")
        elif category[:2] == ("models", "presentation_summary"):
            directory = ("models", "presentation")
        elif category[0] == "retention_excretion":
            directory = ("retention_excretion",)
        path = "/".join((*directory, f"{name}.py"))
    return AnalysisSpec(
        name=name,
        path=path,
        category=category,
        figure_category=figure_category,
        status=status,
        role=role,
        minimum_pairs=minimum_pairs,
        run_per_complete_cycle=run_per_cycle,
        replacement=replacement,
        note=note,
    )


ANALYSES: tuple[AnalysisSpec, ...] = (
    # Shared infrastructure -------------------------------------------------
    _spec("dicom_loader", ("core",), STATUS_SHARED, "Lecture DICOM planaire"),
    _spec("dicom_series_dedup", ("core",), STATUS_SHARED, "Détection des séries dupliquées"),
    _spec("planar_processing", ("core",), STATUS_SHARED, "Traitement planaire commun"),
    _spec("qspect_processing", ("core",), STATUS_SHARED, "Traitement Q/SPECT commun"),
    _spec("study_layout", ("core",), STATUS_SHARED, "Découverte patients et cycles"),
    _spec("figure_layout", ("core",), STATUS_SHARED, "Routage canonique des figures"),
    _spec("project_paths", ("core",), STATUS_SHARED, "Chemins canoniques Data et fig"),
    _spec("correction_3DEW", ("corrections", "scatter"), STATUS_SHARED, "Correction TEW/3DEW"),
    _spec("dead_time", ("corrections", "dead_time"), STATUS_SHARED, "Formules de temps mort"),
    _spec("ct_attenuation_correction", ("corrections", "ct_attenuation"), STATUS_SHARED, "Conversion CT et carte d’atténuation"),
    _spec("planar_qspect_crop", ("crop_alignment",), STATUS_SHARED, "Crop fixe et utilitaires historiques"),

    # Corrections -----------------------------------------------------------
    _spec("attenuation_correction", ("corrections", "ct_attenuation"), STATUS_ACTIVE, "Pipeline CTAC planaire", minimum_pairs=2, run_per_cycle=True),
    _spec("compare_ct_conversion_methods", ("corrections", "ct_attenuation"), STATUS_ACTIVE, "Comparaison des conversions HU", minimum_pairs=2, run_per_cycle=True),
    _spec("compare_ct_planar_simpleitk", ("corrections", "ct_attenuation"), STATUS_EXPLORATORY, "Recalage CT-planaire SimpleITK", minimum_pairs=2, run_per_cycle=True),
    _spec("compare_tew_pixel_global", ("corrections", "scatter"), STATUS_ACTIVE, "Comparaison TEW pixel/global", minimum_pairs=2, run_per_cycle=True),
    _spec("plot_attenuation_windows", ("corrections", "scatter"), STATUS_ACTIVE, "Contrôle des fenêtres d’énergie", minimum_pairs=2, run_per_cycle=True),
    _spec("plot_177lu_spectrum_windows", ("corrections", "scatter"), STATUS_ACTIVE, "Schéma des fenêtres Lu-177"),
    _spec("generate_planar_dead_time_report", ("corrections", "dead_time"), STATUS_ACTIVE, "Rapport des taux et du temps mort", minimum_pairs=2, run_per_cycle=True),

    # Crop and alignment ----------------------------------------------------
    _spec("planar_qspect_profile_crop", ("crop_alignment",), STATUS_ACTIVE, "Crop planaire par profils Q/SPECT", minimum_pairs=2, run_per_cycle=True),
    _spec("compare_crop_methods", ("crop_alignment",), STATUS_ACTIVE, "Comparaison fixe/profil", minimum_pairs=2, run_per_cycle=True),
    _spec("compare_ap_pa_alignment_options", ("crop_alignment",), STATUS_ACTIVE, "Orientation AP/PA", minimum_pairs=2, run_per_cycle=True),
    _spec("plot_ap_pa_alignment_test", ("crop_alignment",), STATUS_ACTIVE, "Contrôle visuel AP/PA", minimum_pairs=2, run_per_cycle=True),
    _spec("crop_excluded_counts_test", ("crop_alignment",), STATUS_EXPLORATORY, "Comptes exclus par le crop", minimum_pairs=2, run_per_cycle=True),
    _spec("plot_planar_qspect_coverage", ("crop_alignment",), STATUS_ACTIVE, "Couverture planaire/Q/SPECT"),
    _spec("qspect_viewer", ("quality_control",), STATUS_EXPLORATORY, "Visualiseur Q/SPECT manuel à paramètres locaux", note="Conserve des chemins configurés dans le fichier; ne pas lancer automatiquement."),

    # Current model engines and analyses -----------------------------------
    _spec("patient_effective_sensitivity", ("models",), STATUS_ACTIVE, "Sensibilité effective descriptive", minimum_pairs=2, run_per_cycle=True),
    _spec("patient_sensitivity_multicycle_v3", ("models", "engines"), STATUS_SHARED, "Moteur commun M0–M9"),
    _spec("patient_sensitivity_generalized", ("models",), STATUS_ACTIVE, "Validation généralisée avec cible Q/SPECT"),
    _spec("patient_sensitivity_qspect_slopes", ("models",), STATUS_ACTIVE, "Sensibilités et pentes avec cible Q/SPECT"),
    _spec("patient_sensitivity_urine_generalized", ("models",), STATUS_ACTIVE, "Sensibilités et pentes avec cible urinaire"),
    _spec("patient_sensitivity_reference_comparison", ("models",), STATUS_ACTIVE, "Comparaison urine/Q/SPECT"),
    _spec("patient_sensitivity_qspect_all_timepoints_test", ("models", "tests_exploratoires"), STATUS_EXPLORATORY, "Test Q/SPECT avec tous les timepoints"),

    # Historical model modules still imported by current code --------------
    _spec("attenuation_window_model", ("models", "legacy_dependencies"), STATUS_LEGACY_DEPENDENCY, "Ancien modèle par fenêtres", replacement="patient_sensitivity_multicycle_v3", note="Encore importé par attenuation_global_model_v2."),
    _spec("attenuation_global_model_v2", ("models", "legacy_dependencies"), STATUS_LEGACY_DEPENDENCY, "Ancien modèle global v2", replacement="patient_sensitivity_multicycle_v3", note="Encore importé par le moteur v3."),
    _spec("patient_sensitivity_feature_test_v2", ("models", "legacy_dependencies"), STATUS_LEGACY_DEPENDENCY, "Preuve de concept v2", replacement="patient_sensitivity_generalized", note="Certaines fonctions sont encore importées par v3."),
    _spec("patient_sensitivity_urine_target", ("models", "legacy_dependencies"), STATUS_LEGACY_DEPENDENCY, "Constructeur historique de la cible urinaire", replacement="patient_sensitivity_urine_generalized", note="Le constructeur de données reste utilisé par l’analyse généralisée."),

    # Retention and excretion -----------------------------------------------
    _spec("sang_urine", ("retention_excretion", "core"), STATUS_SHARED, "Lecture et bilan urine/sang"),
    _spec("planar_total_counts_retention", ("retention_excretion",), STATUS_ACTIVE, "Comptes planaires et rétention urinaire"),
    _spec("excretion_calibration_test", ("retention_excretion", "exploratory"), STATUS_EXPLORATORY, "Ancien test de calibration urinaire"),
    _spec("planar_excretion_fit_balance", ("retention_excretion", "exploratory"), STATUS_EXPLORATORY, "Fits réciproques planaire/excrétion"),
    _spec("imaging_excretion_fit_balance", ("retention_excretion", "exploratory"), STATUS_EXPLORATORY, "Fits réciproques imagerie/excrétion"),

    # Validation and quality control ---------------------------------------
    _spec("compare_planar_qspect", ("validation",), STATUS_ACTIVE, "Validation planaire versus Q/SPECT", minimum_pairs=4, run_per_cycle=True),
    _spec("acquisition_timeline", ("quality_control",), STATUS_ACTIVE, "Dates et heures d’acquisition", minimum_pairs=2, run_per_cycle=True),
    _spec("view_patient_images", ("quality_control",), STATUS_ACTIVE, "Aperçu des séries patient"),
    _spec("generate_dicom_metadata_headers", ("quality_control",), STATUS_ACTIVE, "Export des métadonnées DICOM"),
    _spec("report_exact_dicom_duplicates", ("quality_control",), STATUS_ACTIVE, "Rapport des doublons exacts"),

    # Presentation and orchestration ---------------------------------------
    _spec("presentation_model_figures", ("models", "presentation_summary"), STATUS_PRESENTATION, "Figures synthèse des modèles"),
    _spec("presentation_slope_constraint_figures", ("models", "presentation_summary"), STATUS_PRESENTATION, "Figures pentes, urine et décroissance"),
    _spec("run_all_patient_figures", ("admin",), STATUS_ORCHESTRATION, "Exécuteur multi-patient"),
    _spec("analysis_catalog", ("admin",), STATUS_ORCHESTRATION, "Catalogue canonique des analyses"),
    _spec("main", ("admin",), STATUS_ORCHESTRATION, "Lanceur DICOM générique"),
)


BY_NAME = {spec.name: spec for spec in ANALYSES}
FIGURE_CATEGORY_BY_NAME = {
    spec.name: spec.figure_category
    for spec in ANALYSES
    if spec.figure_category is not None
}
FIGURE_CATEGORY_BY_NAME.update(
    {
        # Historical names retained for stable output paths.
        "metadonne": ("quality_control",),
        "metadonné": ("quality_control",),
    }
)

CYCLE_ANALYSIS_ORDER = (
    "acquisition_timeline",
    "patient_effective_sensitivity",
    "planar_qspect_profile_crop",
    "attenuation_correction",
    "compare_crop_methods",
    "compare_ct_conversion_methods",
    "compare_ct_planar_simpleitk",
    "compare_tew_pixel_global",
    "crop_excluded_counts_test",
    "compare_ap_pa_alignment_options",
    "plot_ap_pa_alignment_test",
    "plot_attenuation_windows",
    "generate_planar_dead_time_report",
    "compare_planar_qspect",
)


def cycle_analyses() -> tuple[AnalysisSpec, ...]:
    """Return analyses intentionally executed for every complete cycle."""
    selected = {
        spec.name: spec for spec in ANALYSES if spec.run_per_complete_cycle
    }
    missing = set(selected) - set(CYCLE_ANALYSIS_ORDER)
    if missing:
        raise ValueError(
            "Per-cycle analyses missing from CYCLE_ANALYSIS_ORDER: "
            + ", ".join(sorted(missing))
        )
    return tuple(selected[name] for name in CYCLE_ANALYSIS_ORDER if name in selected)


def by_status(status: str) -> tuple[AnalysisSpec, ...]:
    return tuple(spec for spec in ANALYSES if spec.status == status)


def render_markdown(specs: Iterable[AnalysisSpec] = ANALYSES) -> str:
    """Render an auditable organization report from the registry."""
    ordered = list(specs)
    lines = [
        "# Organisation du code d’analyse",
        "",
        "Cette structure physique reproduit les catégories du dossier `fig/`.",
        "Les imports utilisent maintenant les paquets `core`, `corrections`,",
        "`crop_alignment`, `models`, `retention_excretion`, `validation`,",
        "`quality_control` et `admin`. Le catalogue est la",
        "source unique pour le routage des figures et l’exécution multi-patient.",
        "",
        "## Signification des statuts",
        "",
        "- `active`: analyse actuelle à conserver.",
        "- `shared_library`: fonctions communes utilisées par plusieurs analyses.",
        "- `exploratory`: expérience utile, mais non primaire.",
        "- `legacy_dependency`: ancien module encore importé; ne pas déplacer seul.",
        "- `archive_candidate`: remplacé et non requis par le pipeline actuel.",
        "- `presentation`: figures de synthèse seulement.",
        "- `orchestration`: découverte et exécution des analyses.",
        "",
        "## Catalogue",
        "",
        "| Catégorie | Fichier | Statut | Rôle | Remplacement / note |",
        "|---|---|---|---|---|",
    ]
    for spec in ordered:
        category = "/".join(spec.category)
        detail = " — ".join(value for value in (spec.replacement, spec.note) if value)
        lines.append(
            f"| `{category}` | `{spec.path}` | `{spec.status}` | "
            f"{spec.role} | {detail} |"
        )
    lines.extend(
        [
            "",
            "## Exécution après la réorganisation",
            "",
            "Lancer les modules depuis la racine du projet avec `python -m`:",
            "",
            "- toutes les figures: `.venv/bin/python -m admin.run_all_patient_figures`;",
            "- modèle Q/SPECT: `.venv/bin/python -m models.patient_sensitivity_generalized`;",
            "- pentes Q/SPECT: `.venv/bin/python -m models.patient_sensitivity_qspect_slopes`;",
            "- rétention urinaire: `.venv/bin/python -m retention_excretion.planar_total_counts_retention`;",
            "- rapport de temps mort: `.venv/bin/python -m corrections.dead_time.generate_planar_dead_time_report`.",
            "",
            "## Candidats à l’archivage",
            "",
        ]
    )
    archive_candidates = by_status(STATUS_ARCHIVE_CANDIDATE)
    if archive_candidates:
        for spec in archive_candidates:
            lines.append(f"- `{spec.path}` → {spec.replacement}")
    else:
        lines.append("- Aucun candidat immédiat; l’étape 1 a été nettoyée.")
    lines.extend(
        [
            "",
            "## Regroupements recommandés",
            "",
            "1. Extraire les fonctions encore utilisées de `attenuation_window_model.py`,",
            "   `attenuation_global_model_v2.py` et `patient_sensitivity_feature_test_v2.py`",
            "   vers un moteur partagé sous `models/engines`.",
            "2. Réduire `patient_sensitivity_urine_target.py` au chargement de la cible",
            "   urinaire; conserver les analyses dans `patient_sensitivity_urine_generalized.py`.",
            "3. Après extraction, réévaluer les dépendances restantes avant toute",
            "   nouvelle suppression.",
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown(path: Path | str | None = None) -> Path:
    output = PROJECT_ROOT / "CODE_ORGANIZATION.md" if path is None else Path(path)
    output.write_text(render_markdown() + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    print(write_markdown())
