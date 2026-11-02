# Organisation du code d’analyse

Cette structure physique reproduit les catégories du dossier `fig/`.
Les imports utilisent maintenant les paquets `core`, `corrections`,
`crop_alignment`, `models`, `retention_excretion`, `validation`,
`quality_control` et `admin`. Le catalogue est la
source unique pour le routage des figures et l’exécution multi-patient.

## Signification des statuts

- `active`: analyse actuelle à conserver.
- `shared_library`: fonctions communes utilisées par plusieurs analyses.
- `exploratory`: expérience utile, mais non primaire.
- `legacy_dependency`: ancien module encore importé; ne pas déplacer seul.
- `archive_candidate`: remplacé et non requis par le pipeline actuel.
- `presentation`: figures de synthèse seulement.
- `orchestration`: découverte et exécution des analyses.

## Catalogue

| Catégorie | Fichier | Statut | Rôle | Remplacement / note |
|---|---|---|---|---|
| `core` | `core/dicom_loader.py` | `shared_library` | Lecture DICOM planaire |  |
| `core` | `core/dicom_series_dedup.py` | `shared_library` | Détection des séries dupliquées |  |
| `core` | `core/planar_processing.py` | `shared_library` | Traitement planaire commun |  |
| `core` | `core/qspect_processing.py` | `shared_library` | Traitement Q/SPECT commun |  |
| `core` | `core/study_layout.py` | `shared_library` | Découverte patients et cycles |  |
| `core` | `core/figure_layout.py` | `shared_library` | Routage canonique des figures |  |
| `core` | `core/project_paths.py` | `shared_library` | Chemins canoniques Data et fig |  |
| `corrections/scatter` | `corrections/scatter/correction_3DEW.py` | `shared_library` | Correction TEW/3DEW |  |
| `corrections/dead_time` | `corrections/dead_time/dead_time.py` | `shared_library` | Formules de temps mort |  |
| `corrections/ct_attenuation` | `corrections/ct_attenuation/ct_attenuation_correction.py` | `shared_library` | Conversion CT et carte d’atténuation |  |
| `crop_alignment` | `crop_alignment/planar_qspect_crop.py` | `shared_library` | Crop fixe et utilitaires historiques |  |
| `corrections/ct_attenuation` | `corrections/ct_attenuation/attenuation_correction.py` | `active` | Pipeline CTAC planaire |  |
| `corrections/ct_attenuation` | `corrections/ct_attenuation/compare_ct_conversion_methods.py` | `active` | Comparaison des conversions HU |  |
| `corrections/ct_attenuation` | `corrections/ct_attenuation/compare_ct_planar_simpleitk.py` | `exploratory` | Recalage CT-planaire SimpleITK |  |
| `corrections/scatter` | `corrections/scatter/compare_tew_pixel_global.py` | `active` | Comparaison TEW pixel/global |  |
| `corrections/scatter` | `corrections/scatter/plot_attenuation_windows.py` | `active` | Contrôle des fenêtres d’énergie |  |
| `corrections/scatter` | `corrections/scatter/plot_177lu_spectrum_windows.py` | `active` | Schéma des fenêtres Lu-177 |  |
| `corrections/dead_time` | `corrections/dead_time/generate_planar_dead_time_report.py` | `active` | Rapport des taux et du temps mort |  |
| `crop_alignment` | `crop_alignment/planar_qspect_profile_crop.py` | `active` | Crop planaire par profils Q/SPECT |  |
| `crop_alignment` | `crop_alignment/compare_crop_methods.py` | `active` | Comparaison fixe/profil |  |
| `crop_alignment` | `crop_alignment/compare_ap_pa_alignment_options.py` | `active` | Orientation AP/PA |  |
| `crop_alignment` | `crop_alignment/plot_ap_pa_alignment_test.py` | `active` | Contrôle visuel AP/PA |  |
| `crop_alignment` | `crop_alignment/crop_excluded_counts_test.py` | `exploratory` | Comptes exclus par le crop |  |
| `crop_alignment` | `crop_alignment/plot_planar_qspect_coverage.py` | `active` | Couverture planaire/Q/SPECT |  |
| `quality_control` | `quality_control/qspect_viewer.py` | `exploratory` | Visualiseur Q/SPECT manuel à paramètres locaux | Conserve des chemins configurés dans le fichier; ne pas lancer automatiquement. |
| `models` | `models/patient_effective_sensitivity.py` | `active` | Sensibilité effective descriptive |  |
| `models/engines` | `models/engines/patient_sensitivity_multicycle_v3.py` | `shared_library` | Moteur commun M0–M9 |  |
| `models` | `models/patient_sensitivity_generalized.py` | `active` | Validation généralisée avec cible Q/SPECT |  |
| `models` | `models/patient_sensitivity_qspect_slopes.py` | `active` | Sensibilités et pentes avec cible Q/SPECT |  |
| `models` | `models/patient_sensitivity_urine_generalized.py` | `active` | Sensibilités et pentes avec cible urinaire |  |
| `models` | `models/patient_sensitivity_reference_comparison.py` | `active` | Comparaison urine/Q/SPECT |  |
| `models/tests_exploratoires` | `models/exploratory/patient_sensitivity_qspect_all_timepoints_test.py` | `exploratory` | Test Q/SPECT avec tous les timepoints |  |
| `models/legacy_dependencies` | `models/legacy/attenuation_window_model.py` | `legacy_dependency` | Ancien modèle par fenêtres | patient_sensitivity_multicycle_v3 — Encore importé par attenuation_global_model_v2. |
| `models/legacy_dependencies` | `models/legacy/attenuation_global_model_v2.py` | `legacy_dependency` | Ancien modèle global v2 | patient_sensitivity_multicycle_v3 — Encore importé par le moteur v3. |
| `models/legacy_dependencies` | `models/legacy/patient_sensitivity_feature_test_v2.py` | `legacy_dependency` | Preuve de concept v2 | patient_sensitivity_generalized — Certaines fonctions sont encore importées par v3. |
| `models/legacy_dependencies` | `models/legacy/patient_sensitivity_urine_target.py` | `legacy_dependency` | Constructeur historique de la cible urinaire | patient_sensitivity_urine_generalized — Le constructeur de données reste utilisé par l’analyse généralisée. |
| `retention_excretion/core` | `retention_excretion/sang_urine.py` | `shared_library` | Lecture et bilan urine/sang |  |
| `retention_excretion` | `retention_excretion/planar_total_counts_retention.py` | `active` | Comptes planaires et rétention urinaire |  |
| `retention_excretion/exploratory` | `retention_excretion/excretion_calibration_test.py` | `exploratory` | Ancien test de calibration urinaire |  |
| `retention_excretion/exploratory` | `retention_excretion/planar_excretion_fit_balance.py` | `exploratory` | Fits réciproques planaire/excrétion |  |
| `retention_excretion/exploratory` | `retention_excretion/imaging_excretion_fit_balance.py` | `exploratory` | Fits réciproques imagerie/excrétion |  |
| `validation` | `validation/compare_planar_qspect.py` | `active` | Validation planaire versus Q/SPECT |  |
| `quality_control` | `quality_control/acquisition_timeline.py` | `active` | Dates et heures d’acquisition |  |
| `quality_control` | `quality_control/view_patient_images.py` | `active` | Aperçu des séries patient |  |
| `quality_control` | `quality_control/generate_dicom_metadata_headers.py` | `active` | Export des métadonnées DICOM |  |
| `quality_control` | `quality_control/report_exact_dicom_duplicates.py` | `active` | Rapport des doublons exacts |  |
| `models/presentation_summary` | `models/presentation/presentation_model_figures.py` | `presentation` | Figures synthèse des modèles |  |
| `models/presentation_summary` | `models/presentation/presentation_slope_constraint_figures.py` | `presentation` | Figures pentes, urine et décroissance |  |
| `admin` | `admin/run_all_patient_figures.py` | `orchestration` | Exécuteur multi-patient |  |
| `admin` | `admin/analysis_catalog.py` | `orchestration` | Catalogue canonique des analyses |  |
| `admin` | `admin/main.py` | `orchestration` | Lanceur DICOM générique |  |

## Exécution après la réorganisation

Depuis la racine `Projet_Lu177/`, activer l’environnement puis entrer
dans le sous-dossier source:

```bash
source .venv/bin/activate
cd code
```

Lancer ensuite les modules avec `python -m`:

- toutes les figures: `python -m admin.run_all_patient_figures`;
- modèle Q/SPECT: `python -m models.patient_sensitivity_generalized`;
- pentes Q/SPECT: `python -m models.patient_sensitivity_qspect_slopes`;
- rétention urinaire: `python -m retention_excretion.planar_total_counts_retention`;
- rapport de temps mort: `python -m corrections.dead_time.generate_planar_dead_time_report`.

## Candidats à l’archivage

- Aucun candidat immédiat; l’étape 1 a été nettoyée.

## Regroupements recommandés

1. Extraire les fonctions encore utilisées de `attenuation_window_model.py`,
   `attenuation_global_model_v2.py` et `patient_sensitivity_feature_test_v2.py`
   vers un moteur partagé sous `models/engines`.
2. Réduire `patient_sensitivity_urine_target.py` au chargement de la cible
   urinaire; conserver les analyses dans `patient_sensitivity_urine_generalized.py`.
3. Après extraction, réévaluer les dépendances restantes avant toute
   nouvelle suppression.

