from admin import analysis_catalog
from core.project_paths import PROJECT_ROOT


def test_registered_source_paths_exist():
    missing = [
        spec.path
        for spec in analysis_catalog.ANALYSES
        if not (PROJECT_ROOT / spec.path).exists()
    ]
    assert missing == []


def test_archive_candidates_are_not_run_for_every_cycle():
    assert all(
        not spec.run_per_complete_cycle
        for spec in analysis_catalog.by_status(
            analysis_catalog.STATUS_ARCHIVE_CANDIDATE
        )
    )


def test_cycle_registry_excludes_historical_model_versions():
    names = {spec.name for spec in analysis_catalog.cycle_analyses()}
    assert "patient_sensitivity_feature_test" not in names
    assert "attenuation_window_model" not in names
    assert "attenuation_global_model_v2" not in names


def test_markdown_describes_current_legacy_replacements():
    document = analysis_catalog.render_markdown()
    assert "patient_sensitivity_feature_test_v2.py" in document
    assert "patient_sensitivity_generalized" in document


def test_catalog_covers_all_project_scripts():
    discovered = {
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in PROJECT_ROOT.rglob("*.py")
        if not any(part in {".venv", "tests", "archive"} for part in path.parts)
        and path.name != "__init__.py"
    }
    registered = {spec.path for spec in analysis_catalog.ANALYSES}
    assert discovered == registered
