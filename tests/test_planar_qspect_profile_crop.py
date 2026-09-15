import numpy as np
import pytest

import planar_qspect_profile_crop as profile_crop


def test_shared_profile_crop_resolver_requires_complete_pairs():
    with pytest.raises(ValueError, match="one Q/SPECT acquisition per planar scan"):
        profile_crop.resolve_profile_crop_matches([{}], [{}, {}])


def test_profile_crop_recovers_known_translation_and_passes_qc():
    qspect = np.exp(-0.5 * ((np.arange(40) - 12.0) / 2.5) ** 2)
    qspect += 0.65 * np.exp(-0.5 * ((np.arange(40) - 29.0) / 3.0) ** 2)
    planar = np.zeros(120, dtype=float)
    planar[50:90] = qspect

    result = profile_crop.search_profile_crop(
        planar, qspect, initial_top=45, max_shift_px=15
    )
    result.update({"initial_top": 45, "initial_bottom": 85})
    checked = profile_crop.add_match_quality_control(
        result,
        planar_spacing_mm=2.4,
        minimum_correlation=0.90,
        minimum_peak_margin=-1.0,
        maximum_peak_width_cm=10.0,
    )

    assert checked["proposed_crop_top"] == 50
    assert checked["crop_top"] == 50
    assert checked["match_accepted"] is True


def test_profile_crop_falls_back_when_optimum_hits_search_boundary():
    qspect = np.exp(-0.5 * ((np.arange(30) - 15.0) / 3.0) ** 2)
    planar = np.zeros(100, dtype=float)
    planar[50:80] = qspect

    result = profile_crop.search_profile_crop(
        planar, qspect, initial_top=40, max_shift_px=10
    )
    result.update({"initial_top": 40, "initial_bottom": 70})
    checked = profile_crop.add_match_quality_control(
        result,
        planar_spacing_mm=2.4,
        minimum_peak_margin=-1.0,
        maximum_peak_width_cm=10.0,
    )

    assert checked["proposed_crop_top"] == 50
    assert checked["search_edge_hit"] is True
    assert checked["match_accepted"] is False
    assert checked["crop_top"] == 40
    assert checked["crop_bottom"] == 70
    assert "search_edge_hit" in checked["rejection_reasons"]


def test_profile_crop_expands_a_boundary_limited_search():
    qspect = np.exp(-0.5 * ((np.arange(30) - 15.0) / 3.0) ** 2)
    planar = np.zeros(110, dtype=float)
    planar[53:83] = qspect

    result = profile_crop.search_profile_crop_with_boundary_expansion(
        planar,
        qspect,
        initial_top=40,
        initial_max_shift_px=10,
        maximum_shift_px=20,
        expansion_step_px=5,
    )

    assert result["crop_top"] == 53
    assert result["search_edge_hit"] is False
    assert result["search_expanded"] is True
    assert result["effective_search_limit_px"] == 15
    assert result["search_attempts"] == 2
