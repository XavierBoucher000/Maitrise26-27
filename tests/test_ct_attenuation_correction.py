import numpy as np
import pytest

import ct_attenuation_correction as ctac
import planar_qspect_crop
import attenuation_correction
import compare_crop_methods
import crop_excluded_counts_test


def test_raystation_density_matches_nodes_and_clamps_outside_range():
    result = ctac.raystation_hu_to_mass_density_g_cm3(
        np.array([-2000.0, *ctac.RAYSTATION_HU_NODES, 5000.0])
    )

    assert result[0] == pytest.approx(ctac.RAYSTATION_MASS_DENSITY_G_CM3[0])
    assert result[-1] == pytest.approx(ctac.RAYSTATION_MASS_DENSITY_G_CM3[-1])
    np.testing.assert_allclose(result[1:-1], ctac.RAYSTATION_MASS_DENSITY_G_CM3)


def test_material_classification_uses_configurable_hu_ranges():
    result = ctac.classify_material_from_hu(
        np.array([-1000.0, -500.0, -100.0, 0.0, 500.0])
    )

    np.testing.assert_array_equal(result["material_codes"], [0, 1, 2, 3, 4])
    assert result["material_names"] == ctac.DEFAULT_MATERIAL_NAMES


def test_mass_attenuation_interpolation_is_log_log_and_rejects_extrapolation():
    pairs = ((100.0, 1.0), (400.0, 0.25))
    assert ctac.interpolate_mass_attenuation_cm2_g(pairs, 200.0) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="outside"):
        ctac.interpolate_mass_attenuation_cm2_g(pairs, 500.0)


def test_hu_to_mu_combines_density_and_material_coefficient():
    hu = np.array([-1000.0, -505.0, -67.0, 9.0, 844.0])
    result = ctac.hu_to_mu_208_raystation_materials(hu)

    np.testing.assert_allclose(
        result["mu_208_cm_inv"],
        result["mass_density_g_cm3"] * result["mu_over_rho_cm2_g"],
    )
    assert np.all(np.isfinite(result["mu_208_cm_inv"]))
    assert np.all(result["mu_208_cm_inv"] > 0.0)


def test_new_factor_map_has_expected_shape_and_integral():
    volume = np.zeros((2, 3, 4), dtype=float)
    ct = {"volume": volume, "pixel_spacing": [10.0, 10.0]}
    result = ctac.ct_attenuation_factor_map_raystation_materials(
        ct,
        clip_range=(1.0, np.inf),
    )

    expected_mu = ctac.hu_to_mu_208_raystation_materials(np.array([0.0]))[
        "mu_208_cm_inv"
    ][0]
    assert result["factor_map"].shape == (2, 4)
    np.testing.assert_allclose(result["mu_integral"], 3.0 * expected_mu)
    np.testing.assert_allclose(result["factor_map"], np.exp(0.5 * 3.0 * expected_mu))


def test_apply_crop_can_select_either_conversion_method():
    crop = np.ones((2, 4), dtype=float)
    ct = {"volume": np.zeros((2, 3, 4)), "pixel_spacing": [10.0, 10.0]}

    old = ctac.apply_ct_attenuation_correction_to_crop(crop, ct, "water_scaled")
    new = ctac.apply_ct_attenuation_correction_to_crop(crop, ct, "raystation_materials")

    assert old["conversion_method"] == "water_scaled"
    assert new["conversion_method"] == "raystation_materials"
    assert old["ct_factor_stats"]["mean"] > 1.0
    assert new["ct_factor_stats"]["mean"] > 1.0


def test_simpleitk_resampling_preserves_identical_physical_grid():
    optical_depth = np.arange(20, dtype=float).reshape(4, 5) / 20.0
    result = ctac.resample_optical_depth_to_planar_simpleitk(
        optical_depth,
        source_spacing_yx_mm=(5.0, 4.0),
        planar_shape_yx=optical_depth.shape,
        planar_spacing_yx_mm=(5.0, 4.0),
    )

    np.testing.assert_allclose(result["mu_integral_resampled"], optical_depth)
    np.testing.assert_allclose(
        result["factor_map"],
        np.exp(0.5 * optical_depth),
    )


def test_simpleitk_resampling_uses_planar_shape_and_physical_extents():
    optical_depth = np.ones((3, 3), dtype=float)
    result = ctac.resample_optical_depth_to_planar_simpleitk(
        optical_depth,
        source_spacing_yx_mm=(4.0, 4.0),
        planar_shape_yx=(5, 5),
        planar_spacing_yx_mm=(2.0, 2.0),
    )

    assert result["factor_map"].shape == (5, 5)
    assert result["source_extent_yx_mm"] == pytest.approx((8.0, 8.0))
    assert result["planar_extent_yx_mm"] == pytest.approx((8.0, 8.0))
    np.testing.assert_allclose(result["factor_map"], np.exp(0.5))


def test_excluded_crop_counts_splits_above_inside_and_below():
    image = np.asarray([[1.0], [2.0], [3.0], [4.0]])
    result = planar_qspect_crop.estimate_excluded_crop_counts(
        image, crop_top=1, crop_bottom=3, mask_threshold_fraction=0.0
    )

    assert result["masked_counts"]["above"] == pytest.approx(1.0)
    assert result["masked_counts"]["inside"] == pytest.approx(5.0)
    assert result["masked_counts"]["below"] == pytest.approx(4.0)
    assert result["outside_fraction"] == pytest.approx(0.5)
    assert result["inside_fraction"] == pytest.approx(0.5)


def test_geometric_mean_alignment_option_flips_pa_horizontally():
    ap = np.asarray([[1.0, 4.0]])
    pa = np.asarray([[9.0, 16.0]])

    historical = attenuation_correction.geometric_mean(
        ap, pa, align_pa_to_ap=False
    )
    aligned = attenuation_correction.geometric_mean(
        ap, pa, align_pa_to_ap=True
    )

    np.testing.assert_allclose(historical, [[3.0, 8.0]])
    np.testing.assert_allclose(aligned, [[4.0, 6.0]])


def test_crop_profile_registration_recovers_vertical_shift():
    reference = np.zeros((160,), dtype=float)
    reference[30:50] = 1.0
    reference[95:115] = 0.6
    moving = np.zeros_like(reference)
    moving[37:57] = 1.0
    moving[102:122] = 0.6

    result = compare_crop_methods.estimate_planar_longitudinal_shift(
        reference,
        moving,
        max_shift_px=15,
    )

    assert result["shift_y_px"] == 7
    assert result["correlation"] == pytest.approx(1.0)


def test_registered_crop_translation_preserves_height_at_edges():
    assert compare_crop_methods.translate_crop_bounds((20, 70), 10, 100) == (30, 80)
    assert compare_crop_methods.translate_crop_bounds((20, 70), -40, 100) == (0, 50)
    assert compare_crop_methods.translate_crop_bounds((40, 90), 40, 100) == (50, 100)


def test_silhouette_shift_accepts_consistent_body_edges():
    reference = {"top": 100, "bottom": 900, "length": 800}
    moving = {"top": 84, "bottom": 884, "length": 800}

    result = compare_crop_methods.estimate_silhouette_shift(reference, moving)

    assert result["raw_shift_y_px"] == -16
    assert result["shift_y_px"] == -16
    assert result["accepted"] is True


def test_silhouette_shift_falls_back_when_body_is_incomplete():
    reference = {"top": 100, "bottom": 900, "length": 800}
    moving = {"top": 150, "bottom": 550, "length": 400}

    result = compare_crop_methods.estimate_silhouette_shift(reference, moving)

    assert result["raw_shift_y_px"] == -150
    assert result["shift_y_px"] == 0
    assert result["accepted"] is False


def test_recovered_crop_activity_adds_outside_counts_without_attenuation():
    rows = [
        {
            "day_offset": 0.0,
            "local_dwell_time_s": 10.0,
            "ctac_local_activity_mbq": 5.0,
            "qspect_activity_mbq": 3.0,
            "crop_excluded_counts": {
                "masked_counts": {"outside": 93.6},
                "positive_counts": {"outside": 187.2},
            },
        }
    ]

    result = crop_excluded_counts_test.calculate_recovered_activity(rows)[0]

    assert result["recovered_outside_no_attenuation_mbq"] == pytest.approx(1.0)
    assert result["recovered_outside_positive_no_attenuation_mbq"] == pytest.approx(2.0)
    assert result["hybrid_activity_mbq"] == pytest.approx(6.0)
    assert result["hybrid_over_qspect"] == pytest.approx(2.0)
