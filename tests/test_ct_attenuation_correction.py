import numpy as np
import pytest

import ct_attenuation_correction as ctac


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
