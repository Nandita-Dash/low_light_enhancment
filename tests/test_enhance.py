import cv2
import numpy as np
import pytest

import enhance


def mean_level(cdf):
    pdf = np.diff(cdf.mean(axis=1), prepend=0.0)
    return float((pdf * np.arange(256) / 255).sum())


@pytest.mark.parametrize("target", [0.3, 0.45, 0.6, 0.8])
def test_retarget_cdf_hits_target_mean(learned_reference, target):
    cdf = enhance.retarget_cdf(np.load(enhance.REF_CDF_PATH), target)
    assert mean_level(cdf) == pytest.approx(target, abs=0.01)
    assert np.all(np.diff(cdf, axis=0) >= -1e-12)
    assert cdf[-1] == pytest.approx(1.0)


def test_learned_reference_follows_brightness(learned_reference):
    low, high = enhance.default_reference_cdf(0.35), enhance.default_reference_cdf(0.75)
    assert mean_level(low) < mean_level(high)


def test_pipeline_demo_mode(demo_mode, dark_image):
    res = enhance.run_pipeline(dark_image, truth_img=255 - dark_image)
    assert res["mode"] == "demo"
    for img in res["images"].values():
        assert img.shape == dark_image.shape and img.dtype == np.uint8
    assert res["stats"]["final"]["brightness"] > res["stats"]["original"]["brightness"]
    assert set(res["metrics"]) == {"original", "autoencoder", "final"}


@pytest.mark.parametrize("fixture", ["demo_mode", "underexposing_model"])
def test_brightness_changes_output(request, fixture, learned_reference, dark_image):
    request.getfixturevalue(fixture)
    dim = enhance.run_pipeline(dark_image, brightness=0.35)["stats"]["final"]["brightness"]
    bright = enhance.run_pipeline(dark_image, brightness=0.75)["stats"]["final"]["brightness"]
    assert bright > dim + 10


def test_model_mode_pads_odd_sizes(model_mode, dark_image):
    img = dark_image[:91, :117]
    res = enhance.run_pipeline(img)
    assert res["mode"] == "autoencoder"
    assert res["images"]["final"].shape == img.shape


def test_strength_is_clamped(demo_mode, dark_image):
    full = enhance.run_pipeline(dark_image, strength=1.0)["images"]["final"]
    over = enhance.run_pipeline(dark_image, strength=5.0)["images"]["final"]
    none = enhance.run_pipeline(dark_image, strength=-3.0)
    assert np.array_equal(full, over)
    assert np.array_equal(none["images"]["final"], none["images"]["autoencoder"])


@pytest.mark.parametrize("kwargs", [{"strength": float("nan")}, {"brightness": float("inf")}])
def test_non_finite_controls_rejected(demo_mode, dark_image, kwargs):
    with pytest.raises(ValueError):
        enhance.run_pipeline(dark_image, **kwargs)



@pytest.mark.parametrize("name", ["astronaut", "rocket"])
def test_matching_does_not_fog(demo_mode, low_light_photo, name):
    """Regression: matching lifted blacks to grey and flattened contrast (black 11 -> 75 on astronaut,
    47 -> 108 on rocket), so results looked foggy and washed out."""
    res = enhance.run_pipeline(low_light_photo(name), brightness=0.5)
    L = {k: cv2.cvtColor(res["images"][k], cv2.COLOR_RGB2LAB)[..., 0] for k in ("autoencoder", "final")}
    assert np.percentile(L["final"], 1) <= np.percentile(L["autoencoder"], 1) + 15
    assert L["final"].std() >= 0.9 * L["autoencoder"].std()
