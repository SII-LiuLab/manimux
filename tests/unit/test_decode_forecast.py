import pytest

from manimux.runtime.decode_forecast import DecodeForecast


def forecast(**options) -> DecodeForecast:
    return DecodeForecast(**{"floor_s": 0.06, "size": 3, "mode": "max", **options})


def test_the_floor_is_the_estimate_until_a_decode_finishes():
    assert forecast().seconds == 0.06


def test_max_mode_tracks_the_worst_decode_in_the_window():
    estimator = forecast()
    for sample_ms in (80.0, 120.0, 90.0):
        estimator.observe(sample_ms)
    assert estimator.seconds == pytest.approx(0.12)
    # The window is bounded: the 120 ms outlier leaves after three newer samples.
    for sample_ms in (70.0, 75.0, 80.0):
        estimator.observe(sample_ms)
    assert estimator.seconds == pytest.approx(0.08)


def test_mean_mode_averages_the_window():
    estimator = forecast(mode="mean")
    for sample_ms in (90.0, 120.0, 150.0):
        estimator.observe(sample_ms)
    assert estimator.seconds == pytest.approx(0.12)


def test_the_floor_bounds_a_fast_decode_so_the_seed_source_cannot_flip():
    estimator = forecast(mode="mean")
    estimator.observe(10.0)
    assert estimator.seconds == 0.06


def test_an_empty_window_ignores_every_measurement():
    estimator = forecast(size=0)
    estimator.observe(200.0)
    assert estimator.seconds == 0.06
