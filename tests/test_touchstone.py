import numpy as np
import pytest

from aln_model.config import PreparationConfig
from aln_model.touchstone import (
    interpolate_complex,
    ma_to_complex,
    parse_touchstone_s11,
)


def test_ma_to_complex_converts_degrees():
    result = ma_to_complex(np.array([2.0, 1.0]), np.array([90.0, -180.0]))

    np.testing.assert_allclose(result, np.array([2j, -1 + 0j]), atol=1e-12)


def test_parse_touchstone_requires_hz_s_ma_r_50():
    with pytest.raises(ValueError, match="Hz S MA R 50"):
        parse_touchstone_s11("# GHz S RI R 50\n4.2 0.1 0.2\n", path="bad.s1p")


def test_parse_touchstone_returns_complex_s11():
    freq, s11 = parse_touchstone_s11(
        "! generated\n# Hz S MA R 50\n4200000000 1 0\n4300000000 0.5 90\n",
        path="ok.s1p",
    )

    np.testing.assert_array_equal(freq, [4.2e9, 4.3e9])
    np.testing.assert_allclose(s11, [1 + 0j, 0 + 0.5j], atol=1e-12)


def test_interpolate_complex_interpolates_real_and_imaginary_parts():
    freq = np.array([4.2e9, 4.3e9, 4.4e9])
    values = np.array([1 + 0j, 0 + 1j, -1 + 0j])

    result = interpolate_complex(freq, values, np.array([4.25e9, 4.35e9]))

    np.testing.assert_allclose(result, [0.5 + 0.5j, -0.5 + 0.5j])


def test_interpolate_complex_rejects_extrapolation():
    with pytest.raises(ValueError, match="outside source frequency range"):
        interpolate_complex(
            np.array([4.2e9, 4.3e9]),
            np.array([1 + 0j, 0 + 1j]),
            np.array([4.1e9, 4.2e9]),
        )


def test_default_grid_is_deterministic_common_band():
    grid = PreparationConfig().frequency_grid()

    assert grid[0] == 4.2e9
    assert grid[-1] == 5.39e9
    assert len(grid) == 1191


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frequency_step_hz": 0},
        {"frequency_step_hz": -1},
        {"frequency_start_hz": 5e9, "frequency_stop_hz": 4e9},
        {"frequency_step_hz": 6e6},
    ],
)
def test_preparation_config_rejects_invalid_grid(kwargs):
    with pytest.raises(ValueError):
        PreparationConfig(**kwargs)


@pytest.mark.parametrize(
    "row, message",
    [
        ("0 1 0", "positive"),
        ("-1 1 0", "positive"),
        ("4200000000 -0.1 0", "magnitude"),
    ],
)
def test_parse_touchstone_rejects_nonphysical_ma(row, message):
    text = f"# Hz S MA R 50\n{row}\n4300000000 0.5 0\n"
    with pytest.raises(ValueError, match=message):
        parse_touchstone_s11(text, path="bad.s1p")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frequency_start_hz": np.nan},
        {"frequency_stop_hz": np.inf},
        {"frequency_step_hz": np.nan},
        {"response_hash_tolerance": np.inf},
        {"n_folds": np.inf},
        {"random_state": np.nan},
    ],
)
def test_preparation_config_rejects_nonfinite_values(kwargs):
    with pytest.raises(ValueError, match="finite"):
        PreparationConfig(**kwargs)


@pytest.mark.parametrize(
    "source_frequency, source_values, target_frequency",
    [
        (np.array([]), np.array([], dtype=complex), np.array([1.0])),
        (np.array([1.0, np.nan]), np.array([1j, 2j]), np.array([1.0])),
        (np.array([1.0, 2.0]), np.array([1j, np.nan + 0j]), np.array([1.0])),
        (np.array([1.0, 2.0]), np.array([1j, 2j]), np.array([])),
        (np.array([1.0, 2.0]), np.array([1j, 2j]), np.array([1.5, 1.4])),
        (np.array([1.0, 2.0]), np.array([1j, 2j]), np.array([np.nan])),
    ],
)
def test_interpolate_complex_rejects_invalid_arrays(
    source_frequency, source_values, target_frequency
):
    with pytest.raises(ValueError):
        interpolate_complex(source_frequency, source_values, target_frequency)
