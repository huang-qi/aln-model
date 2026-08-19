import numpy as np

from aln_model.labels import extract_engineering_labels
import aln_model.labels as labels_module


Z0 = 50.0


def _bvd_s11(
    frequency_hz: np.ndarray,
    *,
    fs_hz: float = 4.7e9,
    cm_f: float = 5e-15,
    c0_f: float = 100e-15,
    motional_q: float = 500.0,
    extra_branch: tuple[float, float] | None = None,
) -> np.ndarray:
    omega = 2 * np.pi * frequency_hz

    def branch_admittance(branch_fs: float, branch_cm: float) -> np.ndarray:
        lm = 1 / ((2 * np.pi * branch_fs) ** 2 * branch_cm)
        rm = 2 * np.pi * branch_fs * lm / motional_q
        return 1 / (rm + 1j * omega * lm + 1 / (1j * omega * branch_cm))

    admittance = 1j * omega * c0_f + branch_admittance(fs_hz, cm_f)
    if extra_branch is not None:
        extra_fs, strength = extra_branch
        admittance += branch_admittance(extra_fs, cm_f * strength)
    impedance = 1 / admittance
    return (impedance - Z0) / (impedance + Z0)


def test_extracts_physically_ordered_resonances_and_positive_q():
    frequency = np.linspace(4.2e9, 5.39e9, 1191)
    s11 = _bvd_s11(frequency)

    labels = extract_engineering_labels(frequency, s11)

    assert abs(labels.fs_hz - 4.7e9) < 3e6
    assert labels.fs_hz < labels.fp_hz
    assert labels.k_eff2 > 0
    assert labels.qs > 0
    assert labels.qp > 0
    assert labels.quality_valid
    assert labels.quality_reason == "OK"
    assert labels.order_valid
    assert labels.target_valid_fs
    assert labels.target_valid_fp
    assert labels.target_valid_k_eff2
    assert labels.target_valid_qs
    assert labels.target_valid_qp
    assert labels.ambiguity_reason == "NONE"


def test_boundary_resonance_has_explicit_invalid_flags():
    frequency = np.linspace(4.2e9, 5.39e9, 1191)
    s11 = _bvd_s11(frequency, fs_hz=4.19e9)

    labels = extract_engineering_labels(frequency, s11)

    assert not labels.quality_valid
    assert labels.quality_boundary
    assert labels.quality_reason != "OK"


def test_detects_dangerous_secondary_series_peak():
    frequency = np.linspace(4.2e9, 5.39e9, 1191)
    s11 = _bvd_s11(
        frequency,
        fs_hz=4.7e9,
        extra_branch=(5.12e9, 0.8),
    )

    labels = extract_engineering_labels(frequency, s11)

    assert labels.spurious_peak_count >= 1
    assert labels.spurious_max_ratio >= 0.2
    assert labels.spurious_dangerous
    assert np.isfinite(labels.nearest_spurious_frequency_hz)
    assert labels.main_spurious_delta_hz > 0


def test_algorithm_version_is_explicit_in_label_result():
    frequency = np.linspace(4.2e9, 5.9e9, 1701)
    labels = extract_engineering_labels(
        frequency,
        _bvd_s11(frequency),
        algorithm_version="review-test-v7",
    )

    assert labels.label_algorithm_version == "review-test-v7"


def test_target_validity_flags_do_not_discard_frequencies_when_q_fails(monkeypatch):
    frequency = np.linspace(4.2e9, 5.39e9, 1191)
    calls = iter([np.nan, 100.0])
    monkeypatch.setattr(
        labels_module,
        "_q_from_half_prominence",
        lambda *args, **kwargs: next(calls),
    )

    labels = extract_engineering_labels(frequency, _bvd_s11(frequency))

    assert labels.target_valid_fs
    assert labels.target_valid_fp
    assert labels.target_valid_k_eff2
    assert not labels.target_valid_qs
    assert labels.target_valid_qp


def test_q_width_uses_frequency_axis_on_nonuniform_grid():
    frequency = np.unique(
        np.concatenate(
            [
                np.arange(4.2e9, 4.68e9, 2e6),
                np.arange(4.68e9, 4.72e9, 0.2e6),
                np.arange(4.72e9, 5.0e9, 1e6),
                np.arange(5.0e9, 5.391e9, 2e6),
            ]
        )
    )
    assert {0.2e6, 1e6, 2e6}.issubset(set(np.diff(frequency)))
    peak_frequency = 4.7e9
    fwhm_hz = 4e6
    values = 1 / (1 + (2 * (frequency - peak_frequency) / fwhm_hz) ** 2)
    peak_index = int(np.argmax(values))

    q = labels_module._q_from_half_prominence(frequency, values, peak_index)

    assert abs(q - peak_frequency / fwhm_hz) / (peak_frequency / fwhm_hz) < 0.02


def test_rejects_singular_s11_with_reason_code():
    frequency = np.linspace(4.2e9, 5.39e9, 1191)

    labels = extract_engineering_labels(frequency, np.ones_like(frequency, dtype=complex))

    assert not labels.quality_valid
    assert labels.quality_reason == "NONFINITE_IMPEDANCE"
