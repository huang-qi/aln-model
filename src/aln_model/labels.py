from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths


@dataclass(frozen=True)
class EngineeringLabels:
    fs_hz: float = np.nan
    fp_hz: float = np.nan
    k_eff2: float = np.nan
    qs: float = np.nan
    qp: float = np.nan
    spurious_peak_count: int = 0
    spurious_max_ratio: float = 0.0
    spurious_dangerous: bool = False
    main_mode_frequency_hz: float = np.nan
    nearest_spurious_frequency_hz: float = np.nan
    main_spurious_delta_hz: float = np.nan
    order_valid: bool = False
    quality_mode_crossing: bool = False
    ambiguity_reason: str = "UNSET"
    target_valid_fs: bool = False
    target_valid_fp: bool = False
    target_valid_k_eff2: bool = False
    target_valid_qs: bool = False
    target_valid_qp: bool = False
    target_valid_spurious: bool = False
    quality_valid: bool = False
    quality_boundary: bool = False
    quality_ambiguous: bool = False
    quality_grid_ok: bool = True
    quality_reason: str = "UNSET"
    label_algorithm_version: str = "resonance-v1"

    def as_dict(self) -> dict[str, object]:
        values = asdict(self)
        values.update(
            {
                "fs": self.fs_hz,
                "fp": self.fp_hz,
                "Qs": self.qs,
                "Qp": self.qp,
                "spurious": self.spurious_dangerous,
            }
        )
        return values


def _invalid(
    reason: str,
    *,
    boundary: bool = False,
    grid_ok: bool = True,
    algorithm_version: str,
) -> EngineeringLabels:
    return EngineeringLabels(
        quality_reason=reason,
        quality_boundary=boundary,
        quality_grid_ok=grid_ok,
        ambiguity_reason=reason,
        label_algorithm_version=algorithm_version,
    )


def _prominent_peaks(
    values: np.ndarray, *, minimum_relative_prominence: float
) -> tuple[np.ndarray, np.ndarray]:
    span = float(np.ptp(values))
    if not np.isfinite(span) or span <= 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    peaks, _ = find_peaks(values, prominence=span * minimum_relative_prominence)
    if peaks.size == 0:
        return peaks, np.array([], dtype=float)
    return peaks, peak_prominences(values, peaks)[0]


def _q_from_half_prominence(
    frequency_hz: np.ndarray, values: np.ndarray, peak_index: int
) -> float:
    width_result = peak_widths(values, [peak_index], rel_height=0.5)
    left_ip = float(width_result[2][0])
    right_ip = float(width_result[3][0])
    sample_indices = np.arange(len(frequency_hz), dtype=float)
    left_frequency_hz = float(np.interp(left_ip, sample_indices, frequency_hz))
    right_frequency_hz = float(np.interp(right_ip, sample_indices, frequency_hz))
    width_hz = right_frequency_hz - left_frequency_hz
    return float(frequency_hz[peak_index] / width_hz) if width_hz > 0 else np.nan


def extract_engineering_labels(
    frequency_hz: np.ndarray,
    s11: np.ndarray,
    *,
    reference_ohms: float = 50.0,
    dangerous_spurious_ratio: float = 0.2,
    algorithm_version: str = "resonance-v1",
) -> EngineeringLabels:
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    s11 = np.asarray(s11, dtype=complex)
    grid_ok = (
        frequency_hz.ndim == 1
        and s11.shape == frequency_hz.shape
        and frequency_hz.size >= 5
        and np.all(np.isfinite(frequency_hz))
        and np.all(np.diff(frequency_hz) > 0)
    )
    if not grid_ok:
        return _invalid("INVALID_GRID", grid_ok=False, algorithm_version=algorithm_version)
    denominator = 1 - s11
    if (
        not np.all(np.isfinite(s11))
        or np.any(np.abs(denominator) <= np.finfo(float).eps)
    ):
        return _invalid("NONFINITE_IMPEDANCE", algorithm_version=algorithm_version)
    impedance = reference_ohms * (1 + s11) / denominator
    if not np.all(np.isfinite(impedance)) or np.any(np.abs(impedance) == 0):
        return _invalid("NONFINITE_IMPEDANCE", algorithm_version=algorithm_version)
    admittance = 1 / impedance
    abs_y = np.abs(admittance)
    abs_z = np.abs(impedance)
    series_peaks, series_prominence = _prominent_peaks(
        abs_y, minimum_relative_prominence=0.01
    )
    parallel_peaks, parallel_prominence = _prominent_peaks(
        abs_z, minimum_relative_prominence=0.001
    )
    if series_peaks.size == 0:
        boundary = int(np.argmax(abs_y)) in {0, len(abs_y) - 1}
        return _invalid(
            "NO_SERIES_PEAK", boundary=boundary, algorithm_version=algorithm_version
        )
    if parallel_peaks.size == 0:
        boundary = int(np.argmax(abs_z)) in {0, len(abs_z) - 1}
        return _invalid(
            "NO_PARALLEL_PEAK", boundary=boundary, algorithm_version=algorithm_version
        )

    pairs: list[tuple[float, int, int, int, int]] = []
    for si, series_index in enumerate(series_peaks):
        for pi, parallel_index in enumerate(parallel_peaks):
            if parallel_index > series_index:
                score = float(series_prominence[si] * parallel_prominence[pi])
                pairs.append((score, int(series_index), int(parallel_index), si, pi))
    if not pairs:
        boundary = int(np.argmax(abs_y)) in {0, len(abs_y) - 1} or int(
            np.argmax(abs_z)
        ) in {0, len(abs_z) - 1}
        return _invalid(
            "NO_ORDERED_RESONANCE_PAIR",
            boundary=boundary,
            algorithm_version=algorithm_version,
        )
    pairs.sort(reverse=True)
    _, series_index, parallel_index, main_series_pos, _ = pairs[0]
    fs_hz = float(frequency_hz[series_index])
    fp_hz = float(frequency_hz[parallel_index])
    qs = _q_from_half_prominence(frequency_hz, abs_y, series_index)
    qp = _q_from_half_prominence(frequency_hz, abs_z, parallel_index)

    other_prominences = np.delete(series_prominence, main_series_pos)
    other_indices = np.delete(series_peaks, main_series_pos)
    main_prominence = float(series_prominence[main_series_pos])
    ratios = other_prominences / main_prominence if main_prominence > 0 else np.array([])
    max_ratio = float(np.max(ratios)) if ratios.size else 0.0
    ambiguous = len(pairs) > 1 and pairs[1][0] >= 0.8 * pairs[0][0]
    mode_crossing = bool(
        np.any((other_indices > series_index) & (other_indices < parallel_index))
    )
    if other_indices.size:
        nearest_index = int(
            other_indices[np.argmin(np.abs(frequency_hz[other_indices] - fs_hz))]
        )
        nearest_spurious_hz = float(frequency_hz[nearest_index])
        main_spurious_delta_hz = abs(nearest_spurious_hz - fs_hz)
    else:
        nearest_spurious_hz = np.nan
        main_spurious_delta_hz = np.nan
    ambiguity_reason = (
        "MODE_CROSSING"
        if mode_crossing
        else "COMPETING_RESONANCE_PAIR"
        if ambiguous
        else "NONE"
    )
    boundary = series_index <= 1 or parallel_index >= len(frequency_hz) - 2
    valid = np.isfinite(qs) and np.isfinite(qp) and qs > 0 and qp > 0 and not boundary
    frequency_targets_valid = not boundary and np.isfinite(fs_hz) and np.isfinite(fp_hz)
    coupling_target_valid = (
        frequency_targets_valid
        and np.isfinite((fp_hz**2 - fs_hz**2) / fp_hz**2)
        and fp_hz > fs_hz
    )
    reason = "OK" if valid else "BOUNDARY_RESONANCE" if boundary else "INVALID_Q"
    return EngineeringLabels(
        fs_hz=fs_hz,
        fp_hz=fp_hz,
        k_eff2=float((fp_hz**2 - fs_hz**2) / fp_hz**2),
        qs=float(qs),
        qp=float(qp),
        spurious_peak_count=int(other_prominences.size),
        spurious_max_ratio=max_ratio,
        spurious_dangerous=max_ratio >= dangerous_spurious_ratio,
        main_mode_frequency_hz=fs_hz,
        nearest_spurious_frequency_hz=nearest_spurious_hz,
        main_spurious_delta_hz=main_spurious_delta_hz,
        order_valid=True,
        quality_mode_crossing=mode_crossing,
        ambiguity_reason=ambiguity_reason,
        target_valid_fs=bool(frequency_targets_valid),
        target_valid_fp=bool(frequency_targets_valid),
        target_valid_k_eff2=bool(coupling_target_valid),
        target_valid_qs=bool(np.isfinite(qs) and qs > 0 and not boundary),
        target_valid_qp=bool(np.isfinite(qp) and qp > 0 and not boundary),
        target_valid_spurious=True,
        quality_valid=bool(valid),
        quality_boundary=boundary,
        quality_ambiguous=ambiguous,
        quality_grid_ok=True,
        quality_reason=reason,
        label_algorithm_version=algorithm_version,
    )
