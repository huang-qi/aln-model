from __future__ import annotations

import re

import numpy as np


_EXPECTED_HEADER = re.compile(
    r"^#\s*Hz\s+S\s+MA\s+R\s+50(?:\.0*)?\s*$", re.IGNORECASE
)


def ma_to_complex(magnitude: np.ndarray, phase_degrees: np.ndarray) -> np.ndarray:
    magnitude = np.asarray(magnitude, dtype=float)
    phase_degrees = np.asarray(phase_degrees, dtype=float)
    return magnitude * np.exp(1j * np.deg2rad(phase_degrees))


def parse_touchstone_s11(text: str, *, path: str = "<memory>") -> tuple[np.ndarray, np.ndarray]:
    header_seen = False
    records: list[tuple[float, float, float]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("!", 1)[0].strip()
        if not line:
            continue
        if line.startswith("#"):
            if header_seen or not _EXPECTED_HEADER.match(line):
                raise ValueError(f"{path}: expected '# Hz S MA R 50'")
            header_seen = True
            continue
        if not header_seen:
            raise ValueError(f"{path}:{line_number}: data before '# Hz S MA R 50'")
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"{path}:{line_number}: expected frequency magnitude phase")
        try:
            records.append(tuple(float(value) for value in fields))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: non-numeric Touchstone data") from exc
    if not header_seen:
        raise ValueError(f"{path}: expected '# Hz S MA R 50'")
    if len(records) < 2:
        raise ValueError(f"{path}: fewer than two S11 samples")
    data = np.asarray(records, dtype=float)
    frequencies = data[:, 0]
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{path}: Touchstone data must be finite")
    if np.any(frequencies <= 0):
        raise ValueError(f"{path}: frequencies must be positive")
    if np.any(data[:, 1] < 0):
        raise ValueError(f"{path}: MA magnitude must be non-negative")
    if not np.all(np.diff(frequencies) > 0):
        raise ValueError(f"{path}: frequencies must be finite and strictly increasing")
    return frequencies, ma_to_complex(data[:, 1], data[:, 2])


def interpolate_complex(
    source_frequency_hz: np.ndarray,
    source_values: np.ndarray,
    target_frequency_hz: np.ndarray,
) -> np.ndarray:
    source_frequency_hz = np.asarray(source_frequency_hz, dtype=float)
    source_values = np.asarray(source_values, dtype=complex)
    target_frequency_hz = np.asarray(target_frequency_hz, dtype=float)
    if (
        source_frequency_hz.ndim != 1
        or target_frequency_hz.ndim != 1
        or source_values.ndim != 1
        or source_values.shape != source_frequency_hz.shape
    ):
        raise ValueError("source frequencies and values must be aligned one-dimensional arrays")
    if source_frequency_hz.size == 0 or target_frequency_hz.size == 0:
        raise ValueError("source and target frequency arrays must be non-empty")
    if (
        not np.all(np.isfinite(source_frequency_hz))
        or not np.all(np.isfinite(target_frequency_hz))
        or not np.all(np.isfinite(source_values))
    ):
        raise ValueError("source frequencies, values and target frequencies must be finite")
    if np.any(source_frequency_hz <= 0) or np.any(target_frequency_hz <= 0):
        raise ValueError("source and target frequencies must be positive")
    if not np.all(np.diff(source_frequency_hz) > 0) or not np.all(
        np.diff(target_frequency_hz) > 0
    ):
        raise ValueError("source and target frequencies must be strictly increasing")
    if (
        target_frequency_hz.min() < source_frequency_hz[0]
        or target_frequency_hz.max() > source_frequency_hz[-1]
    ):
        raise ValueError("target grid is outside source frequency range")
    real = np.interp(target_frequency_hz, source_frequency_hz, source_values.real)
    imag = np.interp(target_frequency_hz, source_frequency_hz, source_values.imag)
    return real + 1j * imag
