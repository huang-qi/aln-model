from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreparationConfig:
    frequency_start_hz: float = 4.2e9
    frequency_stop_hz: float = 5.39e9
    frequency_step_hz: float = 1e6
    n_folds: int = 5
    random_state: int = 20260818
    label_algorithm_version: str = "resonance-v1"
    response_hash_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        numeric_values = (
            self.frequency_start_hz,
            self.frequency_stop_hz,
            self.frequency_step_hz,
            self.response_hash_tolerance,
            self.n_folds,
            self.random_state,
        )
        if not all(np.isfinite(value) for value in numeric_values):
            raise ValueError("frequency and tolerance configuration must be finite")
        if self.frequency_step_hz <= 0:
            raise ValueError("frequency_step_hz must be positive")
        if self.frequency_start_hz >= self.frequency_stop_hz:
            raise ValueError("frequency_start_hz must be below frequency_stop_hz")
        intervals = (
            self.frequency_stop_hz - self.frequency_start_hz
        ) / self.frequency_step_hz
        if not np.isclose(intervals, round(intervals), rtol=0, atol=1e-10):
            raise ValueError("frequency step must evenly divide the configured band")
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least two")
        if self.response_hash_tolerance <= 0:
            raise ValueError("response_hash_tolerance must be positive")

    def frequency_grid(self) -> np.ndarray:
        count = int(round(
            (self.frequency_stop_hz - self.frequency_start_hz)
            / self.frequency_step_hz
        )) + 1
        return np.linspace(self.frequency_start_hz, self.frequency_stop_hz, count)
