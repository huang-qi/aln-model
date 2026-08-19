from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RoutePrediction:
    """One route's scalar predictions and optional shared-grid S11 prediction."""

    scalars: pd.DataFrame
    spectrum: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scalars, pd.DataFrame):
            raise TypeError("RoutePrediction scalars must be a pandas DataFrame")
        if "Training_Row_ID" not in self.scalars:
            raise ValueError("RoutePrediction scalars must contain Training_Row_ID")


RouteT = TypeVar("RouteT", bound="Route")


@runtime_checkable
class Route(Protocol):
    """Contract implemented by every independent modeling route."""

    def fit(
        self: RouteT,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        *,
        spectra: np.ndarray | None = None,
        frequency_hz: np.ndarray | None = None,
    ) -> RouteT:
        ...

    def predict(self, features: pd.DataFrame) -> RoutePrediction:
        ...


def reconstruct_physical_targets(
    *,
    center_frequency_hz: np.ndarray,
    log_bandwidth_hz: np.ndarray,
    log_qs: np.ndarray,
    log_qp: np.ndarray,
) -> pd.DataFrame:
    """Invert constrained target transforms used by physics-aware routes."""

    arrays = (
        np.asarray(center_frequency_hz, dtype=float),
        np.asarray(log_bandwidth_hz, dtype=float),
        np.asarray(log_qs, dtype=float),
        np.asarray(log_qp, dtype=float),
    )
    shapes = [value.shape for value in arrays]
    all_scalars = all(value.ndim == 0 for value in arrays)
    if not all_scalars and len(set(shapes)) != 1:
        raise ValueError("transformed target inputs must have the same shape or all be scalars")
    center, log_bandwidth, qs_log, qp_log = arrays
    with np.errstate(over="ignore", invalid="ignore"):
        bandwidth = np.exp(log_bandwidth)
        qs = np.exp(qs_log)
        qp = np.exp(qp_log)
        fs_hz = center - bandwidth / 2
        fp_hz = center + bandwidth / 2
        k_eff2 = 1 - np.square(fs_hz / fp_hz)
    valid = (
        np.isfinite(fs_hz)
        & np.isfinite(fp_hz)
        & np.isfinite(k_eff2)
        & np.isfinite(qs)
        & np.isfinite(qp)
        & (fs_hz > 0)
        & (fp_hz > fs_hz)
        & (k_eff2 >= 0)
        & (k_eff2 <= 1)
        & (qs > 0)
        & (qp > 0)
    )
    if not np.all(valid):
        raise ValueError("transformed targets do not reconstruct to physical values")
    return pd.DataFrame(
        {
            "fs_hz": fs_hz.ravel(),
            "fp_hz": fp_hz.ravel(),
            "k_eff2": k_eff2.ravel(),
            "qs": qs.ravel(),
            "qp": qp.ravel(),
        }
    )
