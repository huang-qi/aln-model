from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from ..features import build_physics_features
from .base import RoutePrediction, reconstruct_physical_targets


def _strict_ids(frame: pd.DataFrame, name: str) -> pd.Series:
    if "Training_Row_ID" not in frame:
        raise ValueError(f"{name} missing Training_Row_ID")
    ids = frame["Training_Row_ID"]
    if ids.isna().any() or not ids.map(lambda value: isinstance(value, str)).all():
        raise ValueError(f"{name} Training_Row_ID values must be canonical strings")
    if ids.str.strip().eq("").any() or not ids.str.strip().equals(ids):
        raise ValueError(f"{name} contains noncanonical Training_Row_ID")
    if ids.duplicated().any():
        raise ValueError(f"{name} contains duplicate Training_Row_ID")
    return ids


class PhysicsBoostRoute:
    """Physics-feature boosted trees with constrained scalar target transforms."""

    def __init__(self, *, random_state: int = 0, max_iter: int = 100) -> None:
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        *,
        spectra: np.ndarray | None = None,
        frequency_hz: np.ndarray | None = None,
    ) -> PhysicsBoostRoute:
        feature_ids = _strict_ids(features, "features")
        target_ids = _strict_ids(targets, "targets")
        if set(feature_ids) != set(target_ids):
            raise ValueError("targets Training_Row_ID set does not match features")
        targets = targets.set_index("Training_Row_ID").loc[feature_ids].reset_index()
        x = build_physics_features(features).replace([np.inf, -np.inf], np.nan)
        fs = pd.to_numeric(targets["fs_hz"], errors="coerce").to_numpy(float)
        fp = pd.to_numeric(targets["fp_hz"], errors="coerce").to_numpy(float)
        qs = pd.to_numeric(targets["qs"], errors="coerce").to_numpy(float)
        qp = pd.to_numeric(targets["qp"], errors="coerce").to_numpy(float)
        valid_fs = targets.get("target_valid_fs", True)
        valid_fp = targets.get("target_valid_fp", True)
        valid_qs = targets.get("target_valid_qs", True)
        valid_qp = targets.get("target_valid_qp", True)
        frequency_mask = (
            np.asarray(valid_fs, dtype=bool)
            & np.asarray(valid_fp, dtype=bool)
            & np.isfinite(fs)
            & np.isfinite(fp)
            & (fs > 0)
            & (fp > fs)
        )
        qs_mask = np.asarray(valid_qs, dtype=bool) & np.isfinite(qs) & (qs > 0)
        qp_mask = np.asarray(valid_qp, dtype=bool) & np.isfinite(qp) & (qp > 0)
        log_bandwidth = np.full(len(targets), np.nan)
        log_bandwidth[frequency_mask] = np.log(
            fp[frequency_mask] - fs[frequency_mask]
        )
        log_qs = np.full(len(targets), np.nan)
        log_qs[qs_mask] = np.log(qs[qs_mask])
        log_qp = np.full(len(targets), np.nan)
        log_qp[qp_mask] = np.log(qp[qp_mask])
        transformed = {
            "center_frequency_hz": ((fs + fp) / 2, frequency_mask),
            "log_bandwidth_hz": (log_bandwidth, frequency_mask),
            "log_qs": (log_qs, qs_mask),
            "log_qp": (log_qp, qp_mask),
        }
        self.regressors_ = {}
        for name, (values, mask) in transformed.items():
            if not np.any(mask):
                raise ValueError(f"no valid training targets for {name}")
            model = HistGradientBoostingRegressor(
                max_iter=self.max_iter, random_state=self.random_state
            )
            model.fit(x.loc[mask], values[mask])
            self.regressors_[name] = model

        raw_spurious = pd.to_numeric(
            targets["spurious_dangerous"], errors="coerce"
        ).to_numpy(float)
        valid_spurious = np.asarray(
            targets.get("target_valid_spurious", True), dtype=bool
        ) & np.isfinite(raw_spurious)
        if not np.any(valid_spurious):
            raise ValueError("no valid training targets for spurious_dangerous")
        if not np.isin(raw_spurious[valid_spurious], [0.0, 1.0]).all():
            raise ValueError("spurious_dangerous must contain binary labels")
        spurious = raw_spurious[valid_spurious].astype(bool)
        self.spurious_probability_ = float(np.mean(spurious))
        self.spurious_model_ = None
        if np.unique(spurious).size == 2:
            self.spurious_model_ = HistGradientBoostingClassifier(
                max_iter=self.max_iter, random_state=self.random_state
            ).fit(x.loc[valid_spurious], spurious)
        return self

    def predict(self, features: pd.DataFrame) -> RoutePrediction:
        ids = _strict_ids(features, "features")
        x = build_physics_features(features).replace([np.inf, -np.inf], np.nan)
        transformed = {
            name: model.predict(x) for name, model in self.regressors_.items()
        }
        scalars = reconstruct_physical_targets(**transformed)
        scalars["spurious_dangerous"] = (
            np.full(len(x), self.spurious_probability_)
            if self.spurious_model_ is None
            else self.spurious_model_.predict_proba(x)[:, 1]
        )
        scalars.insert(
            0, "Training_Row_ID", ids.to_numpy(copy=True)
        )
        return RoutePrediction(scalars=scalars)
