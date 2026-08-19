from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from ..features import build_physics_features
from ..metrics import TARGET_VALIDITY_COLUMNS
from .base import RoutePrediction, reconstruct_physical_targets


class RFFKernelRoute:
    """Smooth nonlinear scalar route using random Fourier RBF features."""

    def __init__(
        self,
        *,
        n_components: int = 256,
        gamma: float | None = None,
        ridge_alpha: float = 1.0,
        logistic_c: float = 1.0,
        random_state: int = 0,
    ) -> None:
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        if gamma is not None and gamma <= 0:
            raise ValueError("gamma must be positive")
        if ridge_alpha < 0:
            raise ValueError("ridge_alpha must be non-negative")
        if logistic_c <= 0:
            raise ValueError("logistic_c must be positive")
        self.n_components = n_components
        self.gamma = gamma
        self.ridge_alpha = ridge_alpha
        self.logistic_c = logistic_c
        self.random_state = random_state

    @staticmethod
    def _ids(frame: pd.DataFrame, name: str) -> pd.Series:
        if "Training_Row_ID" not in frame:
            raise ValueError(f"{name} missing Training_Row_ID")
        ids = frame["Training_Row_ID"]
        if ids.isna().any() or not ids.map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"{name} Training_Row_ID values must be canonical strings")
        if ids.str.strip().eq("").any() or not ids.str.strip().equals(ids):
            raise ValueError(f"{name} contains blank Training_Row_ID")
        if ids.duplicated().any():
            raise ValueError(f"{name} contains duplicate Training_Row_ID")
        return ids.copy()

    @staticmethod
    def _numeric_features(features: pd.DataFrame) -> np.ndarray:
        matrix = build_physics_features(features).to_numpy(dtype=float)
        matrix[~np.isfinite(matrix)] = np.nan
        return matrix

    @staticmethod
    def _target_mask(targets: pd.DataFrame, target: str) -> np.ndarray:
        values = pd.to_numeric(targets[target], errors="coerce").to_numpy(float)
        mask = np.isfinite(values)
        validity = TARGET_VALIDITY_COLUMNS[target]
        if validity in targets:
            mask &= targets[validity].fillna(False).astype(bool).to_numpy()
        return mask

    def fit(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        *,
        spectra: np.ndarray | None = None,
        frequency_hz: np.ndarray | None = None,
    ) -> RFFKernelRoute:
        feature_ids = self._ids(features, "features")
        target_ids = self._ids(targets, "targets")
        if set(feature_ids.astype(str)) != set(target_ids.astype(str)):
            raise ValueError("targets Training_Row_ID set does not match features")
        target_positions = pd.Series(
            np.arange(len(target_ids)), index=target_ids.astype(str).to_numpy()
        )
        targets = targets.iloc[
            target_positions.loc[feature_ids.astype(str)].to_numpy(dtype=int)
        ].reset_index(drop=True)

        missing_targets = {
            "fs_hz",
            "fp_hz",
            "qs",
            "qp",
            "spurious_dangerous",
        }.difference(targets.columns)
        if missing_targets:
            raise ValueError(f"targets missing columns: {sorted(missing_targets)}")

        raw_features = self._numeric_features(features)
        self.imputer_ = SimpleImputer(strategy="median")
        imputed = self.imputer_.fit_transform(raw_features)
        self.scaler_ = StandardScaler()
        scaled = self.scaler_.fit_transform(imputed)
        resolved_gamma = self.gamma if self.gamma is not None else 1.0 / scaled.shape[1]
        self.rff_ = RBFSampler(
            gamma=resolved_gamma,
            n_components=self.n_components,
            random_state=self.random_state,
        )
        kernel_features = self.rff_.fit_transform(scaled)

        fs = pd.to_numeric(targets["fs_hz"], errors="coerce").to_numpy(float)
        fp = pd.to_numeric(targets["fp_hz"], errors="coerce").to_numpy(float)
        frequency_mask = (
            self._target_mask(targets, "fs_hz")
            & self._target_mask(targets, "fp_hz")
            & (fs > 0)
            & (fp > fs)
        )
        log_bandwidth = np.full(len(targets), np.nan)
        log_bandwidth[frequency_mask] = np.log(
            fp[frequency_mask] - fs[frequency_mask]
        )
        transformed = {
            "center_frequency_hz": ((fs + fp) / 2.0, frequency_mask),
            "log_bandwidth_hz": (log_bandwidth, frequency_mask),
        }
        for target in ("qs", "qp"):
            values = pd.to_numeric(targets[target], errors="coerce").to_numpy(float)
            mask = self._target_mask(targets, target) & (values > 0)
            logged = np.full(len(targets), np.nan)
            logged[mask] = np.log(values[mask])
            transformed[f"log_{target}"] = (
                logged,
                mask,
            )

        self.regressors_: dict[str, Ridge] = {}
        for name, (values, mask) in transformed.items():
            if not np.any(mask):
                raise ValueError(f"no valid training rows for {name}")
            model = Ridge(alpha=self.ridge_alpha)
            model.fit(kernel_features[mask], values[mask])
            self.regressors_[name] = model

        class_mask = self._target_mask(targets, "spurious_dangerous")
        classes = pd.to_numeric(
            targets["spurious_dangerous"], errors="coerce"
        ).to_numpy(float)
        classes = classes[class_mask]
        if classes.size == 0 or not np.isin(classes, [0.0, 1.0]).all():
            raise ValueError(
                "spurious_dangerous must contain valid binary training labels"
            )
        unique_classes = np.unique(classes)
        self.constant_spurious_probability_: float | None = None
        self.classifier_: LogisticRegression | None = None
        if unique_classes.size == 1:
            self.constant_spurious_probability_ = float(unique_classes[0])
        else:
            self.classifier_ = LogisticRegression(
                C=self.logistic_c,
                max_iter=1000,
                random_state=self.random_state,
            )
            self.classifier_.fit(kernel_features[class_mask], classes.astype(int))
        self._fitted = True
        return self

    def predict(self, features: pd.DataFrame) -> RoutePrediction:
        if not getattr(self, "_fitted", False):
            raise RuntimeError("RFFKernelRoute must be fitted before predict")
        ids = self._ids(features, "features")
        raw_features = self._numeric_features(features)
        imputed = self.imputer_.transform(raw_features)
        scaled = self.scaler_.transform(imputed)
        kernel_features = self.rff_.transform(scaled)
        transformed = {
            name: model.predict(kernel_features)
            for name, model in self.regressors_.items()
        }

        largest = np.finfo(float).max / 8.0
        log_bandwidth = np.clip(
            transformed["log_bandwidth_hz"], -700.0, np.log(largest)
        )
        bandwidth = np.exp(log_bandwidth)
        center = np.nan_to_num(
            transformed["center_frequency_hz"], nan=0.0, posinf=largest, neginf=0.0
        )
        center = np.maximum(center, np.nextafter(bandwidth / 2.0, np.inf))
        center = np.minimum(center, largest)
        physical = reconstruct_physical_targets(
            center_frequency_hz=center,
            log_bandwidth_hz=log_bandwidth,
            log_qs=np.clip(transformed["log_qs"], -700.0, 700.0),
            log_qp=np.clip(transformed["log_qp"], -700.0, 700.0),
        )
        if self.constant_spurious_probability_ is not None:
            probability = np.full(len(features), self.constant_spurious_probability_)
        else:
            assert self.classifier_ is not None
            positive_index = int(np.flatnonzero(self.classifier_.classes_ == 1)[0])
            probability = self.classifier_.predict_proba(kernel_features)[
                :, positive_index
            ]
        physical.insert(0, "Training_Row_ID", ids.to_numpy(copy=True))
        physical["spurious_dangerous"] = probability
        return RoutePrediction(scalars=physical)


__all__ = ["RFFKernelRoute"]
