from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from ..features import build_physics_features
from ..metrics import TARGET_VALIDITY_COLUMNS
from .base import RoutePrediction, reconstruct_physical_targets


@dataclass
class FunctionalPCARoute:
    """Joint scalar/spectrum surrogate with train-fold-only functional PCA."""

    n_spectrum_components: int = 24
    ridge_alpha: float = 10.0
    random_state: int = 0

    def __post_init__(self) -> None:
        if self.n_spectrum_components < 1:
            raise ValueError("n_spectrum_components must be positive")
        if self.ridge_alpha < 0:
            raise ValueError("ridge_alpha must be non-negative")
        self._is_fitted = False

    @staticmethod
    def _ids(frame: pd.DataFrame, name: str) -> np.ndarray:
        if "Training_Row_ID" not in frame:
            raise ValueError(f"{name} missing Training_Row_ID")
        raw = frame["Training_Row_ID"]
        if raw.isna().any() or raw.astype(str).str.strip().eq("").any():
            raise ValueError(f"{name} contains blank Training_Row_ID")
        ids = raw.astype(str).str.strip().to_numpy()
        if len(set(ids)) != len(ids):
            raise ValueError(f"{name} contains duplicate Training_Row_ID")
        return ids

    def _fit_features(self, features: pd.DataFrame) -> np.ndarray:
        values = build_physics_features(features).to_numpy(dtype=float)
        self._imputer = SimpleImputer(strategy="median")
        self._scaler = StandardScaler()
        return self._scaler.fit_transform(self._imputer.fit_transform(values))

    def _transform_features(self, features: pd.DataFrame) -> np.ndarray:
        values = build_physics_features(features).to_numpy(dtype=float)
        return self._scaler.transform(self._imputer.transform(values))

    def fit(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        *,
        spectra: np.ndarray | None = None,
        frequency_hz: np.ndarray | None = None,
    ) -> FunctionalPCARoute:
        feature_ids = self._ids(features, "features")
        target_ids = self._ids(targets, "targets")
        if not np.array_equal(feature_ids, target_ids):
            raise ValueError("features and targets Training_Row_ID order differs")
        if spectra is None:
            raise ValueError("spectra are required for functional PCA")
        spectrum = np.asarray(spectra)
        if spectrum.ndim != 2 or spectrum.shape[0] != len(features):
            raise ValueError("spectra must be a row-aligned two-dimensional matrix")
        if not np.iscomplexobj(spectrum) or not (
            np.isfinite(spectrum.real).all() and np.isfinite(spectrum.imag).all()
        ):
            raise ValueError("spectra must contain finite complex values")
        if frequency_hz is None:
            raise ValueError("frequency grid is required for functional PCA")
        frequency = np.asarray(frequency_hz, dtype=float)
        if (
            frequency.ndim != 1
            or frequency.size != spectrum.shape[1]
            or not np.isfinite(frequency).all()
            or not np.all(np.diff(frequency) > 0)
        ):
            raise ValueError("frequency grid must match spectra and strictly increase")

        required = {"fs_hz", "fp_hz", "qs", "qp", "spurious_dangerous"}
        missing = required.difference(targets.columns)
        if missing:
            raise ValueError(f"targets missing required columns: {sorted(missing)}")
        fs = pd.to_numeric(targets["fs_hz"], errors="coerce").to_numpy(float)
        fp = pd.to_numeric(targets["fp_hz"], errors="coerce").to_numpy(float)
        qs = pd.to_numeric(targets["qs"], errors="coerce").to_numpy(float)
        qp = pd.to_numeric(targets["qp"], errors="coerce").to_numpy(float)
        def validity(target: str) -> np.ndarray:
            column = TARGET_VALIDITY_COLUMNS[target]
            if column not in targets:
                return np.ones(len(targets), dtype=bool)
            return targets[column].fillna(False).astype(bool).to_numpy()

        frequency_mask = (
            validity("fs_hz")
            & validity("fp_hz")
            & np.isfinite(fs)
            & np.isfinite(fp)
            & (fs > 0)
            & (fp > fs)
        )
        qs_mask = validity("qs") & np.isfinite(qs) & (qs > 0)
        qp_mask = validity("qp") & np.isfinite(qp) & (qp > 0)
        if not frequency_mask.any() or not qs_mask.any() or not qp_mask.any():
            raise ValueError("one or more scalar targets have no valid physical rows")

        design = self._fit_features(features)
        frequency_targets = np.column_stack(((fs + fp) / 2, np.log(fp - fs)))
        self._frequency_model = Ridge(alpha=self.ridge_alpha).fit(
            design[frequency_mask], frequency_targets[frequency_mask]
        )
        self._qs_model = Ridge(alpha=self.ridge_alpha).fit(
            design[qs_mask], np.log(qs[qs_mask])
        )
        self._qp_model = Ridge(alpha=self.ridge_alpha).fit(
            design[qp_mask], np.log(qp[qp_mask])
        )

        functional = np.concatenate((spectrum.real, spectrum.imag), axis=1)
        component_count = min(
            self.n_spectrum_components,
            functional.shape[0] - 1,
            functional.shape[1],
        )
        if component_count < 1:
            raise ValueError("at least two spectra are required for functional PCA")
        self._spectrum_pca = PCA(
            n_components=component_count,
            svd_solver="randomized" if component_count < min(functional.shape) else "full",
            random_state=self.random_state,
        )
        scores = self._spectrum_pca.fit_transform(functional)
        self._spectrum_model = Ridge(alpha=self.ridge_alpha).fit(design, scores)

        labels = pd.to_numeric(
            targets["spurious_dangerous"], errors="coerce"
        ).to_numpy(float)
        spurious_mask = validity("spurious_dangerous") & np.isfinite(labels)
        if not spurious_mask.any() or not np.isin(
            labels[spurious_mask], [0.0, 1.0]
        ).all():
            raise ValueError("spurious_dangerous has no valid binary labels")
        unique = np.unique(labels[spurious_mask])
        self._constant_spurious = float(unique[0]) if unique.size == 1 else None
        self._spurious_model = None
        if unique.size > 1:
            self._spurious_model = LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=self.random_state,
            ).fit(design[spurious_mask], labels[spurious_mask])
        self.training_counts_ = {
            "frequency": int(frequency_mask.sum()),
            "qs": int(qs_mask.sum()),
            "qp": int(qp_mask.sum()),
            "spurious": int(spurious_mask.sum()),
        }
        self._spectrum_width = spectrum.shape[1]
        self.frequency_hz_ = frequency.copy()
        self._is_fitted = True
        return self

    def predict(self, features: pd.DataFrame) -> RoutePrediction:
        if not self._is_fitted:
            raise RuntimeError("route must be fitted before prediction")
        ids = self._ids(features, "features")
        design = self._transform_features(features)
        frequency = self._frequency_model.predict(design)
        scalar = reconstruct_physical_targets(
            center_frequency_hz=frequency[:, 0],
            log_bandwidth_hz=frequency[:, 1],
            log_qs=self._qs_model.predict(design),
            log_qp=self._qp_model.predict(design),
        )
        if self._spurious_model is None:
            probability = np.full(len(features), self._constant_spurious, dtype=float)
        else:
            positive_column = int(np.flatnonzero(self._spurious_model.classes_ == 1)[0])
            probability = self._spurious_model.predict_proba(design)[:, positive_column]
        scalar.insert(0, "Training_Row_ID", ids)
        scalar["spurious_dangerous"] = probability

        scores = self._spectrum_model.predict(design)
        if scores.ndim == 1:
            scores = scores[:, None]
        functional = self._spectrum_pca.inverse_transform(scores)
        width = self._spectrum_width
        spectrum = functional[:, :width] + 1j * functional[:, width:]
        # A passive one-port cannot have reflection magnitude above one. Ridge
        # reconstruction is unconstrained, so project only violating points to
        # the unit disk while retaining their predicted phase.
        magnitude = np.abs(spectrum)
        violating = magnitude > 1.0
        spectrum[violating] /= magnitude[violating]
        return RoutePrediction(scalars=scalar, spectrum=spectrum)
