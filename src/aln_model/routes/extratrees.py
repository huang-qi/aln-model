from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ..metrics import SCALAR_TARGETS, TARGET_VALIDITY_COLUMNS
from .base import RoutePrediction


_ID_COLUMN = "Training_Row_ID"
# k_eff2 is reconstructed exactly from fs/fp at prediction time, so fitting a
# sixth forest for it would consume compute without affecting the output.
_REGRESSION_TARGETS = ("fs_hz", "fp_hz", "qs", "qp")
_CLASSIFICATION_TARGET = "spurious_dangerous"


class ExtraTreesRawRoute:
    """ExtraTrees baseline over the raw prepared geometry and state columns."""

    def __init__(
        self,
        *,
        n_estimators: int = 300,
        random_state: int = 0,
        n_jobs: int | None = -1,
        min_samples_leaf: int = 1,
    ) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.min_samples_leaf = min_samples_leaf

    def fit(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        *,
        spectra: np.ndarray | None = None,
        frequency_hz: np.ndarray | None = None,
    ) -> ExtraTreesRawRoute:
        feature_ids = self._ids(features, "features")
        target_ids = self._ids(targets, "targets")
        if feature_ids.tolist() != target_ids.tolist():
            raise ValueError("targets Training_Row_ID order does not match features")
        missing_targets = set(SCALAR_TARGETS).difference(targets.columns)
        if missing_targets:
            raise ValueError(f"targets missing scalar columns: {sorted(missing_targets)}")

        self.feature_columns_ = [column for column in features if column != _ID_COLUMN]
        if not self.feature_columns_:
            raise ValueError("features must contain at least one predictor column")
        numeric_columns = [
            column
            for column in self.feature_columns_
            if is_numeric_dtype(features[column].dtype)
        ]
        categorical_columns = [
            column for column in self.feature_columns_ if column not in numeric_columns
        ]
        transformers: list[tuple[str, object, list[str]]] = []
        if numeric_columns:
            transformers.append(
                (
                    "numeric",
                    SimpleImputer(
                        strategy="constant", fill_value=0.0, keep_empty_features=True
                    ),
                    numeric_columns,
                )
            )
        if categorical_columns:
            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        [
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="constant",
                                    fill_value="__MISSING__",
                                    keep_empty_features=True,
                                ),
                            ),
                            (
                                "onehot",
                                OneHotEncoder(handle_unknown="ignore"),
                            ),
                        ]
                    ),
                    categorical_columns,
                )
            )
        self.preprocessor_ = ColumnTransformer(transformers)
        transformed = self.preprocessor_.fit_transform(features[self.feature_columns_])

        self.regressors_: dict[str, ExtraTreesRegressor] = {}
        for offset, target in enumerate(_REGRESSION_TARGETS):
            values = pd.to_numeric(targets[target], errors="coerce").to_numpy(float)
            mask = self._target_mask(targets, target) & np.isfinite(values)
            if not mask.any():
                raise ValueError(f"target {target} has no valid training rows")
            model = ExtraTreesRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state + offset,
                n_jobs=self.n_jobs,
                min_samples_leaf=self.min_samples_leaf,
            )
            model.fit(transformed[mask], values[mask])
            self.regressors_[target] = model

        class_values = pd.to_numeric(
            targets[_CLASSIFICATION_TARGET], errors="coerce"
        ).to_numpy(float)
        class_mask = self._target_mask(targets, _CLASSIFICATION_TARGET) & np.isfinite(
            class_values
        )
        if not class_mask.any():
            raise ValueError(
                f"target {_CLASSIFICATION_TARGET} has no valid training rows"
            )
        self.classifier_ = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state + len(_REGRESSION_TARGETS),
            n_jobs=self.n_jobs,
            min_samples_leaf=self.min_samples_leaf,
        )
        self.classifier_.fit(transformed[class_mask], class_values[class_mask] >= 0.5)
        return self

    def predict(self, features: pd.DataFrame) -> RoutePrediction:
        if not hasattr(self, "preprocessor_"):
            raise RuntimeError("route must be fitted before predict")
        ids = self._ids(features, "features")
        missing = set(self.feature_columns_).difference(features.columns)
        if missing:
            raise ValueError(f"features missing fitted columns: {sorted(missing)}")
        transformed = self.preprocessor_.transform(features[self.feature_columns_])
        raw = {
            target: self.regressors_[target].predict(transformed)
            for target in _REGRESSION_TARGETS
        }

        fs_hz = np.minimum(raw["fs_hz"], raw["fp_hz"])
        fp_hz = np.maximum(raw["fs_hz"], raw["fp_hz"])
        tied = fp_hz <= fs_hz
        fp_hz[tied] = np.nextafter(fs_hz[tied], np.inf)
        qs = np.maximum(raw["qs"], np.finfo(float).tiny)
        qp = np.maximum(raw["qp"], np.finfo(float).tiny)
        k_eff2 = 1 - np.square(fs_hz / fp_hz)

        class_positions = np.flatnonzero(self.classifier_.classes_ == True)  # noqa: E712
        if class_positions.size:
            spurious_probability = self.classifier_.predict_proba(transformed)[
                :, class_positions[0]
            ]
        else:
            spurious_probability = np.zeros(len(features), dtype=float)
        scalars = pd.DataFrame(
            {
                _ID_COLUMN: ids.to_numpy(),
                "fs_hz": fs_hz,
                "fp_hz": fp_hz,
                "k_eff2": k_eff2,
                "qs": qs,
                "qp": qp,
                "spurious_dangerous": spurious_probability,
            }
        )
        return RoutePrediction(scalars=scalars)

    @staticmethod
    def _ids(frame: pd.DataFrame, name: str) -> pd.Series:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame")
        if _ID_COLUMN not in frame:
            raise ValueError(f"{name} missing {_ID_COLUMN}")
        raw = frame[_ID_COLUMN]
        if raw.isna().any() or raw.astype(str).str.strip().eq("").any():
            raise ValueError(f"{name} contains blank {_ID_COLUMN}")
        ids = raw.astype(str).str.strip()
        if ids.duplicated().any():
            raise ValueError(f"{name} contains duplicate {_ID_COLUMN}")
        return ids

    @staticmethod
    def _target_mask(targets: pd.DataFrame, target: str) -> np.ndarray:
        validity_column = TARGET_VALIDITY_COLUMNS[target]
        if validity_column not in targets:
            return np.ones(len(targets), dtype=bool)
        return targets[validity_column].fillna(False).astype(bool).to_numpy()


ExtraTreesRoute = ExtraTreesRawRoute

__all__ = ["ExtraTreesRawRoute", "ExtraTreesRoute"]
