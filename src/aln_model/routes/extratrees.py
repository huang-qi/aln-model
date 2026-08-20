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
_Q_TARGETS = ("qs", "qp")
_ROBUST_Q_MIN_SAMPLES_LEAF = {"qs": 4, "qp": 1}
_QS_ROBUST_WEIGHT = 0.25
_QP_ROBUST_WEIGHT = 0.5
_QP_ROBUST_TEMPLATES = frozenset({"T11", "T01", "T00"})
_QP_ROBUST_EG_STATE = "patterned"
_CLASSIFICATION_TARGET = "spurious_dangerous"


def _blend_q_predictions(
    features: pd.DataFrame,
    primary_qs: np.ndarray,
    robust_qs: np.ndarray,
    primary_qp: np.ndarray,
    robust_qp: np.ndarray,
    *,
    qs_robust_weight: float,
    qp_robust_weight: float,
    qp_robust_templates: frozenset[str],
    qp_robust_eg_state: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the frozen robust-head blend without changing non-Q predictions."""

    qs = (1.0 - qs_robust_weight) * primary_qs + qs_robust_weight * robust_qs
    eg_state = features.get("EG_State", pd.Series("", index=features.index))
    template = features.get("Template", pd.Series("", index=features.index))
    eligible_eg_state = (
        eg_state.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(qp_robust_eg_state.strip().lower())
    )
    normalized_template = template.fillna("").astype(str).str.strip().str.upper()
    use_robust_qp = (
        eligible_eg_state & normalized_template.isin(qp_robust_templates)
    ).to_numpy()
    qp_blend = (
        (1.0 - qp_robust_weight) * primary_qp + qp_robust_weight * robust_qp
    )
    qp = np.where(use_robust_qp, qp_blend, primary_qp)
    return np.asarray(qs, dtype=float), np.asarray(qp, dtype=float)


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

        self.robust_q_regressors_: dict[str, ExtraTreesRegressor] = {}
        self.qs_robust_weight_ = _QS_ROBUST_WEIGHT
        self.qp_robust_weight_ = _QP_ROBUST_WEIGHT
        self.qp_robust_templates_ = _QP_ROBUST_TEMPLATES
        self.qp_robust_eg_state_ = _QP_ROBUST_EG_STATE
        for target in _Q_TARGETS:
            values = pd.to_numeric(targets[target], errors="coerce").to_numpy(float)
            mask = self._target_mask(targets, target) & np.isfinite(values)
            primary = self.regressors_[target]
            model = ExtraTreesRegressor(
                n_estimators=primary.n_estimators,
                criterion="absolute_error",
                max_features=primary.max_features,
                random_state=primary.random_state,
                n_jobs=primary.n_jobs,
                min_samples_leaf=_ROBUST_Q_MIN_SAMPLES_LEAF[target],
            )
            model.fit(transformed[mask], values[mask])
            self.robust_q_regressors_[target] = model

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
        robust_q_regressors = getattr(self, "robust_q_regressors_", None)
        has_fitted_q_blend = (
            isinstance(robust_q_regressors, dict)
            and all(target in robust_q_regressors for target in _Q_TARGETS)
            and hasattr(self, "qs_robust_weight_")
            and hasattr(self, "qp_robust_weight_")
            and hasattr(self, "qp_robust_templates_")
            and hasattr(self, "qp_robust_eg_state_")
        )
        if has_fitted_q_blend:
            robust_q = {
                target: robust_q_regressors[target].predict(transformed)
                for target in _Q_TARGETS
            }
            qs, qp = _blend_q_predictions(
                features,
                raw["qs"],
                robust_q["qs"],
                raw["qp"],
                robust_q["qp"],
                qs_robust_weight=self.qs_robust_weight_,
                qp_robust_weight=self.qp_robust_weight_,
                qp_robust_templates=self.qp_robust_templates_,
                qp_robust_eg_state=self.qp_robust_eg_state_,
            )
        else:
            qs, qp = raw["qs"], raw["qp"]
        qs = np.maximum(qs, np.finfo(float).tiny)
        qp = np.maximum(qp, np.finfo(float).tiny)
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
