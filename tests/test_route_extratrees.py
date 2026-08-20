from __future__ import annotations

from io import BytesIO
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest

from aln_model.metrics import SCALAR_TARGETS, TARGET_VALIDITY_COLUMNS
from aln_model.routes import extratrees
from aln_model.routes.extratrees import ExtraTreesRawRoute


def _features(count: int = 16) -> pd.DataFrame:
    signal = np.arange(count, dtype=float)
    return pd.DataFrame(
        {
            "Training_Row_ID": [f"row-{index:02d}" for index in range(count)],
            "Train_A_res_um2": signal,
            "Train_L_top_um": np.where(signal % 5 == 0, np.nan, signal + 10),
            "Template": np.where(signal % 2 == 0, "T11", "T01"),
            "EG_State": np.where(signal % 3 == 0, "patterned", "blanket"),
        }
    )


def _targets(features: pd.DataFrame) -> pd.DataFrame:
    signal = features["Train_A_res_um2"].to_numpy(float)
    fs = 4.8e9 + signal * 1e6
    fp = fs + 2e8
    frame = pd.DataFrame(
        {
            "Training_Row_ID": features["Training_Row_ID"],
            "fs_hz": fs,
            "fp_hz": fp,
            "k_eff2": 1 - np.square(fs / fp),
            "qs": 100 + signal,
            "qp": 200 + signal,
            "spurious_dangerous": signal >= len(signal) / 2,
        }
    )
    for column in TARGET_VALIDITY_COLUMNS.values():
        frame[column] = True
    return frame


def test_raw_route_predicts_contract_with_probabilities_and_physical_repair():
    train_features = _features()
    route = ExtraTreesRawRoute(n_estimators=12, random_state=7, n_jobs=1)
    route.fit(train_features, _targets(train_features))
    validation = pd.DataFrame(
        {
            "Training_Row_ID": ["validation-a", "validation-b"],
            "Train_A_res_um2": [2.5, np.nan],
            "Train_L_top_um": [np.nan, 19.0],
            "Template": ["UNSEEN_TEMPLATE", None],
            "EG_State": ["patterned", "UNSEEN_STATE"],
        }
    )

    prediction = route.predict(validation)

    assert prediction.spectrum is None
    assert prediction.scalars.columns.tolist() == ["Training_Row_ID", *SCALAR_TARGETS]
    assert prediction.scalars["Training_Row_ID"].tolist() == validation[
        "Training_Row_ID"
    ].tolist()
    values = prediction.scalars.loc[:, SCALAR_TARGETS].to_numpy(float)
    assert np.isfinite(values).all()
    assert (prediction.scalars["fp_hz"] > prediction.scalars["fs_hz"]).all()
    np.testing.assert_allclose(
        prediction.scalars["k_eff2"],
        1 - np.square(prediction.scalars["fs_hz"] / prediction.scalars["fp_hz"]),
    )
    assert (prediction.scalars[["qs", "qp"]] > 0).all().all()
    assert prediction.scalars["spurious_dangerous"].between(0, 1).all()


def test_fit_uses_each_target_validity_mask_and_not_invalid_outlier():
    features = _features()
    clean = _targets(features)
    contaminated = clean.copy()
    contaminated.loc[0, ["fs_hz", "fp_hz", "k_eff2", "qs", "qp"]] = 1e30
    contaminated.loc[0, "spurious_dangerous"] = True
    for column in TARGET_VALIDITY_COLUMNS.values():
        contaminated.loc[0, column] = False

    clean_route = ExtraTreesRawRoute(n_estimators=12, random_state=11, n_jobs=1).fit(
        features.iloc[1:].reset_index(drop=True),
        clean.iloc[1:].reset_index(drop=True),
    )
    masked_route = ExtraTreesRawRoute(n_estimators=12, random_state=11, n_jobs=1).fit(
        features, contaminated
    )

    expected = clean_route.predict(features.iloc[[3, 7]]).scalars
    actual = masked_route.predict(features.iloc[[3, 7]]).scalars
    np.testing.assert_allclose(
        actual.loc[:, SCALAR_TARGETS].to_numpy(float),
        expected.loc[:, SCALAR_TARGETS].to_numpy(float),
    )


def test_fit_rejects_target_ids_that_are_not_aligned_with_features():
    features = _features()
    targets = _targets(features).iloc[::-1].reset_index(drop=True)

    with pytest.raises(ValueError, match="Training_Row_ID order"):
        ExtraTreesRawRoute(n_estimators=4, n_jobs=1).fit(features, targets)


def test_all_missing_numeric_training_column_is_preserved_without_future_warning():
    features = _features()
    features["Train_PF_nm"] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        route = ExtraTreesRawRoute(n_estimators=4, n_jobs=1).fit(
            features, _targets(features)
        )

    prediction = route.predict(features.iloc[[0]])
    assert np.isfinite(prediction.scalars.loc[:, SCALAR_TARGETS].to_numpy(float)).all()


def test_fit_adds_absolute_error_q_heads_with_frozen_leaf_sizes_and_seeds():
    features = _features()
    route = ExtraTreesRawRoute(n_estimators=7, random_state=13, n_jobs=1).fit(
        features, _targets(features)
    )

    assert set(route.robust_q_regressors_) == {"qs", "qp"}
    for target, leaf_size in (("qs", 4), ("qp", 1)):
        primary = route.regressors_[target]
        robust = route.robust_q_regressors_[target]
        assert robust.criterion == "absolute_error"
        assert robust.min_samples_leaf == leaf_size
        assert robust.n_estimators == primary.n_estimators
        assert robust.max_features == primary.max_features
        assert robust.n_jobs == primary.n_jobs
        assert robust.random_state == primary.random_state


def test_q_blend_uses_fixed_qs_weight_and_patterned_non_t10_qp_weight():
    features = pd.DataFrame(
        {
            "EG_State": [
                " patterned ",
                "patterned",
                "blanket",
                "unknown",
                None,
                "patterned",
                "patterned",
                "patterned",
            ],
            "Template": [" t11 ", "T10", "T11", "T01", "T11", "UNKNOWN", None, " "],
        }
    )
    primary_qs = np.arange(100.0, 900.0, 100.0)
    robust_qs = primary_qs + 40.0
    primary_qp = np.arange(10.0, 90.0, 10.0)
    robust_qp = primary_qp + 4.0

    qs, qp = extratrees._blend_q_predictions(
        features,
        primary_qs,
        robust_qs,
        primary_qp,
        robust_qp,
        qs_robust_weight=0.25,
        qp_robust_weight=0.5,
        qp_robust_templates=frozenset({"T11", "T01", "T00"}),
        qp_robust_eg_state="patterned",
    )

    np.testing.assert_allclose(
        qs, [110.0, 210.0, 310.0, 410.0, 510.0, 610.0, 710.0, 810.0]
    )
    np.testing.assert_allclose(qp, [12.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])


def test_fitted_q_blend_configuration_is_stable_if_module_defaults_change(monkeypatch):
    features = _features()
    route = ExtraTreesRawRoute(n_estimators=12, random_state=19, n_jobs=1).fit(
        features, _targets(features)
    )
    validation = features.iloc[[1, 3, 6, 9]].copy()
    before = route.predict(validation).scalars[["qs", "qp"]].to_numpy()

    monkeypatch.setattr(extratrees, "_QS_ROBUST_WEIGHT", 1.0)
    monkeypatch.setattr(extratrees, "_QP_ROBUST_WEIGHT", 0.0)
    monkeypatch.setattr(extratrees, "_QP_ROBUST_TEMPLATES", frozenset())
    monkeypatch.setattr(
        extratrees, "_QP_ROBUST_EG_STATE", "blanket", raising=False
    )
    after = route.predict(validation).scalars[["qs", "qp"]].to_numpy()

    np.testing.assert_allclose(after, before)
    assert route.qs_robust_weight_ == 0.25
    assert route.qp_robust_weight_ == 0.5
    assert route.qp_robust_templates_ == frozenset({"T11", "T01", "T00"})
    assert route.qp_robust_eg_state_ == "patterned"


def test_joblib_roundtrip_and_legacy_route_without_robust_attrs_use_primary_q_heads():
    features = _features()
    route = ExtraTreesRawRoute(n_estimators=8, random_state=23, n_jobs=1).fit(
        features, _targets(features)
    )
    serialized = BytesIO()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated",
            category=DeprecationWarning,
            module="joblib.numpy_pickle",
        )
        joblib.dump(route, serialized)
        serialized.seek(0)
        loaded = joblib.load(serialized)
    validation = features.iloc[[2, 5, 8]].copy()
    expected_current = route.predict(validation).scalars[["qs", "qp"]].to_numpy()
    actual_current = loaded.predict(validation).scalars[["qs", "qp"]].to_numpy()
    np.testing.assert_allclose(actual_current, expected_current)

    for attribute in (
        "robust_q_regressors_",
        "qs_robust_weight_",
        "qp_robust_weight_",
        "qp_robust_templates_",
        "qp_robust_eg_state_",
    ):
        delattr(loaded, attribute)
    transformed = loaded.preprocessor_.transform(validation[loaded.feature_columns_])
    expected_legacy = np.column_stack(
        [loaded.regressors_[target].predict(transformed) for target in ("qs", "qp")]
    )

    actual_legacy = loaded.predict(validation).scalars[["qs", "qp"]].to_numpy()

    np.testing.assert_allclose(actual_legacy, expected_legacy)
    assert np.isfinite(actual_legacy).all()


def test_missing_fitted_qp_eg_state_uses_legacy_primary_q_heads():
    features = _features()
    route = ExtraTreesRawRoute(n_estimators=8, random_state=29, n_jobs=1).fit(
        features, _targets(features)
    )
    if hasattr(route, "qp_robust_eg_state_"):
        delattr(route, "qp_robust_eg_state_")
    validation = features.iloc[[1, 4, 7]].copy()
    transformed = route.preprocessor_.transform(validation[route.feature_columns_])
    expected = np.column_stack(
        [route.regressors_[target].predict(transformed) for target in ("qs", "qp")]
    )

    actual = route.predict(validation).scalars[["qs", "qp"]].to_numpy()

    np.testing.assert_allclose(actual, expected)
