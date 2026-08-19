from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from aln_model.metrics import SCALAR_TARGETS, TARGET_VALIDITY_COLUMNS
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
