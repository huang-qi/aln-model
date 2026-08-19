from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import warnings

from aln_model.features import PHYSICAL_COLUMNS, STATE_COLUMNS
from aln_model.routes.rff_kernel import RFFKernelRoute


def _features(ids: list[str], offset: float = 0.0) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, row_id in enumerate(ids):
        value = float(index) + offset
        row: dict[str, object] = {
            "Training_Row_ID": row_id,
            "Template": ("T11", "T10", "T01", "T00")[index % 4],
            "EG_State": "patterned" if index % 2 else "blanket",
            "has_FL": index % 2,
            "has_AG": 1,
            "has_patterned_EG": index % 2,
            "EG_material_present": 1,
            "has_PF": index % 3 == 0,
            "is_BASIC": index % 2 == 0,
        }
        for column_index, column in enumerate(PHYSICAL_COLUMNS):
            row[column] = 10.0 + column_index + value
        if not row["has_PF"]:
            row["Train_PF_nm"] = np.nan
        rows.append(row)
    return pd.DataFrame(
        rows, columns=["Training_Row_ID", *STATE_COLUMNS, *PHYSICAL_COLUMNS]
    )


def _targets(ids: list[str]) -> pd.DataFrame:
    index = np.arange(len(ids), dtype=float)
    fs = 5.0e9 + index * 1.0e6
    fp = fs + 3.5e8 + index * 2.0e5
    return pd.DataFrame(
        {
            "Training_Row_ID": ids,
            "fs_hz": fs,
            "fp_hz": fp,
            "k_eff2": 1.0 - np.square(fs / fp),
            "qs": 500.0 + index,
            "qp": 250.0 + index,
            "spurious_dangerous": (index % 2).astype(bool),
            "target_valid_fs": True,
            "target_valid_fp": True,
            "target_valid_k_eff2": True,
            "target_valid_qs": True,
            "target_valid_qp": True,
            "target_valid_spurious": True,
        }
    )


def test_predict_preserves_ids_and_reconstructs_physical_targets() -> None:
    train_ids = [f"train-{index}" for index in range(8)]
    predict_ids = ["prediction-z", "prediction-a"]
    route = RFFKernelRoute(n_components=16, random_state=7)

    route.fit(_features(train_ids), _targets(train_ids))
    prediction = route.predict(_features(predict_ids, offset=0.25)).scalars

    assert prediction["Training_Row_ID"].tolist() == predict_ids
    assert prediction.columns.tolist() == [
        "Training_Row_ID",
        "fs_hz",
        "fp_hz",
        "k_eff2",
        "qs",
        "qp",
        "spurious_dangerous",
    ]
    assert np.all(prediction["fs_hz"] > 0)
    assert np.all(prediction["fp_hz"] > prediction["fs_hz"])
    np.testing.assert_allclose(
        prediction["k_eff2"],
        1.0 - np.square(prediction["fs_hz"] / prediction["fp_hz"]),
    )
    assert np.all(prediction[["qs", "qp"]] > 0)
    assert np.all(prediction["spurious_dangerous"].between(0.0, 1.0))


def test_random_state_is_reproducible_and_targets_align_by_id() -> None:
    train_ids = [f"train-{index}" for index in range(10)]
    features = _features(train_ids)
    targets = (
        _targets(train_ids).sample(frac=1.0, random_state=99).reset_index(drop=True)
    )
    validation = _features(["validation-2", "validation-1"], offset=0.5)

    first = RFFKernelRoute(n_components=12, random_state=11).fit(features, targets)
    second = RFFKernelRoute(n_components=12, random_state=11).fit(features, targets)

    pd.testing.assert_frame_equal(
        first.predict(validation).scalars,
        second.predict(validation).scalars,
    )


def test_single_class_fold_returns_constant_spurious_probability() -> None:
    train_ids = [f"train-{index}" for index in range(6)]
    targets = _targets(train_ids)
    targets["spurious_dangerous"] = False
    route = RFFKernelRoute(n_components=8, random_state=3).fit(
        _features(train_ids), targets
    )

    prediction = route.predict(_features(["validation"], offset=0.2)).scalars

    assert prediction.loc[0, "spurious_dangerous"] == 0.0


def test_prediction_does_not_refit_train_only_preprocessing() -> None:
    train_ids = [f"train-{index}" for index in range(8)]
    route = RFFKernelRoute(n_components=10, random_state=5).fit(
        _features(train_ids), _targets(train_ids)
    )
    imputer_statistics = route.imputer_.statistics_.copy()
    scaler_mean = route.scaler_.mean_.copy()
    random_weights = route.rff_.random_weights_.copy()
    validation = _features(["validation"], offset=1.0e12)

    route.predict(validation)

    np.testing.assert_array_equal(route.imputer_.statistics_, imputer_statistics)
    np.testing.assert_array_equal(route.scaler_.mean_, scaler_mean)
    np.testing.assert_array_equal(route.rff_.random_weights_, random_weights)


def test_invalid_targets_do_not_log_before_validity_filtering() -> None:
    ids = [f"train-{index}" for index in range(8)]
    targets = _targets(ids)
    targets.loc[0, "fp_hz"] = targets.loc[0, "fs_hz"] - 1
    targets.loc[0, "target_valid_fp"] = False
    targets.loc[1, "qs"] = -1
    targets.loc[1, "target_valid_qs"] = False
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        RFFKernelRoute(n_components=8).fit(_features(ids), targets)


def test_ids_must_be_canonical_strings() -> None:
    features = _features(["ok", "also-ok"])
    features.loc[0, "Training_Row_ID"] = " ok "
    with pytest.raises(ValueError, match="canonical|blank"):
        RFFKernelRoute(n_components=4).fit(features, _targets(["ok", "also-ok"]))
