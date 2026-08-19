from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from aln_model.routes.physics_boost import PhysicsBoostRoute


def _features(n_rows: int) -> pd.DataFrame:
    x = np.linspace(0.0, 1.0, n_rows)
    return pd.DataFrame(
        {
            "Training_Row_ID": [f"TR-{index:04d}" for index in range(n_rows)],
            "Train_A_res_um2": 700.0 + 4_800.0 * x,
            "Train_L_top_um": 30.0 + 50.0 * x,
            "Train_L_bot_um": 46.0 + 50.0 * x,
            "Train_L_air_um": 40.0 + 50.0 * x,
            "Train_h_FL_nm": np.where(x > 0.5, 60.0, 0.0),
            "Train_L_FL_um": np.where(x > 0.5, 2.0, 0.0),
            "Train_h_AG_nm": 80.0,
            "Train_L_AG_um": 1.0 + x,
            "Train_h_EG_nm": 3.0,
            "Train_L_EG_um": x,
            "Train_PF_nm": np.where(x > 0.75, 100.0, np.nan),
            "Template": np.where(x > 0.5, "T11", "T01"),
            "EG_State": np.where(x > 0.5, "patterned", "blanket"),
            "has_FL": (x > 0.5).astype(int),
            "has_AG": 1,
            "has_patterned_EG": (x > 0.5).astype(int),
            "EG_material_present": 1,
            "has_PF": (x > 0.75).astype(int),
            "is_BASIC": 0,
        }
    )


def _targets(features: pd.DataFrame) -> pd.DataFrame:
    x = (features["Train_A_res_um2"].to_numpy() - 700.0) / 4_800.0
    center = 5.0e9 + 2.0e8 * x
    bandwidth = 2.5e8 + 5.0e7 * x
    fs = center - bandwidth / 2
    fp = center + bandwidth / 2
    result = pd.DataFrame(
        {
            "Training_Row_ID": features["Training_Row_ID"],
            "fs_hz": fs,
            "fp_hz": fp,
            "k_eff2": 1.0 - np.square(fs / fp),
            "qs": np.exp(5.5 + x),
            "qp": np.exp(5.0 + 0.5 * x),
            "spurious_dangerous": x > 0.6,
        }
    )
    for suffix in ("fs", "fp", "k_eff2", "qs", "qp", "spurious"):
        result[f"target_valid_{suffix}"] = True
    return result


def test_fit_predict_returns_id_aligned_physical_probabilistic_predictions():
    features = _features(80)
    route = PhysicsBoostRoute(random_state=11, max_iter=30).fit(
        features.iloc[:64], _targets(features.iloc[:64])
    )

    prediction = route.predict(features.iloc[64:])

    assert prediction.spectrum is None
    assert prediction.scalars["Training_Row_ID"].tolist() == features.iloc[64:][
        "Training_Row_ID"
    ].tolist()
    assert set(prediction.scalars) == {
        "Training_Row_ID",
        "fs_hz",
        "fp_hz",
        "k_eff2",
        "qs",
        "qp",
        "spurious_dangerous",
    }
    assert np.isfinite(prediction.scalars.drop(columns="Training_Row_ID")).all().all()
    assert (prediction.scalars["fs_hz"] > 0).all()
    assert (prediction.scalars["fp_hz"] > prediction.scalars["fs_hz"]).all()
    np.testing.assert_allclose(
        prediction.scalars["k_eff2"],
        1.0
        - np.square(
            prediction.scalars["fs_hz"] / prediction.scalars["fp_hz"]
        ),
    )
    assert (prediction.scalars[["qs", "qp"]] > 0).all().all()
    assert prediction.scalars["spurious_dangerous"].between(0, 1).all()


def test_fit_aligns_shuffled_targets_by_strict_training_row_id():
    features = _features(80)
    targets = _targets(features)
    validation = _features(12).assign(
        Training_Row_ID=[f"VAL-{index:04d}" for index in range(12)]
    )
    ordered = PhysicsBoostRoute(random_state=5, max_iter=20).fit(features, targets)
    shuffled = PhysicsBoostRoute(random_state=5, max_iter=20).fit(
        features, targets.sample(frac=1.0, random_state=19).reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(
        ordered.predict(validation).scalars,
        shuffled.predict(validation).scalars,
    )


def test_fit_uses_target_validity_masks_and_supports_one_class_spurious_labels():
    features = _features(48)
    targets = _targets(features)
    targets.loc[:5, ["fs_hz", "fp_hz", "k_eff2"]] = np.nan
    targets.loc[:5, ["target_valid_fs", "target_valid_fp", "target_valid_k_eff2"]] = (
        False
    )
    targets.loc[6:10, "qs"] = np.nan
    targets.loc[6:10, "target_valid_qs"] = False
    targets.loc[11:15, "qp"] = np.nan
    targets.loc[11:15, "target_valid_qp"] = False
    targets["spurious_dangerous"] = False

    prediction = PhysicsBoostRoute(random_state=3, max_iter=10).fit(
        features, targets
    ).predict(features.iloc[:8])

    assert np.isfinite(prediction.scalars.drop(columns="Training_Row_ID")).all().all()
    assert prediction.scalars["spurious_dangerous"].eq(0.0).all()


def test_invalid_nonphysical_frequency_labels_do_not_emit_runtime_warnings():
    features = _features(40)
    targets = _targets(features)
    targets.loc[0, "fp_hz"] = targets.loc[0, "fs_hz"] - 1.0
    targets.loc[0, ["target_valid_fp", "target_valid_k_eff2"]] = False

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        PhysicsBoostRoute(random_state=2, max_iter=5).fit(features, targets)


def test_spurious_labels_must_be_binary():
    features = _features(30)
    targets = _targets(features)
    targets["spurious_dangerous"] = targets["spurious_dangerous"].astype(int)
    targets.loc[0, "spurious_dangerous"] = 2
    with pytest.raises(ValueError, match="binary"):
        PhysicsBoostRoute(max_iter=5).fit(features, targets)
