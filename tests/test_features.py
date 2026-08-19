import numpy as np
import pandas as pd
import pytest

from aln_model.features import build_physics_features, geometry_fingerprint


def _geometry_rows() -> pd.DataFrame:
    base = {
        "Template": "T11",
        "EG_State": "patterned",
        "Train_A_res_um2": 2500.0,
        "Train_L_top_um": 60.0,
        "Train_L_bot_um": 56.0,
        "Train_L_air_um": 50.0,
        "Train_h_FL_nm": 70.0,
        "Train_L_FL_um": 1.5,
        "Train_h_AG_nm": 80.0,
        "Train_L_AG_um": 1.25,
        "Train_h_EG_nm": 3.0,
        "Train_L_EG_um": 0.75,
        "Train_PF_nm": 0.0,
        "has_FL": 1,
        "has_AG": 1,
        "has_patterned_EG": 1,
        "EG_material_present": 1,
        "has_PF": 0,
        "is_BASIC": 0,
    }
    first = dict(base, SNP_A_res_um2=9999.0, SNP_File="raw-a.s1p")
    second = dict(base, SNP_A_res_um2=2500.0, SNP_File="raw-b.s1p")
    third = dict(base, Train_L_AG_um=1.5, SNP_A_res_um2=2500.0)
    return pd.DataFrame([first, second, third])


def test_geometry_fingerprint_uses_authoritative_train_values_only():
    fingerprints = geometry_fingerprint(_geometry_rows())

    assert fingerprints.iloc[0] == fingerprints.iloc[1]
    assert fingerprints.iloc[0] != fingerprints.iloc[2]


def test_physics_features_encode_eg_state_and_ratios():
    rows = _geometry_rows().iloc[[0]].copy()

    features = build_physics_features(rows)

    assert features.loc[0, "eg_patterned"] == 1.0
    assert features.loc[0, "eg_blanket"] == 0.0
    assert features.loc[0, "air_to_bot_ratio"] == 50.0 / 56.0
    assert features.loc[0, "ag_to_top_ratio"] == 1.25 / 60.0
    assert "SNP_A_res_um2" not in features.columns
    assert features.loc[0, "topology_T11"] == 1.0
    assert features.loc[0, "has_AG"] == 1.0


def test_pf_nan_normalization_preserves_physical_equivalence_and_unknown_state():
    rows = _geometry_rows().iloc[[0, 1, 2]].copy()
    rows.loc[:, "Train_PF_nm"] = [np.nan, 0.0, np.nan]
    rows.loc[:, "has_PF"] = [0, 0, 1]

    features = build_physics_features(rows)
    fingerprints = geometry_fingerprint(rows)

    assert features["Train_PF_nm"].tolist() == [0.0, 0.0, 0.0]
    assert features["pf_value_missing"].tolist() == [0.0, 0.0, 1.0]
    assert fingerprints.iloc[0] == fingerprints.iloc[1]
    assert fingerprints.iloc[0] != fingerprints.iloc[2]


def test_geometry_fingerprint_accepts_integer_typed_physical_columns():
    rows = _geometry_rows().iloc[[0, 1]].copy()
    rows.loc[:, list(rows.filter(like="Train_").columns)] = rows.filter(
        like="Train_"
    ).fillna(0).astype("int64")
    for column in rows.filter(like="Train_").columns:
        rows[column] = rows[column].astype("int64")

    fingerprints = geometry_fingerprint(rows)

    assert fingerprints.iloc[0] == fingerprints.iloc[1]


def test_features_require_every_state_column():
    rows = _geometry_rows().drop(columns="has_AG")

    with pytest.raises(ValueError, match="state columns"):
        build_physics_features(rows)


def test_zero_ratio_denominators_emit_zero_and_invalid_flags():
    rows = _geometry_rows().iloc[[0]].copy()
    rows.loc[0, "Train_L_bot_um"] = 0.0
    rows.loc[0, "Train_L_top_um"] = 0.0

    features = build_physics_features(rows)

    ratio_columns = [
        "air_to_bot_ratio",
        "ag_to_top_ratio",
        "fl_to_top_ratio",
        "eg_to_top_ratio",
    ]
    assert features[ratio_columns].eq(0).all().all()
    assert features.filter(like="_ratio_invalid").eq(1).all().all()
    assert np.isfinite(features.select_dtypes(include=[np.number])).all().all()
