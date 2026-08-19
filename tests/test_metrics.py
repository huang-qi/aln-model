import numpy as np
import pandas as pd
import pytest

from aln_model.metrics import evaluate_predictions


def _truth() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "Training_Row_ID": ["a", "b", "c", "d"],
            "fs_hz": [5.000e9, 5.100e9, 5.200e9, 5.300e9],
            "fp_hz": [5.200e9, 5.300e9, 5.400e9, 5.500e9],
            "qs": [100.0, 200.0, 400.0, 800.0],
            "qp": [120.0, 240.0, 480.0, 960.0],
            "spurious_dangerous": [False, True, True, False],
        }
    )
    frame["k_eff2"] = 1 - (frame["fs_hz"] / frame["fp_hz"]) ** 2
    return frame


def test_scalar_metric_units_and_classification_scores():
    truth = _truth()
    prediction = truth.copy()
    prediction["fs_hz"] += np.array([2, -4, 6, -8]) * 1e6
    prediction["fp_hz"] += np.array([1, -3, 5, -7]) * 1e6
    prediction["k_eff2"] = 1 - (prediction["fs_hz"] / prediction["fp_hz"]) ** 2
    coupling_error = np.abs(prediction["k_eff2"] - truth["k_eff2"]).mean() * 100
    prediction["qs"] *= np.array([np.e, 1 / np.e, np.e, 1 / np.e])
    prediction["qp"] *= np.exp(np.array([0.5, -0.5, 0.5, -0.5]))
    prediction["spurious_dangerous"] = [0.1, 0.8, 0.4, 0.3]

    metrics = evaluate_predictions(truth, prediction)

    assert metrics["fs_mae_mhz"] == pytest.approx(5.0)
    assert metrics["fp_mae_mhz"] == pytest.approx(4.0)
    assert metrics["k_eff2_mae_percentage_points"] == pytest.approx(coupling_error)
    assert metrics["qs_log_mae"] == pytest.approx(1.0)
    assert metrics["qp_log_mae"] == pytest.approx(0.5)
    assert metrics["spurious_aucpr"] == pytest.approx(1.0)
    assert metrics["spurious_brier"] == pytest.approx(0.125)
    assert metrics["physical_violation_rate"] == pytest.approx(0.0)


def test_target_validity_masks_are_applied_target_by_target():
    truth = _truth().iloc[:2].copy()
    prediction = truth.copy()
    prediction.loc[1, "fs_hz"] += 1e9
    prediction.loc[1, "k_eff2"] += 0.5
    truth["target_valid_fs"] = [True, False]
    truth["target_valid_k_eff2"] = [True, False]

    metrics = evaluate_predictions(truth, prediction)

    assert metrics["fs_mae_mhz"] == pytest.approx(0.0)
    assert metrics["k_eff2_mae_percentage_points"] == pytest.approx(0.0)


def test_physical_violations_include_order_q_coupling_and_nonfinite_outputs():
    truth = _truth()
    prediction = truth.copy()
    prediction.loc[0, "fp_hz"] = prediction.loc[0, "fs_hz"]
    prediction.loc[1, "qs"] = 0.0
    prediction.loc[2, "k_eff2"] = 1.1
    prediction.loc[3, "fs_hz"] = np.nan

    metrics = evaluate_predictions(truth, prediction)

    assert metrics["physical_violation_rate"] == pytest.approx(1.0)


def test_physical_violation_checks_coupling_identity_with_tolerance():
    truth = _truth().iloc[:2].copy()
    prediction = truth.copy()
    prediction.loc[prediction.index[0], "k_eff2"] += 1e-10
    prediction.loc[prediction.index[1], "k_eff2"] += 1e-4

    metrics = evaluate_predictions(truth, prediction)

    assert metrics["physical_violation_rate"] == pytest.approx(0.5)


def test_optional_spectral_metric_is_complex_rmse():
    truth = _truth().iloc[:2]
    prediction = truth.copy()
    truth_spectrum = np.zeros((2, 3), dtype=complex)
    predicted_spectrum = np.full((2, 3), 3 + 4j, dtype=complex)

    metrics = evaluate_predictions(
        truth,
        prediction,
        true_spectrum=truth_spectrum,
        predicted_spectrum=predicted_spectrum,
    )

    assert metrics["spectrum_complex_rmse"] == pytest.approx(5.0)


def test_metrics_align_predictions_and_spectra_by_training_row_id():
    truth = _truth()
    prediction = truth.iloc[::-1].reset_index(drop=True)
    true_spectrum = np.arange(4, dtype=float)[:, None].astype(complex)
    predicted_spectrum = true_spectrum[::-1]

    metrics = evaluate_predictions(
        truth,
        prediction,
        true_spectrum=true_spectrum,
        predicted_spectrum=predicted_spectrum,
    )

    assert metrics["fs_mae_mhz"] == pytest.approx(0.0)
    assert metrics["spectrum_complex_rmse"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.drop(columns="Training_Row_ID"), "Training_Row_ID"),
        (
            lambda frame: frame.assign(Training_Row_ID=["a", "a", "c", "d"]),
            "duplicate",
        ),
        (
            lambda frame: frame.assign(Training_Row_ID=["a", "b", "c", "extra"]),
            "set does not match",
        ),
    ],
)
def test_metrics_reject_missing_duplicate_or_mismatched_ids(mutate, message):
    truth = _truth()

    with pytest.raises(ValueError, match=message):
        evaluate_predictions(truth, mutate(truth.copy()))


def test_worst_five_percent_frequency_error_uses_row_mean():
    count = 20
    fs = 5e9 + np.arange(count) * 1e6
    fp = fs + 200e6
    truth = pd.DataFrame(
        {
            "Training_Row_ID": [f"row-{i}" for i in range(count)],
            "fs_hz": fs,
            "fp_hz": fp,
            "k_eff2": 1 - (fs / fp) ** 2,
            "qs": 100.0,
            "qp": 200.0,
            "spurious_dangerous": False,
        }
    )
    prediction = truth.copy()
    errors_mhz = np.ones(count)
    errors_mhz[-1] = 20
    prediction["fs_hz"] += errors_mhz * 1e6
    prediction["fp_hz"] += errors_mhz * 1e6
    prediction["k_eff2"] = 1 - (prediction["fs_hz"] / prediction["fp_hz"]) ** 2

    metrics = evaluate_predictions(truth, prediction)

    assert metrics["worst5_frequency_error_mhz"] == pytest.approx(20.0)


def test_worst_five_percent_uses_each_frequency_target_validity_independently():
    count = 20
    fs = 5e9 + np.arange(count) * 1e6
    fp = fs + 200e6
    truth = pd.DataFrame(
        {
            "Training_Row_ID": [f"row-{i}" for i in range(count)],
            "fs_hz": fs,
            "fp_hz": fp,
            "k_eff2": 1 - (fs / fp) ** 2,
            "qs": 100.0,
            "qp": 200.0,
            "spurious_dangerous": False,
            "target_valid_fs": True,
            "target_valid_fp": True,
        }
    )
    prediction = truth.copy()
    prediction["fs_hz"] += 1e6
    prediction["fp_hz"] += 1e6
    prediction.loc[18, "fs_hz"] += 29e6
    prediction.loc[18, "target_valid_fp"] = False
    truth.loc[18, "target_valid_fp"] = False
    prediction.loc[19, "fp_hz"] += 19e6
    prediction.loc[19, "target_valid_fs"] = False
    truth.loc[19, "target_valid_fs"] = False
    prediction.loc[17, ["target_valid_fs", "target_valid_fp"]] = False
    truth.loc[17, ["target_valid_fs", "target_valid_fp"]] = False
    prediction["k_eff2"] = 1 - (prediction["fs_hz"] / prediction["fp_hz"]) ** 2

    metrics = evaluate_predictions(truth, prediction)

    assert metrics["worst5_frequency_error_mhz"] == pytest.approx(30.0)


def test_valid_truth_with_nonfinite_frequency_prediction_scores_as_infinite_everywhere():
    truth = _truth()
    prediction = truth.copy()
    prediction.loc[0, "fs_hz"] = np.nan
    slice_features = pd.DataFrame(
        {
            "Training_Row_ID": truth["Training_Row_ID"],
            "Template": ["T11", "T11", "T10", "T10"],
            "EG_State": ["patterned", "blanket", "patterned", "blanket"],
            "Train_A_res_um2": [1.0, 2.0, 3.0, 4.0],
        }
    )

    metrics = evaluate_predictions(
        truth,
        prediction,
        slice_features=slice_features,
        area_res_tertile_edges=(2.0, 3.0),
    )

    assert np.isinf(metrics["fs_mae_mhz"])
    assert np.isinf(metrics["worst5_frequency_error_mhz"])
    assert np.isinf(metrics["slices"]["Template"]["T11"]["fs_mae_mhz"])
    assert np.isinf(metrics["slices"]["A_res_tertile"]["low"]["fs_mae_mhz"])


def test_spurious_predictions_must_be_probabilities():
    truth = _truth()
    prediction = truth.copy()
    prediction["spurious_dangerous"] = [0.0, 0.5, 1.0, 1.01]

    with pytest.raises(ValueError, match="probabilities"):
        evaluate_predictions(truth, prediction)
