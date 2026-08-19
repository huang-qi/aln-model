from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


SCALAR_TARGETS = (
    "fs_hz",
    "fp_hz",
    "k_eff2",
    "qs",
    "qp",
    "spurious_dangerous",
)
TARGET_VALIDITY_COLUMNS = {
    "fs_hz": "target_valid_fs",
    "fp_hz": "target_valid_fp",
    "k_eff2": "target_valid_k_eff2",
    "qs": "target_valid_qs",
    "qp": "target_valid_qp",
    "spurious_dangerous": "target_valid_spurious",
}
COUPLING_ABSOLUTE_TOLERANCE = 1e-8
COUPLING_RELATIVE_TOLERANCE = 1e-6


def _canonical_ids(frame: pd.DataFrame, name: str) -> pd.Series:
    if "Training_Row_ID" not in frame:
        raise ValueError(f"{name} missing Training_Row_ID")
    raw_ids = frame["Training_Row_ID"]
    if raw_ids.isna().any() or raw_ids.astype(str).str.strip().eq("").any():
        raise ValueError(f"{name} contains blank Training_Row_ID")
    ids = raw_ids.astype(str).str.strip()
    if ids.duplicated().any():
        raise ValueError(f"{name} contains duplicate Training_Row_ID")
    return ids


def _align_frame_to_ids(
    frame: pd.DataFrame, expected_ids: pd.Series, name: str
) -> tuple[pd.DataFrame, np.ndarray]:
    ids = _canonical_ids(frame, name)
    if set(ids) != set(expected_ids):
        raise ValueError(f"{name} Training_Row_ID set does not match truth")
    positions = pd.Series(np.arange(len(ids)), index=ids.to_numpy())
    order = positions.loc[expected_ids.tolist()].to_numpy(dtype=int)
    return frame.iloc[order].reset_index(drop=True), order


def _require_targets(frame: pd.DataFrame, name: str) -> None:
    missing = set(SCALAR_TARGETS).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} missing scalar prediction columns: {sorted(missing)}")


def _valid_mask(truth: pd.DataFrame, target: str) -> np.ndarray:
    values = pd.to_numeric(truth[target], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(values)
    validity_column = TARGET_VALIDITY_COLUMNS[target]
    if validity_column in truth:
        mask &= truth[validity_column].fillna(False).astype(bool).to_numpy()
    return mask


def _absolute_error(
    truth: pd.DataFrame, prediction: pd.DataFrame, target: str
) -> np.ndarray:
    mask = _valid_mask(truth, target)
    actual = pd.to_numeric(truth.loc[mask, target], errors="coerce").to_numpy(float)
    predicted = pd.to_numeric(
        prediction.loc[mask, target], errors="coerce"
    ).to_numpy(float)
    if actual.size == 0:
        return np.array([], dtype=float)
    if not np.all(np.isfinite(predicted)):
        return np.array([np.inf])
    return np.abs(predicted - actual)


def _mean_or_nan(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def evaluate_predictions(
    truth: pd.DataFrame,
    prediction: pd.DataFrame,
    *,
    true_spectrum: np.ndarray | None = None,
    predicted_spectrum: np.ndarray | None = None,
    slice_features: pd.DataFrame | None = None,
    area_res_tertile_edges: tuple[float, float] | None = None,
) -> dict[str, object]:
    """Compute the frozen scalar metrics and optional complex-spectrum RMSE."""

    _require_targets(truth, "truth")
    _require_targets(prediction, "prediction")
    if len(truth) != len(prediction):
        raise ValueError("truth and prediction row counts differ")
    truth_ids = _canonical_ids(truth, "truth")
    truth = truth.assign(Training_Row_ID=truth_ids).reset_index(drop=True)
    prediction, prediction_order = _align_frame_to_ids(
        prediction, truth_ids, "prediction"
    )
    if predicted_spectrum is not None:
        predicted_spectrum = np.asarray(predicted_spectrum)[prediction_order]

    fs_error = _absolute_error(truth, prediction, "fs_hz")
    fp_error = _absolute_error(truth, prediction, "fp_hz")
    coupling_error = _absolute_error(truth, prediction, "k_eff2")
    metrics = {
        "fs_mae_mhz": _mean_or_nan(fs_error) / 1e6,
        "fp_mae_mhz": _mean_or_nan(fp_error) / 1e6,
        "k_eff2_mae_percentage_points": _mean_or_nan(coupling_error) * 100,
    }

    fs_frequency_mask = _valid_mask(truth, "fs_hz")
    fp_frequency_mask = _valid_mask(truth, "fp_hz")
    row_fs_error = np.abs(
        pd.to_numeric(prediction["fs_hz"], errors="coerce").to_numpy(float)
        - pd.to_numeric(truth["fs_hz"], errors="coerce").to_numpy(float)
    )
    row_fp_error = np.abs(
        pd.to_numeric(prediction["fp_hz"], errors="coerce").to_numpy(float)
        - pd.to_numeric(truth["fp_hz"], errors="coerce").to_numpy(float)
    )
    row_fs_error[fs_frequency_mask & ~np.isfinite(row_fs_error)] = np.inf
    row_fp_error[fp_frequency_mask & ~np.isfinite(row_fp_error)] = np.inf
    valid_frequency_count = fs_frequency_mask.astype(int) + fp_frequency_mask.astype(int)
    frequency_error_sum = np.where(fs_frequency_mask, row_fs_error, 0.0) + np.where(
        fp_frequency_mask, row_fp_error, 0.0
    )
    eligible_frequency_error = (
        frequency_error_sum[valid_frequency_count > 0]
        / valid_frequency_count[valid_frequency_count > 0]
        / 1e6
    )
    if eligible_frequency_error.size:
        worst_count = max(1, int(np.ceil(0.05 * eligible_frequency_error.size)))
        metrics["worst5_frequency_error_mhz"] = float(
            np.sort(eligible_frequency_error)[-worst_count:].mean()
        )
    else:
        metrics["worst5_frequency_error_mhz"] = float("nan")

    for target in ("qs", "qp"):
        mask = _valid_mask(truth, target)
        actual = pd.to_numeric(truth.loc[mask, target], errors="coerce").to_numpy(float)
        predicted = pd.to_numeric(
            prediction.loc[mask, target], errors="coerce"
        ).to_numpy(float)
        if actual.size == 0:
            value = float("nan")
        elif (
            np.any(actual <= 0)
            or np.any(predicted <= 0)
            or not np.all(np.isfinite(predicted))
        ):
            value = float("inf")
        else:
            value = float(np.mean(np.abs(np.log(predicted) - np.log(actual))))
        metrics[f"{target}_log_mae"] = value

    spurious_mask = _valid_mask(truth, "spurious_dangerous")
    spurious_truth = truth.loc[spurious_mask, "spurious_dangerous"].astype(bool).to_numpy()
    spurious_probability = pd.to_numeric(
        prediction.loc[spurious_mask, "spurious_dangerous"], errors="coerce"
    ).to_numpy(float)
    if not np.all(np.isfinite(spurious_probability)) or np.any(
        (spurious_probability < 0) | (spurious_probability > 1)
    ):
        raise ValueError("spurious predictions must be finite probabilities in [0, 1]")
    if spurious_truth.size:
        metrics["spurious_brier"] = float(
            np.mean(np.square(spurious_probability - spurious_truth.astype(float)))
        )
        metrics["spurious_aucpr"] = (
            float(average_precision_score(spurious_truth, spurious_probability))
            if np.unique(spurious_truth).size == 2
            else float("nan")
        )
    else:
        metrics["spurious_brier"] = float("nan")
        metrics["spurious_aucpr"] = float("nan")

    numeric_prediction = prediction.loc[
        :, ["fs_hz", "fp_hz", "k_eff2", "qs", "qp"]
    ].apply(pd.to_numeric, errors="coerce")
    fs_hz = numeric_prediction["fs_hz"].to_numpy(float)
    fp_hz = numeric_prediction["fp_hz"].to_numpy(float)
    k_eff2 = numeric_prediction["k_eff2"].to_numpy(float)
    qs = numeric_prediction["qs"].to_numpy(float)
    qp = numeric_prediction["qp"].to_numpy(float)
    finite = np.isfinite(numeric_prediction.to_numpy(float)).all(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        coupling_from_frequencies = 1 - np.square(fs_hz / fp_hz)
    coupling_consistent = np.isclose(
        k_eff2,
        coupling_from_frequencies,
        rtol=COUPLING_RELATIVE_TOLERANCE,
        atol=COUPLING_ABSOLUTE_TOLERANCE,
    )
    physical = (
        finite
        & (fs_hz > 0)
        & (fp_hz > fs_hz)
        & (k_eff2 >= 0)
        & (k_eff2 <= 1)
        & coupling_consistent
        & (qs > 0)
        & (qp > 0)
    )
    metrics["physical_violation_rate"] = float(np.mean(~physical))

    if (true_spectrum is None) != (predicted_spectrum is None):
        raise ValueError("both true and predicted spectra are required for spectral RMSE")
    if true_spectrum is not None and predicted_spectrum is not None:
        actual_spectrum = np.asarray(true_spectrum)
        route_spectrum = np.asarray(predicted_spectrum)
        if actual_spectrum.shape != route_spectrum.shape:
            raise ValueError("true and predicted spectrum shapes differ")
        difference = route_spectrum - actual_spectrum
        metrics["spectrum_complex_rmse"] = (
            float(np.sqrt(np.mean(np.square(np.abs(difference)))))
            if np.isfinite(difference.real).all() and np.isfinite(difference.imag).all()
            else float("inf")
        )
    if slice_features is not None:
        aligned_features, _ = _align_frame_to_ids(
            slice_features, truth_ids, "slice features"
        )
        slices: dict[str, dict[str, dict[str, float | int]]] = {}

        def add_slice_group(group_name: str, values: pd.Series, categories: list[str]) -> None:
            group: dict[str, dict[str, float | int]] = {}
            normalized_values = values.astype(str)
            for category in categories:
                slice_mask = normalized_values.eq(category).to_numpy()
                fs_slice_error = row_fs_error[
                    slice_mask & _valid_mask(truth, "fs_hz")
                ]
                fp_slice_error = row_fp_error[
                    slice_mask & _valid_mask(truth, "fp_hz")
                ]
                group[category] = {
                    "count": int(slice_mask.sum()),
                    "fs_mae_mhz": _mean_or_nan(fs_slice_error) / 1e6,
                    "fp_mae_mhz": _mean_or_nan(fp_slice_error) / 1e6,
                }
            slices[group_name] = group

        for column in ("Template", "EG_State"):
            if column not in aligned_features:
                raise ValueError(f"slice features missing {column}")
            values = aligned_features[column].fillna("<MISSING>").astype(str)
            add_slice_group(column, values, sorted(values.unique().tolist()))
        if "Train_A_res_um2" not in aligned_features:
            raise ValueError("slice features missing Train_A_res_um2")
        if area_res_tertile_edges is None:
            raise ValueError("area_res_tertile_edges are required for slice metrics")
        area = pd.to_numeric(
            aligned_features["Train_A_res_um2"], errors="coerce"
        ).to_numpy(float)
        if not np.isfinite(area).all():
            raise ValueError("Train_A_res_um2 must be finite for slice metrics")
        lower, upper = area_res_tertile_edges
        area_bins = pd.Series(
            np.where(area <= lower, "low", np.where(area <= upper, "mid", "high"))
        )
        add_slice_group("A_res_tertile", area_bins, ["low", "mid", "high"])
        metrics["slices"] = slices
    return metrics
