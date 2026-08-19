from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .features import STATE_COLUMNS
from .metrics import SCALAR_TARGETS, TARGET_VALIDITY_COLUMNS, evaluate_predictions
from .routes.base import Route, RoutePrediction


@dataclass(frozen=True)
class PreparedDataset:
    labels: pd.DataFrame
    folds: pd.DataFrame
    features: pd.DataFrame
    targets: pd.DataFrame
    spectra: np.ndarray
    spectra_ids: np.ndarray
    frequency_hz: np.ndarray
    area_res_tertile_edges: tuple[float, float]


@dataclass(frozen=True)
class OOFResult:
    predictions: pd.DataFrame
    spectrum: np.ndarray | None
    metrics: dict[str, object]


def _canonical_ids(frame: pd.DataFrame, artifact: str) -> pd.Series:
    if "Training_Row_ID" not in frame:
        raise ValueError(f"{artifact} missing Training_Row_ID")
    ids = frame["Training_Row_ID"]
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any():
        raise ValueError(f"{artifact} contains blank Training_Row_ID")
    ids = ids.astype(str).str.strip()
    if ids.duplicated().any():
        raise ValueError(f"{artifact} contains duplicate Training_Row_ID")
    return ids


def _align_frame(
    frame: pd.DataFrame, ordered_ids: pd.Series, artifact: str
) -> pd.DataFrame:
    ids = _canonical_ids(frame, artifact)
    expected = set(ordered_ids)
    actual = set(ids)
    if actual != expected:
        raise ValueError(f"{artifact} Training_Row_ID set does not match labels")
    aligned = frame.assign(Training_Row_ID=ids).set_index("Training_Row_ID")
    return aligned.loc[ordered_ids.tolist()].reset_index()


def load_prepared_dataset(directory: str | Path) -> PreparedDataset:
    """Read and ID-align the immutable prepared artifact contract."""

    directory = Path(directory)
    labels = pd.read_csv(directory / "labels.csv")
    label_ids = _canonical_ids(labels, "labels")
    labels = labels.assign(Training_Row_ID=label_ids).reset_index(drop=True)
    missing_targets = set(SCALAR_TARGETS).difference(labels.columns)
    if missing_targets:
        raise ValueError(f"labels missing scalar targets: {sorted(missing_targets)}")
    missing_validity = set(TARGET_VALIDITY_COLUMNS.values()).difference(labels.columns)
    if missing_validity:
        raise ValueError(f"labels missing target validity columns: {sorted(missing_validity)}")

    folds = _align_frame(pd.read_csv(directory / "folds.csv"), label_ids, "folds")
    if "fold" not in folds:
        raise ValueError("folds missing fold")
    if folds["fold"].isna().any():
        raise ValueError("folds contains missing fold assignments")

    with np.load(directory / "spectra.npz", allow_pickle=False) as archive:
        if not {"s11", "training_row_id"}.issubset(archive.files):
            raise ValueError("spectra.npz must contain s11 and training_row_id")
        raw_spectra = np.asarray(archive["s11"])
        spectrum_ids = pd.Series(archive["training_row_id"].astype(str))
    if not np.iscomplexobj(raw_spectra):
        raise ValueError("spectra s11 must have complex dtype")
    if not (
        np.isfinite(raw_spectra.real).all()
        and np.isfinite(raw_spectra.imag).all()
    ):
        raise ValueError("spectra s11 must be finite")
    spectrum_frame = pd.DataFrame({"Training_Row_ID": spectrum_ids})
    canonical_spectrum_ids = _canonical_ids(spectrum_frame, "spectra")
    if set(canonical_spectrum_ids) != set(label_ids):
        raise ValueError("spectra Training_Row_ID set does not match labels")
    if raw_spectra.ndim != 2 or raw_spectra.shape[0] != len(spectrum_ids):
        raise ValueError("spectra s11 must be a row-aligned two-dimensional matrix")
    spectrum_positions = pd.Series(
        np.arange(len(spectrum_ids)), index=canonical_spectrum_ids.to_numpy()
    )
    spectra = raw_spectra[spectrum_positions.loc[label_ids.tolist()].to_numpy(int)]
    frequency_hz = np.asarray(
        np.load(directory / "frequency.npy", allow_pickle=False), dtype=float
    )
    if frequency_hz.ndim != 1:
        raise ValueError("frequency grid must be one-dimensional")
    if frequency_hz.size == 0:
        raise ValueError("frequency grid must be non-empty")
    if not np.isfinite(frequency_hz).all():
        raise ValueError("frequency grid must be finite")
    if not np.all(np.diff(frequency_hz) > 0):
        raise ValueError("frequency grid must be strictly increasing")
    if spectra.shape[1] != len(frequency_hz):
        raise ValueError("spectrum width does not match the shared frequency grid")

    input_columns = ["Training_Row_ID"] + [
        column
        for column in labels.columns
        if column.startswith("Train_") or column in STATE_COLUMNS
    ]
    validity_columns = list(dict.fromkeys(TARGET_VALIDITY_COLUMNS.values()))
    targets = labels.loc[
        :, ["Training_Row_ID", *SCALAR_TARGETS, *validity_columns]
    ].copy()
    if "Train_A_res_um2" not in labels:
        raise ValueError("labels missing Train_A_res_um2 for fixed area slices")
    area = pd.to_numeric(labels["Train_A_res_um2"], errors="coerce").to_numpy(float)
    if not np.isfinite(area).all():
        raise ValueError("Train_A_res_um2 must be finite for fixed area slices")
    area_edges = tuple(float(value) for value in np.quantile(area, [1 / 3, 2 / 3]))
    return PreparedDataset(
        labels=labels,
        folds=folds,
        features=labels.loc[:, input_columns].copy(),
        targets=targets,
        spectra=spectra,
        spectra_ids=label_ids.to_numpy(copy=True),
        frequency_hz=frequency_hz,
        area_res_tertile_edges=area_edges,
    )


def _validate_route_prediction(
    prediction: RoutePrediction,
    validation_features: pd.DataFrame,
    expected_spectrum_width: int,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    if not isinstance(prediction, RoutePrediction):
        raise TypeError("route predict() must return RoutePrediction")
    scalars = prediction.scalars.reset_index(drop=True).copy()
    missing = set(SCALAR_TARGETS).difference(scalars.columns)
    if missing:
        raise ValueError(f"missing scalar prediction columns: {sorted(missing)}")
    if len(scalars) != len(validation_features):
        raise ValueError("route prediction row count does not match validation features")
    if "Training_Row_ID" not in scalars:
        raise ValueError("route prediction missing Training_Row_ID")
    expected_ids = validation_features["Training_Row_ID"].astype(str).str.strip()
    raw_ids = scalars["Training_Row_ID"]
    if raw_ids.isna().any() or raw_ids.astype(str).str.strip().eq("").any():
        raise ValueError("route prediction contains blank Training_Row_ID")
    actual_ids = raw_ids.astype(str).str.strip()
    if actual_ids.duplicated().any():
        raise ValueError("route prediction contains duplicate Training_Row_ID")
    if set(actual_ids) != set(expected_ids):
        raise ValueError("route prediction Training_Row_ID set does not match validation")
    positions = pd.Series(np.arange(len(actual_ids)), index=actual_ids.to_numpy())
    order = positions.loc[expected_ids.tolist()].to_numpy(dtype=int)
    scalars = scalars.iloc[order].reset_index(drop=True)
    scalars["Training_Row_ID"] = expected_ids.to_numpy()
    spectrum = None if prediction.spectrum is None else np.asarray(prediction.spectrum)
    if spectrum is not None and spectrum.shape != (
        len(validation_features),
        expected_spectrum_width,
    ):
        raise ValueError("route spectrum shape does not match validation rows/grid")
    if spectrum is not None:
        spectrum = spectrum[order]
    return scalars, spectrum


def _strict_dataset_ids(frame: pd.DataFrame, component: str) -> list[str]:
    if "Training_Row_ID" not in frame:
        raise ValueError(f"{component} missing Training_Row_ID")
    raw_ids = frame["Training_Row_ID"]
    if raw_ids.isna().any():
        raise ValueError(f"{component} contains blank Training_Row_ID")
    if not raw_ids.map(lambda value: isinstance(value, str)).all():
        raise ValueError(f"{component} Training_Row_ID values must be canonical strings")
    canonical_ids = raw_ids.str.strip()
    if canonical_ids.eq("").any():
        raise ValueError(f"{component} contains blank Training_Row_ID")
    if not canonical_ids.equals(raw_ids):
        raise ValueError(f"{component} Training_Row_ID values are not canonical")
    if canonical_ids.duplicated().any():
        raise ValueError(f"{component} contains duplicate Training_Row_ID")
    return canonical_ids.tolist()


def run_oof(
    route_factory: Callable[[], Route],
    dataset: PreparedDataset,
    *,
    permute_training_targets: bool = False,
    random_state: int = 0,
) -> OOFResult:
    """Fit a fresh route per fold and return row-aligned out-of-fold predictions."""

    n_rows = len(dataset.labels)
    if not (
        len(dataset.folds)
        == len(dataset.features)
        == len(dataset.targets)
        == dataset.spectra.shape[0]
        == n_rows
    ):
        raise ValueError("prepared dataset components have inconsistent row counts")
    reference_ids = _strict_dataset_ids(dataset.labels, "labels")
    for name, component in (
        ("features", dataset.features),
        ("targets", dataset.targets),
        ("folds", dataset.folds),
    ):
        component_ids = _strict_dataset_ids(component, name)
        if component_ids != reference_ids:
            raise ValueError(f"{name} Training_Row_ID order does not match labels")
    spectra_ids = _strict_dataset_ids(
        pd.DataFrame({"Training_Row_ID": dataset.spectra_ids}), "spectra_ids"
    )
    if spectra_ids != reference_ids:
        raise ValueError("spectra_ids Training_Row_ID order does not match labels")
    rng = np.random.default_rng(random_state)
    scalar_output = pd.DataFrame(index=np.arange(n_rows), columns=SCALAR_TARGETS)
    predicted_spectrum: np.ndarray | None = None
    spectral_contract: bool | None = None

    for fold_value in sorted(dataset.folds["fold"].unique().tolist()):
        validation_mask = dataset.folds["fold"].eq(fold_value).to_numpy()
        train_positions = np.flatnonzero(~validation_mask)
        validation_positions = np.flatnonzero(validation_mask)
        if not len(train_positions) or not len(validation_positions):
            raise ValueError(f"fold {fold_value!r} has an empty train or validation split")
        train_features = dataset.features.iloc[train_positions].reset_index(drop=True)
        train_targets = dataset.targets.iloc[train_positions].reset_index(drop=True)
        train_spectra = dataset.spectra[train_positions]
        if permute_training_targets:
            permutation = rng.permutation(len(train_targets))
            target_value_columns = [
                column
                for column in train_targets.columns
                if column != "Training_Row_ID"
            ]
            train_targets.loc[:, target_value_columns] = (
                train_targets.iloc[permutation]
                .loc[:, target_value_columns]
                .to_numpy()
            )
            train_spectra = train_spectra[permutation]
        route = route_factory()
        if not isinstance(route, Route):
            raise TypeError("route_factory must return an object with fit() and predict()")
        route.fit(
            train_features,
            train_targets,
            spectra=train_spectra,
            frequency_hz=dataset.frequency_hz.copy(),
        )
        validation_features = dataset.features.iloc[validation_positions].reset_index(
            drop=True
        )
        fold_scalars, fold_spectrum = _validate_route_prediction(
            route.predict(validation_features),
            validation_features,
            len(dataset.frequency_hz),
        )
        has_spectrum = fold_spectrum is not None
        if spectral_contract is None:
            spectral_contract = has_spectrum
            if has_spectrum:
                predicted_spectrum = np.empty_like(dataset.spectra)
        elif has_spectrum != spectral_contract:
            raise ValueError("route returned spectra for only some folds")
        scalar_output.loc[validation_positions, list(SCALAR_TARGETS)] = fold_scalars.loc[
            :, SCALAR_TARGETS
        ].to_numpy()
        if fold_spectrum is not None and predicted_spectrum is not None:
            predicted_spectrum[validation_positions] = fold_spectrum

    predictions = pd.DataFrame(
        {
            "Training_Row_ID": dataset.labels["Training_Row_ID"].astype(str).to_numpy(),
            "fold": dataset.folds["fold"].to_numpy(),
        }
    )
    for target in SCALAR_TARGETS:
        predictions[target] = pd.to_numeric(scalar_output[target], errors="coerce")
    metrics = evaluate_predictions(
        dataset.targets,
        predictions,
        true_spectrum=dataset.spectra if predicted_spectrum is not None else None,
        predicted_spectrum=predicted_spectrum,
        slice_features=dataset.features,
        area_res_tertile_edges=dataset.area_res_tertile_edges,
    )
    return OOFResult(
        predictions=predictions,
        spectrum=predicted_spectrum,
        metrics=metrics,
    )


def run_permutation_control(
    route_factory: Callable[[], Route],
    dataset: PreparedDataset,
    *,
    random_state: int = 0,
) -> OOFResult:
    """Run OOF evaluation with targets permuted independently inside each train fold."""

    return run_oof(
        route_factory,
        dataset,
        permute_training_targets=True,
        random_state=random_state,
    )
