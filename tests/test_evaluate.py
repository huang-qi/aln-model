from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aln_model.evaluate import (
    load_prepared_dataset,
    run_oof,
    run_permutation_control,
)
from aln_model.routes.base import RoutePrediction, reconstruct_physical_targets


TARGETS = ("fs_hz", "fp_hz", "k_eff2", "qs", "qp", "spurious_dangerous")


def _write_prepared(directory: Path, n_rows: int = 12) -> pd.DataFrame:
    directory.mkdir()
    row_ids = [f"row-{index:02d}" for index in range(n_rows)]
    signal = np.arange(1, n_rows + 1, dtype=float)
    fs = 4.5e9 + signal * 10e6
    fp = fs + 200e6
    labels = pd.DataFrame(
        {
            "Training_Row_ID": row_ids,
            "Train_A_res_um2": signal,
            "Template": ["T11", "T10", "T01", "T00"] * (n_rows // 4),
            "EG_State": ["patterned", "blanket"] * (n_rows // 2),
            "fs_hz": fs,
            "fp_hz": fp,
            "k_eff2": (fp**2 - fs**2) / fp**2,
            "qs": 100.0 + signal,
            "qp": 200.0 + signal,
            "spurious_dangerous": signal > n_rows / 2,
        }
    )
    validity_columns = {
        "fs_hz": "target_valid_fs",
        "fp_hz": "target_valid_fp",
        "k_eff2": "target_valid_k_eff2",
        "qs": "target_valid_qs",
        "qp": "target_valid_qp",
        "spurious_dangerous": "target_valid_spurious",
    }
    for column in validity_columns.values():
        labels[column] = True
    labels["target_valid_fs"] = signal.astype(int) % 3 != 0
    labels.to_csv(directory / "labels.csv", index=False)

    fold = np.arange(n_rows) % 3
    folds = pd.DataFrame({"Training_Row_ID": row_ids, "fold": fold})
    folds.iloc[::-1].to_csv(directory / "folds.csv", index=False)
    spectrum = signal[:, None] * np.array([1 + 1j, 2 - 1j])
    reverse = np.arange(n_rows - 1, -1, -1)
    np.savez_compressed(
        directory / "spectra.npz",
        s11=spectrum[reverse],
        training_row_id=np.asarray(row_ids)[reverse],
    )
    np.save(directory / "frequency.npy", np.array([4.2e9, 5.39e9]))
    return labels


def test_loader_aligns_all_prepared_artifacts_by_training_row_id(tmp_path: Path):
    labels = _write_prepared(tmp_path / "prepared")

    dataset = load_prepared_dataset(tmp_path / "prepared")

    assert dataset.labels["Training_Row_ID"].tolist() == labels["Training_Row_ID"].tolist()
    assert dataset.folds["Training_Row_ID"].tolist() == labels["Training_Row_ID"].tolist()
    assert dataset.spectra[:, 0].real.tolist() == list(range(1, 13))
    assert dataset.frequency_hz.tolist() == [4.2e9, 5.39e9]
    assert set(TARGETS).issubset(dataset.targets.columns)
    assert "Training_Row_ID" in dataset.features
    assert "fs_hz" not in dataset.features
    assert dataset.area_res_tertile_edges == pytest.approx((14 / 3, 25 / 3))
    assert "prepared-v1" not in load_prepared_dataset.__doc__


@pytest.mark.parametrize("component", ["features", "targets", "folds", "spectra_ids"])
def test_oof_rejects_equal_length_shifted_component_ids(tmp_path: Path, component):
    _write_prepared(tmp_path / "prepared")
    dataset = load_prepared_dataset(tmp_path / "prepared")
    if component == "spectra_ids":
        broken = replace(dataset, spectra_ids=np.roll(dataset.spectra_ids, 1))
    else:
        shifted = getattr(dataset, component).iloc[np.roll(np.arange(12), 1)].reset_index(
            drop=True
        )
        broken = replace(dataset, **{component: shifted})

    with pytest.raises(ValueError, match=f"{component} Training_Row_ID order"):
        run_oof(lambda: _SignalRoute(_Recorder()), broken)


def test_oof_rejects_noncanonical_or_duplicate_component_ids(tmp_path: Path):
    _write_prepared(tmp_path / "prepared")
    dataset = load_prepared_dataset(tmp_path / "prepared")
    noncanonical = dataset.features.copy()
    noncanonical.loc[0, "Training_Row_ID"] = " row-00 "
    duplicate = dataset.targets.copy()
    duplicate.loc[0, "Training_Row_ID"] = duplicate.loc[1, "Training_Row_ID"]

    with pytest.raises(ValueError, match="features.*canonical"):
        run_oof(lambda: _SignalRoute(_Recorder()), replace(dataset, features=noncanonical))
    with pytest.raises(ValueError, match="targets.*duplicate"):
        run_oof(lambda: _SignalRoute(_Recorder()), replace(dataset, targets=duplicate))


@pytest.mark.parametrize(
    ("frequency", "message"),
    [
        (np.array([]), "frequency.*non-empty"),
        (np.array([4.2e9, np.nan]), "frequency.*finite"),
        (np.array([4.2e9, 4.2e9]), "frequency.*strictly increasing"),
        (np.array([4.3e9, 4.2e9]), "frequency.*strictly increasing"),
    ],
)
def test_loader_rejects_invalid_frequency_grid(tmp_path: Path, frequency, message):
    directory = tmp_path / "prepared"
    _write_prepared(directory)
    np.save(directory / "frequency.npy", frequency)
    if len(frequency) == 0:
        with np.load(directory / "spectra.npz", allow_pickle=False) as archive:
            ids = archive["training_row_id"]
        np.savez_compressed(
            directory / "spectra.npz",
            s11=np.empty((12, 0), dtype=complex),
            training_row_id=ids,
        )

    with pytest.raises(ValueError, match=message):
        load_prepared_dataset(directory)


@pytest.mark.parametrize("kind", ["real", "nonfinite"])
def test_loader_requires_finite_complex_spectra(tmp_path: Path, kind):
    directory = tmp_path / "prepared"
    _write_prepared(directory)
    with np.load(directory / "spectra.npz", allow_pickle=False) as archive:
        spectra = archive["s11"]
        ids = archive["training_row_id"]
    if kind == "real":
        spectra = spectra.real
    else:
        spectra = spectra.copy()
        spectra[0, 0] = np.nan + 1j * np.nan
    np.savez_compressed(
        directory / "spectra.npz", s11=spectra, training_row_id=ids
    )

    with pytest.raises(ValueError, match="spectra.*complex|spectra.*finite"):
        load_prepared_dataset(directory)


def test_physical_reconstruction_enforces_order_positive_q_and_coupling_identity():
    prediction = reconstruct_physical_targets(
        center_frequency_hz=np.array([5.0e9]),
        log_bandwidth_hz=np.log(np.array([200e6])),
        log_qs=np.log(np.array([300.0])),
        log_qp=np.log(np.array([400.0])),
    )

    assert prediction.loc[0, "fs_hz"] == pytest.approx(4.9e9)
    assert prediction.loc[0, "fp_hz"] == pytest.approx(5.1e9)
    assert prediction.loc[0, "fp_hz"] > prediction.loc[0, "fs_hz"]
    assert prediction.loc[0, "qs"] > 0
    assert prediction.loc[0, "qp"] > 0
    expected_k_eff2 = 1 - (4.9e9 / 5.1e9) ** 2
    assert prediction.loc[0, "k_eff2"] == pytest.approx(expected_k_eff2)


@pytest.mark.parametrize(
    "inputs",
    [
        {
            "center_frequency_hz": np.ones((2, 1)) * 5e9,
            "log_bandwidth_hz": np.ones((1, 2)) * np.log(2e8),
            "log_qs": np.ones((2, 1)) * np.log(300),
            "log_qp": np.ones((2, 1)) * np.log(400),
        },
        {
            "center_frequency_hz": np.array([5e9, 5.1e9]),
            "log_bandwidth_hz": np.log(2e8),
            "log_qs": np.log(np.array([300.0, 310.0])),
            "log_qp": np.log(np.array([400.0, 410.0])),
        },
    ],
)
def test_physical_reconstruction_rejects_broadcastable_mismatched_shapes(inputs):
    with pytest.raises(ValueError, match="same shape|all be scalars"):
        reconstruct_physical_targets(**inputs)


@dataclass
class _Recorder:
    fits: list[
        tuple[tuple[str, ...], tuple[str, ...], dict[str, tuple[float, bool, float]]]
    ] = field(default_factory=list)


class _SignalRoute:
    def __init__(self, recorder: _Recorder):
        self.recorder = recorder
        self.coefficients: dict[str, np.ndarray] = {}

    def fit(self, features, targets, *, spectra=None, frequency_hz=None):
        ids = tuple(features["Training_Row_ID"])
        fitted_rows = {
            row_id: (float(fs), bool(valid), float(spectrum.real))
            for row_id, fs, valid, spectrum in zip(
                ids,
                targets["fs_hz"],
                targets["target_valid_fs"],
                spectra[:, 0],
            )
        }
        self.recorder.fits.append(
            (ids, tuple(targets["Training_Row_ID"]), fitted_rows)
        )
        x = features["Train_A_res_um2"].to_numpy(dtype=float)
        for target in TARGETS[:-1]:
            self.coefficients[target] = np.polyfit(
                x, targets[target].to_numpy(dtype=float), 1
            )
        self.spurious_mean = float(targets["spurious_dangerous"].mean())
        assert spectra is not None and spectra.shape[0] == len(features)
        assert frequency_hz is not None
        return self

    def predict(self, features):
        x = features["Train_A_res_um2"].to_numpy(dtype=float)
        scalars = pd.DataFrame({"Training_Row_ID": features["Training_Row_ID"]})
        for target, coef in self.coefficients.items():
            scalars[target] = np.polyval(coef, x)
        scalars["spurious_dangerous"] = self.spurious_mean
        spectrum = x[:, None] * np.array([1 + 1j, 2 - 1j])
        order = np.arange(len(features) - 1, -1, -1)
        return RoutePrediction(
            scalars=scalars.iloc[order].reset_index(drop=True), spectrum=spectrum[order]
        )


def test_oof_fits_only_train_indices_and_retains_validation_ids(tmp_path: Path):
    labels = _write_prepared(tmp_path / "prepared")
    dataset = load_prepared_dataset(tmp_path / "prepared")
    recorder = _Recorder()

    result = run_oof(lambda: _SignalRoute(recorder), dataset)

    all_ids = set(labels["Training_Row_ID"])
    assert len(recorder.fits) == 3
    for fold_number, (fit_ids, target_ids, _) in enumerate(recorder.fits):
        validation_ids = set(
            dataset.folds.loc[dataset.folds["fold"] == fold_number, "Training_Row_ID"]
        )
        assert set(fit_ids) == all_ids - validation_ids
        assert set(fit_ids).isdisjoint(validation_ids)
        assert target_ids == fit_ids
    assert result.predictions["Training_Row_ID"].tolist() == labels["Training_Row_ID"].tolist()
    assert result.predictions["fold"].tolist() == dataset.folds["fold"].tolist()
    assert result.spectrum is not None
    np.testing.assert_allclose(result.spectrum, dataset.spectra)
    assert result.metrics["fs_mae_mhz"] < 1e-10
    assert result.metrics["spectrum_complex_rmse"] < 1e-10
    assert result.metrics["slices"]["Template"]["T11"]["count"] == 3
    assert result.metrics["slices"]["Template"]["T11"]["fs_mae_mhz"] < 1e-10
    assert result.metrics["slices"]["EG_State"]["blanket"]["count"] == 6
    assert result.metrics["slices"]["A_res_tertile"]["low"]["count"] == 4
    assert result.metrics["slices"]["A_res_tertile"]["mid"]["count"] == 4
    assert result.metrics["slices"]["A_res_tertile"]["high"]["count"] == 4


def test_permutation_control_shuffles_training_targets_only_and_loses_signal(tmp_path: Path):
    labels = _write_prepared(tmp_path / "prepared")
    dataset = load_prepared_dataset(tmp_path / "prepared")
    recorder = _Recorder()

    control = run_permutation_control(
        lambda: _SignalRoute(recorder), dataset, random_state=17
    )

    all_ids = set(labels["Training_Row_ID"])
    association_changed = False
    for fold_number, (fit_ids, target_ids, fitted_rows) in enumerate(recorder.fits):
        validation_ids = set(
            dataset.folds.loc[dataset.folds["fold"] == fold_number, "Training_Row_ID"]
        )
        assert set(fit_ids) == all_ids - validation_ids
        assert target_ids == fit_ids
        original = labels.set_index("Training_Row_ID").loc[list(fit_ids), "fs_hz"]
        original_validity = (
            labels.set_index("Training_Row_ID")
            .loc[list(fit_ids), "target_valid_fs"]
        )
        assert sorted((value[0], value[1]) for value in fitted_rows.values()) == sorted(
            zip(original.tolist(), original_validity.tolist())
        )
        for fs_hz, _, spectrum_signal in fitted_rows.values():
            assert fs_hz == pytest.approx(4.5e9 + spectrum_signal * 10e6)
        association_changed |= any(
            fitted_rows[row_id][0] != original.loc[row_id] for row_id in fit_ids
        )
    assert association_changed
    assert control.metrics["fs_mae_mhz"] > 10.0


class _SpectralSignalRoute:
    def fit(self, features, targets, *, spectra=None, frequency_hz=None):
        x = features["Train_A_res_um2"].to_numpy(dtype=float)
        self.real = [np.polyfit(x, spectra[:, j].real, 1) for j in range(spectra.shape[1])]
        self.imag = [np.polyfit(x, spectra[:, j].imag, 1) for j in range(spectra.shape[1])]
        self.scalar_means = {
            target: float(targets[target].mean()) for target in TARGETS
        }
        return self

    def predict(self, features):
        x = features["Train_A_res_um2"].to_numpy(dtype=float)
        scalars = pd.DataFrame({"Training_Row_ID": features["Training_Row_ID"]})
        for target, value in self.scalar_means.items():
            scalars[target] = value
        spectrum = np.column_stack(
            [
                np.polyval(real, x) + 1j * np.polyval(imag, x)
                for real, imag in zip(self.real, self.imag)
            ]
        )
        return RoutePrediction(scalars=scalars, spectrum=spectrum)


def test_permutation_control_also_breaks_spectral_signal(tmp_path: Path):
    _write_prepared(tmp_path / "prepared")
    dataset = load_prepared_dataset(tmp_path / "prepared")

    real = run_oof(_SpectralSignalRoute, dataset)
    control = run_permutation_control(
        _SpectralSignalRoute, dataset, random_state=17
    )

    assert real.metrics["spectrum_complex_rmse"] < 1e-10
    assert control.metrics["spectrum_complex_rmse"] > 1.0


class _IncompleteRoute(_SignalRoute):
    def predict(self, features):
        prediction = super().predict(features)
        return RoutePrediction(
            scalars=prediction.scalars.drop(columns="qp"),
            spectrum=prediction.spectrum,
        )


def test_oof_rejects_route_predictions_missing_contract_targets(tmp_path: Path):
    _write_prepared(tmp_path / "prepared")
    dataset = load_prepared_dataset(tmp_path / "prepared")

    with pytest.raises(ValueError, match="missing scalar prediction columns"):
        run_oof(lambda: _IncompleteRoute(_Recorder()), dataset)


class _BadIDRoute(_SignalRoute):
    def __init__(self, recorder, mode):
        super().__init__(recorder)
        self.mode = mode

    def predict(self, features):
        prediction = super().predict(features)
        scalars = prediction.scalars.copy()
        if self.mode == "missing_column":
            scalars = scalars.drop(columns="Training_Row_ID")
        elif self.mode == "duplicate":
            scalars.loc[0, "Training_Row_ID"] = scalars.loc[1, "Training_Row_ID"]
        elif self.mode == "mismatched_set":
            scalars.loc[0, "Training_Row_ID"] = "not-a-validation-row"
        elif self.mode == "missing_row":
            scalars = scalars.iloc[:-1]
        return RoutePrediction(scalars=scalars, spectrum=prediction.spectrum)


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("missing_column", "Training_Row_ID"),
        ("duplicate", "duplicate"),
        ("mismatched_set", "set does not match"),
        ("missing_row", "row count"),
    ],
)
def test_oof_rejects_invalid_route_prediction_ids(tmp_path: Path, mode, message):
    _write_prepared(tmp_path / "prepared")
    dataset = load_prepared_dataset(tmp_path / "prepared")

    with pytest.raises(ValueError, match=message):
        run_oof(lambda: _BadIDRoute(_Recorder(), mode), dataset)
