from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aln_model.features import PHYSICAL_COLUMNS, STATE_COLUMNS
from aln_model.routes.functional_pca import FunctionalPCARoute


def _features(n: int) -> pd.DataFrame:
    rows = pd.DataFrame({"Training_Row_ID": [f"row-{i}" for i in range(n)]})
    for j, column in enumerate(PHYSICAL_COLUMNS):
        rows[column] = 1.0 + j + np.linspace(0, 1, n)
    rows["Template"] = np.resize(["T11", "T10", "T01", "T00"], n)
    rows["EG_State"] = np.resize(["patterned", "blanket"], n)
    for column in STATE_COLUMNS:
        if column not in {"Template", "EG_State"}:
            rows[column] = np.resize([0, 1], n)
    return rows


def _targets(n: int) -> pd.DataFrame:
    fs = 4.8e9 + np.arange(n) * 1e6
    fp = fs + 3.0e8 + np.arange(n) * 1e5
    return pd.DataFrame(
        {
            "Training_Row_ID": [f"row-{i}" for i in range(n)],
            "fs_hz": fs,
            "fp_hz": fp,
            "k_eff2": 1 - (fs / fp) ** 2,
            "qs": 350 + np.arange(n),
            "qp": 220 + np.arange(n),
            "spurious_dangerous": np.resize([0, 0, 1], n),
        }
    )


def _spectra(n: int, width: int) -> np.ndarray:
    phase = np.linspace(0, 2 * np.pi, width)
    scale = np.linspace(0.7, 0.9, n)[:, None]
    return scale * np.exp(1j * (phase[None, :] + np.arange(n)[:, None] / 20))


def test_functional_pca_route_predicts_constrained_scalars_and_complex_spectrum():
    n, width = 12, 21
    route = FunctionalPCARoute(n_spectrum_components=5, ridge_alpha=1.0)
    features = _features(n)
    route.fit(
        features,
        _targets(n),
        spectra=_spectra(n, width),
        frequency_hz=np.linspace(4.2e9, 5.39e9, width),
    )

    prediction = route.predict(features.iloc[[3, 1, 9]].reset_index(drop=True))

    assert prediction.scalars["Training_Row_ID"].tolist() == ["row-3", "row-1", "row-9"]
    assert prediction.spectrum is not None
    assert prediction.spectrum.shape == (3, width)
    assert np.iscomplexobj(prediction.spectrum)
    assert np.isfinite(prediction.spectrum.real).all()
    assert np.isfinite(prediction.spectrum.imag).all()
    assert (np.abs(prediction.spectrum) <= 1.0 + 1e-12).all()
    scalar = prediction.scalars
    assert (scalar["fp_hz"] > scalar["fs_hz"]).all()
    assert (scalar[["qs", "qp"]] > 0).all().all()
    np.testing.assert_allclose(
        scalar["k_eff2"], 1 - (scalar["fs_hz"] / scalar["fp_hz"]) ** 2
    )
    assert scalar["spurious_dangerous"].between(0, 1).all()


def test_functional_pca_route_requires_valid_spectrum_contract():
    route = FunctionalPCARoute(n_spectrum_components=2)
    features = _features(6)
    targets = _targets(6)
    with pytest.raises(ValueError, match="spectra"):
        route.fit(features, targets)
    with pytest.raises(ValueError, match="frequency"):
        route.fit(
            features,
            targets,
            spectra=_spectra(6, 8),
            frequency_hz=np.arange(7, dtype=float),
        )


def test_functional_pca_route_ignores_explicitly_invalid_target_rows():
    n, width = 9, 12
    features = _features(n)
    targets = _targets(n)
    for name in ("fs", "fp", "k_eff2", "qs", "qp", "spurious"):
        targets[f"target_valid_{name}"] = True
    targets.loc[0, "qs"] = np.nan
    targets.loc[0, "target_valid_qs"] = False
    targets.loc[1, "spurious_dangerous"] = np.nan
    targets.loc[1, "target_valid_spurious"] = False

    route = FunctionalPCARoute(n_spectrum_components=3).fit(
        features,
        targets,
        spectra=_spectra(n, width),
        frequency_hz=np.linspace(4.2e9, 5.39e9, width),
    )

    assert route.training_counts_ == {"frequency": 9, "qs": 8, "qp": 9, "spurious": 8}
    prediction = route.predict(features.iloc[:2])
    assert np.isfinite(
        prediction.scalars[["fs_hz", "fp_hz", "qs", "qp"]].to_numpy(float)
    ).all()
