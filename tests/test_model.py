from __future__ import annotations

import numpy as np
import pandas as pd

from aln_model.model import ApplicabilityDomain, SurrogateBundle
from aln_model.routes.base import RoutePrediction


class _ScalarRoute:
    def predict(self, features: pd.DataFrame) -> RoutePrediction:
        n = len(features)
        fs = np.full(n, 4.8e9)
        fp = np.full(n, 5.1e9)
        return RoutePrediction(
            pd.DataFrame(
                {
                    "Training_Row_ID": features["Training_Row_ID"],
                    "fs_hz": fs,
                    "fp_hz": fp,
                    "k_eff2": 1 - (fs / fp) ** 2,
                    "qs": 400.0,
                    "qp": 250.0,
                    "spurious_dangerous": 0.1,
                }
            )
        )


class _SpectrumRoute(_ScalarRoute):
    def predict(self, features: pd.DataFrame) -> RoutePrediction:
        base = super().predict(features)
        return RoutePrediction(base.scalars, np.full((len(features), 3), 0.5 + 0.1j))


def test_bundle_combines_scalar_and_spectrum_heads_with_ood_flags():
    domain = ApplicabilityDomain(
        numeric_bounds={"Train_A_res_um2": (10.0, 20.0)},
        categorical_values={"Template": ("T11", "T10")},
    )
    bundle = SurrogateBundle(
        scalar_route=_ScalarRoute(),
        spectrum_route=_SpectrumRoute(),
        frequency_hz=np.array([4.2e9, 4.3e9, 4.4e9]),
        applicability=domain,
        scalar_route_name="scalar",
        spectrum_route_name="spectrum",
    )
    features = pd.DataFrame(
        {
            "Training_Row_ID": ["inside", "outside"],
            "Train_A_res_um2": [15.0, 25.0],
            "Template": ["T11", "NEW"],
        }
    )

    result = bundle.predict(features)

    assert result.scalars["Training_Row_ID"].tolist() == ["inside", "outside"]
    assert result.scalars["ood"].tolist() == [False, True]
    assert result.scalars.loc[1, "ood_reasons"] == "Train_A_res_um2;Template"
    assert result.spectrum.shape == (2, 3)
    np.testing.assert_array_equal(result.frequency_hz, bundle.frequency_hz)


def test_applicability_preserves_missingness_seen_during_training():
    training = pd.DataFrame({"Train_PF_nm": [100.0, np.nan, 200.0]})
    domain = ApplicabilityDomain.from_features(training)

    assessed = domain.assess(pd.DataFrame({"Train_PF_nm": [np.nan, 300.0]}))

    assert assessed["ood"].tolist() == [False, True]
