from __future__ import annotations

import math

from aln_model.experiments import adjudicate_routes


def test_adjudication_ranks_lower_errors_and_higher_aucpr():
    metrics = {
        "route-a": {
            "fs_mae_mhz": 2.0,
            "fp_mae_mhz": 3.0,
            "k_eff2_mae_percentage_points": 0.2,
            "qs_log_mae": 0.1,
            "qp_log_mae": 0.1,
            "worst5_frequency_error_mhz": 8.0,
            "spurious_brier": 0.02,
            "spurious_aucpr": 0.8,
            "physical_violation_rate": 0.0,
        },
        "route-b": {
            "fs_mae_mhz": 4.0,
            "fp_mae_mhz": 5.0,
            "k_eff2_mae_percentage_points": 0.5,
            "qs_log_mae": 0.2,
            "qp_log_mae": 0.3,
            "worst5_frequency_error_mhz": 12.0,
            "spurious_brier": 0.08,
            "spurious_aucpr": 0.4,
            "physical_violation_rate": 0.0,
        },
    }
    permutation = {
        name: {
            **values,
            "fs_mae_mhz": values["fs_mae_mhz"] * 10,
            "fp_mae_mhz": values["fp_mae_mhz"] * 10,
            "spurious_aucpr": 0.01,
        }
        for name, values in metrics.items()
    }

    decision = adjudicate_routes(metrics, permutation)

    assert decision["scalar_winner"] == "route-a"
    assert decision["ranking"][0]["route"] == "route-a"
    assert decision["ranking"][0]["signal_metrics_won"] >= 3


def test_adjudication_penalizes_nonfinite_metrics():
    base = {
        "fs_mae_mhz": 1.0,
        "fp_mae_mhz": 1.0,
        "k_eff2_mae_percentage_points": 0.1,
        "qs_log_mae": 0.1,
        "qp_log_mae": 0.1,
        "worst5_frequency_error_mhz": 2.0,
        "spurious_brier": 0.1,
        "spurious_aucpr": 0.5,
        "physical_violation_rate": 0.0,
    }
    metrics = {"valid": dict(base), "broken": {**base, "fs_mae_mhz": math.inf}}

    decision = adjudicate_routes(metrics, {})

    assert decision["scalar_winner"] == "valid"
    assert decision["ranking"][-1]["route"] == "broken"
