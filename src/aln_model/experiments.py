from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import OOFResult, load_prepared_dataset, run_oof, run_permutation_control
from .prepare import _atomic_publish_directory
from .routes.base import Route
from .routes.extratrees import ExtraTreesRawRoute
from .routes.functional_pca import FunctionalPCARoute
from .routes.physics_boost import PhysicsBoostRoute
from .routes.rff_kernel import RFFKernelRoute


LOWER_IS_BETTER = (
    "fs_mae_mhz",
    "fp_mae_mhz",
    "k_eff2_mae_percentage_points",
    "qs_log_mae",
    "qp_log_mae",
    "worst5_frequency_error_mhz",
    "spurious_brier",
    "physical_violation_rate",
)
HIGHER_IS_BETTER = ("spurious_aucpr",)


def default_route_factories() -> dict[str, Callable[[], Route]]:
    """Frozen, reproducible route configurations used by the benchmark."""

    return {
        "extra_trees_raw": lambda: ExtraTreesRawRoute(
            n_estimators=200,
            min_samples_leaf=2,
            n_jobs=8,
            random_state=20260818,
        ),
        "physics_boost": lambda: PhysicsBoostRoute(
            max_iter=180, random_state=20260818
        ),
        "rff_kernel": lambda: RFFKernelRoute(
            n_components=384, ridge_alpha=1.0, random_state=20260818
        ),
        "functional_pca": lambda: FunctionalPCARoute(
            n_spectrum_components=24, ridge_alpha=10.0, random_state=20260818
        ),
    }


def _finite_number(value: object, *, higher: bool) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return -np.inf if higher else np.inf
    if not np.isfinite(numeric):
        return -np.inf if higher else np.inf
    return numeric


def adjudicate_routes(
    metrics: Mapping[str, Mapping[str, object]],
    permutation_metrics: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Rank scalar routes by transparent equal-weight metric ranks.

    Metrics have incompatible units, so ranking them before averaging avoids an
    arbitrary unit-dependent weighted sum. Non-finite values receive the worst
    rank. Permutation results are reported as a signal audit, not folded into the
    winning score.
    """

    if not metrics:
        raise ValueError("at least one successful route is required")
    names = list(metrics)
    rank_frame = pd.DataFrame(index=names, dtype=float)
    for key in LOWER_IS_BETTER:
        values = pd.Series(
            {name: _finite_number(metrics[name].get(key), higher=False) for name in names}
        )
        rank_frame[key] = values.rank(method="average", ascending=True)
    for key in HIGHER_IS_BETTER:
        values = pd.Series(
            {name: _finite_number(metrics[name].get(key), higher=True) for name in names}
        )
        rank_frame[key] = values.rank(method="average", ascending=False)
    average_rank = rank_frame.mean(axis=1)

    rows: list[dict[str, object]] = []
    for name in names:
        won = 0
        tested = 0
        control = permutation_metrics.get(name)
        if control is not None:
            for key in LOWER_IS_BETTER:
                real = _finite_number(metrics[name].get(key), higher=False)
                shuffled = _finite_number(control.get(key), higher=False)
                if np.isfinite(real) and np.isfinite(shuffled):
                    tested += 1
                    won += int(real < shuffled)
            for key in HIGHER_IS_BETTER:
                real = _finite_number(metrics[name].get(key), higher=True)
                shuffled = _finite_number(control.get(key), higher=True)
                if np.isfinite(real) and np.isfinite(shuffled):
                    tested += 1
                    won += int(real > shuffled)
        rows.append(
            {
                "route": name,
                "average_metric_rank": float(average_rank[name]),
                "signal_metrics_won": won,
                "signal_metrics_tested": tested,
            }
        )
    rows.sort(key=lambda row: (row["average_metric_rank"], row["route"]))
    best_score = float(rows[0]["average_metric_rank"])
    tied = [
        str(row["route"])
        for row in rows
        if np.isclose(float(row["average_metric_rank"]), best_score)
    ]
    spectral = [
        (name, _finite_number(values.get("spectrum_complex_rmse"), higher=False))
        for name, values in metrics.items()
        if "spectrum_complex_rmse" in values
    ]
    spectral = [item for item in spectral if np.isfinite(item[1])]
    return {
        "scalar_winner": str(rows[0]["route"]),
        "scalar_ties": tied,
        "spectrum_winner": min(spectral, key=lambda item: item[1])[0]
        if spectral
        else None,
        "ranking": rows,
        "method": "equal-weight average ranks across frozen scalar metrics",
    }


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def _write_result(directory: Path, name: str, result: OOFResult) -> None:
    result.predictions.to_csv(directory / f"{name}_oof.csv", index=False)
    if result.spectrum is not None:
        np.savez_compressed(
            directory / f"{name}_oof_spectra.npz",
            s11=result.spectrum,
            training_row_id=result.predictions["Training_Row_ID"].to_numpy(str),
        )
    (directory / f"{name}_metrics.json").write_text(
        json.dumps(result.metrics, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def run_benchmark(
    prepared_directory: str | Path,
    output_directory: str | Path,
    *,
    route_factories: Mapping[str, Callable[[], Route]] | None = None,
    run_permutations: bool = True,
) -> dict[str, object]:
    """Run all routes against identical folds and publish immutable OOF outputs."""

    dataset = load_prepared_dataset(prepared_directory)
    factories = dict(route_factories or default_route_factories())
    if not factories:
        raise ValueError("route_factories cannot be empty")
    output = Path(output_directory).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    metrics: dict[str, Mapping[str, object]] = {}
    controls: dict[str, Mapping[str, object]] = {}
    failures: dict[str, str] = {}
    try:
        for name, factory in factories.items():
            try:
                result = run_oof(factory, dataset)
                _write_result(temporary, name, result)
                metrics[name] = result.metrics
                if run_permutations:
                    control = run_permutation_control(
                        factory, dataset, random_state=20260818
                    )
                    controls[name] = control.metrics
                    (temporary / f"{name}_permutation_metrics.json").write_text(
                        json.dumps(
                            control.metrics,
                            indent=2,
                            sort_keys=True,
                            default=_json_default,
                        ),
                        encoding="utf-8",
                    )
            except Exception as exc:  # each route is an independent experiment
                failures[name] = f"{type(exc).__name__}: {exc}"
        decision = adjudicate_routes(metrics, controls)
        manifest = {
            "prepared_directory": str(Path(prepared_directory).resolve()),
            "row_count": len(dataset.labels),
            "fold_count": int(dataset.folds["fold"].nunique()),
            "routes_requested": list(factories),
            "routes_succeeded": list(metrics),
            "route_failures": failures,
            "permutation_controls_run": run_permutations,
            "decision": decision,
        }
        (temporary / "benchmark.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=_json_default),
            encoding="utf-8",
        )
        pd.DataFrame(decision["ranking"]).to_csv(
            temporary / "ranking.csv", index=False
        )
        _atomic_publish_directory(temporary, output)
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
