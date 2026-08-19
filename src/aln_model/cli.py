from __future__ import annotations

import argparse
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PreparationConfig
from .prepare import _atomic_publish_directory, prepare_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aln-model")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="prepare immutable shared artifacts")
    prepare.add_argument("--workbook", required=True)
    prepare.add_argument("--archive", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--frequency-start-hz", type=float, default=4.2e9)
    prepare.add_argument("--frequency-stop-hz", type=float, default=5.39e9)
    prepare.add_argument("--frequency-step-hz", type=float, default=1e6)
    prepare.add_argument("--folds", type=int, default=5)
    prepare.add_argument("--random-state", type=int, default=20260818)
    benchmark = subparsers.add_parser(
        "benchmark", help="run common-fold OOF evaluation for every route"
    )
    benchmark.add_argument("--prepared", required=True)
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--skip-permutation", action="store_true")
    train = subparsers.add_parser("train", help="fit and publish the selected bundle")
    train.add_argument("--prepared", required=True)
    train.add_argument("--benchmark", required=True)
    train.add_argument("--output", required=True)
    predict = subparsers.add_parser("predict", help="predict labels and shared-band S11")
    predict.add_argument("--model", required=True)
    predict.add_argument("--input", required=True)
    predict.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        config = PreparationConfig(
            frequency_start_hz=args.frequency_start_hz,
            frequency_stop_hz=args.frequency_stop_hz,
            frequency_step_hz=args.frequency_step_hz,
            n_folds=args.folds,
            random_state=args.random_state,
        )
        prepare_dataset(
            args.workbook,
            args.archive,
            args.output,
            config=config,
        )
        return 0
    if args.command == "benchmark":
        from .experiments import run_benchmark

        run_benchmark(
            args.prepared,
            args.output,
            run_permutations=not args.skip_permutation,
        )
        return 0
    if args.command == "train":
        from .model import train_final_bundle

        train_final_bundle(args.prepared, args.benchmark, args.output)
        return 0
    if args.command == "predict":
        from .model import load_bundle

        features = pd.read_csv(args.input)
        if "Training_Row_ID" not in features:
            features.insert(
                0,
                "Training_Row_ID",
                [f"prediction-{index + 1:06d}" for index in range(len(features))],
            )
        prediction = load_bundle(args.model).predict(features)
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent)
        )
        try:
            prediction.scalars.to_csv(
                temporary / "prediction_labels.csv", index=False
            )
            np.savez_compressed(
                temporary / "prediction_spectra.npz",
                s11=prediction.spectrum,
                frequency_hz=prediction.frequency_hz,
                training_row_id=prediction.scalars["Training_Row_ID"].to_numpy(str),
            )
            _atomic_publish_directory(temporary, output)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
