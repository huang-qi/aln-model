from __future__ import annotations

import hashlib
import json
import platform
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from .evaluate import load_prepared_dataset
from .experiments import default_route_factories
from .features import PHYSICAL_COLUMNS, STATE_COLUMNS
from .prepare import _atomic_publish_directory
from .routes.base import Route


@dataclass(frozen=True)
class ApplicabilityDomain:
    numeric_bounds: dict[str, tuple[float, float]]
    categorical_values: dict[str, tuple[str, ...]]
    numeric_allows_missing: dict[str, bool] = field(default_factory=dict)

    @classmethod
    def from_features(cls, features: pd.DataFrame) -> ApplicabilityDomain:
        numeric: dict[str, tuple[float, float]] = {}
        allows_missing: dict[str, bool] = {}
        for column in PHYSICAL_COLUMNS:
            if column in features:
                values = pd.to_numeric(features[column], errors="coerce").to_numpy(float)
                finite = values[np.isfinite(values)]
                if finite.size:
                    numeric[column] = (float(finite.min()), float(finite.max()))
                    allows_missing[column] = bool((~np.isfinite(values)).any())
        categorical: dict[str, tuple[str, ...]] = {}
        for column in ("Template", "EG_State"):
            if column in features:
                values = features[column].dropna().astype(str).str.strip()
                categorical[column] = tuple(sorted(values.unique().tolist()))
        return cls(
            numeric_bounds=numeric,
            categorical_values=categorical,
            numeric_allows_missing=allows_missing,
        )

    def assess(self, features: pd.DataFrame) -> pd.DataFrame:
        reasons: list[list[str]] = [[] for _ in range(len(features))]
        for column, (lower, upper) in self.numeric_bounds.items():
            if column not in features:
                for row in reasons:
                    row.append(column)
                continue
            values = pd.to_numeric(features[column], errors="coerce").to_numpy(float)
            finite = np.isfinite(values)
            missing_is_ood = not self.numeric_allows_missing.get(column, False)
            invalid = (~finite & missing_is_ood) | (
                finite & ((values < lower) | (values > upper))
            )
            for position in np.flatnonzero(invalid):
                reasons[int(position)].append(column)
        for column, allowed in self.categorical_values.items():
            if column not in features:
                for row in reasons:
                    row.append(column)
                continue
            values = features[column].fillna("").astype(str).str.strip()
            invalid = ~values.isin(allowed).to_numpy()
            for position in np.flatnonzero(invalid):
                reasons[int(position)].append(column)
        return pd.DataFrame(
            {
                "ood": [bool(row) for row in reasons],
                "ood_reasons": [";".join(row) for row in reasons],
            }
        )


@dataclass(frozen=True)
class SurrogatePrediction:
    scalars: pd.DataFrame
    spectrum: np.ndarray
    frequency_hz: np.ndarray


@dataclass
class SurrogateBundle:
    scalar_route: Route
    spectrum_route: Route
    frequency_hz: np.ndarray
    applicability: ApplicabilityDomain
    scalar_route_name: str
    spectrum_route_name: str

    def predict(self, features: pd.DataFrame) -> SurrogatePrediction:
        scalar = self.scalar_route.predict(features).scalars.reset_index(drop=True)
        spectral_prediction = self.spectrum_route.predict(features)
        if spectral_prediction.spectrum is None:
            raise RuntimeError("configured spectrum route did not return a spectrum")
        spectral_ids = spectral_prediction.scalars["Training_Row_ID"].astype(str)
        scalar_ids = scalar["Training_Row_ID"].astype(str)
        if not spectral_ids.equals(scalar_ids):
            raise ValueError("scalar and spectrum route prediction IDs differ")
        ood = self.applicability.assess(features)
        scalar["ood"] = ood["ood"].to_numpy()
        scalar["ood_reasons"] = ood["ood_reasons"].to_numpy()
        spectrum = np.asarray(spectral_prediction.spectrum)
        if spectrum.shape != (len(features), len(self.frequency_hz)):
            raise ValueError("spectrum route output shape differs from model frequency grid")
        return SurrogatePrediction(
            scalars=scalar,
            spectrum=spectrum,
            frequency_hz=np.asarray(self.frequency_hz).copy(),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_final_bundle(
    prepared_directory: str | Path,
    benchmark_directory: str | Path,
    output_directory: str | Path,
) -> dict[str, object]:
    """Fit selected routes on all prepared rows and publish a model bundle."""

    prepared = Path(prepared_directory).expanduser().resolve()
    benchmark = Path(benchmark_directory).expanduser().resolve()
    benchmark_manifest = json.loads((benchmark / "benchmark.json").read_text("utf-8"))
    decision = benchmark_manifest["decision"]
    scalar_name = decision["scalar_winner"]
    spectrum_name = decision["spectrum_winner"]
    if spectrum_name is None:
        raise ValueError("benchmark did not produce a valid spectrum route")
    factories = default_route_factories()
    if scalar_name not in factories or spectrum_name not in factories:
        raise ValueError("benchmark selected an unknown route")
    dataset = load_prepared_dataset(prepared)
    scalar_route = factories[scalar_name]()
    scalar_route.fit(
        dataset.features,
        dataset.targets,
        spectra=dataset.spectra,
        frequency_hz=dataset.frequency_hz,
    )
    if spectrum_name == scalar_name:
        spectrum_route = scalar_route
    else:
        spectrum_route = factories[spectrum_name]()
        spectrum_route.fit(
            dataset.features,
            dataset.targets,
            spectra=dataset.spectra,
            frequency_hz=dataset.frequency_hz,
        )
    bundle = SurrogateBundle(
        scalar_route=scalar_route,
        spectrum_route=spectrum_route,
        frequency_hz=dataset.frequency_hz.copy(),
        applicability=ApplicabilityDomain.from_features(dataset.features),
        scalar_route_name=scalar_name,
        spectrum_route_name=spectrum_name,
    )

    output = Path(output_directory).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        joblib.dump(bundle, temporary / "model.joblib", compress=3)
        metadata: dict[str, object] = {
            "scalar_route": scalar_name,
            "spectrum_route": spectrum_name,
            "training_rows": len(dataset.features),
            "frequency_points": len(dataset.frequency_hz),
            "frequency_start_hz": float(dataset.frequency_hz[0]),
            "frequency_stop_hz": float(dataset.frequency_hz[-1]),
            "prepared_audit_sha256": _sha256(prepared / "audit.json"),
            "benchmark_sha256": _sha256(benchmark / "benchmark.json"),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "native_label_head": True,
            "shared_band_spectrum_head": True,
            "note": (
                "Engineering labels are learned from full native sweeps; the complex "
                "spectrum head covers only the shared 4.20-5.39 GHz grid."
            ),
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        (temporary / "applicability.json").write_text(
            json.dumps(
                {
                    "numeric_bounds": bundle.applicability.numeric_bounds,
                    "numeric_allows_missing": bundle.applicability.numeric_allows_missing,
                    "categorical_values": bundle.applicability.categorical_values,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        _atomic_publish_directory(temporary, output)
        return metadata
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def load_bundle(path: str | Path) -> SurrogateBundle:
    model = joblib.load(Path(path))
    if not isinstance(model, SurrogateBundle):
        raise TypeError("model file does not contain an ALN SurrogateBundle")
    return model
