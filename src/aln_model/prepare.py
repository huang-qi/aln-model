from __future__ import annotations

import json
import ctypes
import errno
import os
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .config import PreparationConfig
from .data import load_training_index_with_audit, map_archive_members
from .features import STATE_COLUMNS, geometry_fingerprint
from .folds import (
    assign_group_folds,
    connected_group_ids,
    response_near_duplicate_clusters,
    response_quantized_hash,
)
from .labels import extract_engineering_labels
from .touchstone import interpolate_complex, parse_touchstone_s11


@dataclass(frozen=True)
class PreparationArtifacts:
    labels_csv: Path
    spectra_npz: Path
    frequency_npy: Path
    folds_csv: Path
    audit_json: Path
    audit_markdown: Path


def _write_audit_markdown(path: Path, audit: dict[str, object]) -> None:
    invalid_reasons = audit["invalid_reason_counts"]
    lines = [
        "# Preparation audit",
        "",
        f"- Selected rows: {audit['selected_rows']}",
        f"- Valid labels: {audit['valid_labels']}",
        f"- Invalid labels: {audit['invalid_labels']}",
        f"- Frequency points: {audit['frequency_points']}",
        f"- Geometry groups: {audit['geometry_groups']}",
        f"- Response groups: {audit['response_groups']}",
        f"- Final connected groups: {audit['final_groups']}",
        "",
        "## Invalid reason counts",
        "",
    ]
    if invalid_reasons:
        lines.extend(f"- {reason}: {count}" for reason, count in invalid_reasons.items())
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prepare_dataset_in_directory(
    workbook: str | Path,
    archive: str | Path,
    output_directory: str | Path,
    *,
    config: PreparationConfig | None = None,
) -> PreparationArtifacts:
    config = config or PreparationConfig()
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    selected_rows, selection_audit = load_training_index_with_audit(workbook)
    rows = map_archive_members(selected_rows, archive)
    fingerprints = geometry_fingerprint(rows)
    frequency_grid = config.frequency_grid()
    spectra: list[np.ndarray] = []
    label_records: list[dict[str, object]] = []
    with ZipFile(archive) as zf:
        for (_, row), fingerprint in zip(rows.iterrows(), fingerprints):
            member = str(row["ZIP_Member"])
            try:
                text = zf.read(member).decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise ValueError(f"{member}: Touchstone file is not UTF-8 text") from exc
            source_frequency, source_s11 = parse_touchstone_s11(text, path=member)
            shared_s11 = interpolate_complex(
                source_frequency, source_s11, frequency_grid
            )
            labels = extract_engineering_labels(
                source_frequency,
                source_s11,
                algorithm_version=config.label_algorithm_version,
            )
            spectra.append(shared_s11)
            fs_outside_shared_band = bool(
                np.isfinite(labels.fs_hz)
                and not (
                    config.frequency_start_hz
                    <= labels.fs_hz
                    <= config.frequency_stop_hz
                )
            )
            fp_outside_shared_band = bool(
                np.isfinite(labels.fp_hz)
                and not (
                    config.frequency_start_hz
                    <= labels.fp_hz
                    <= config.frequency_stop_hz
                )
            )
            has_pf_value = pd.to_numeric(
                pd.Series([row.get("has_PF", 0)]), errors="coerce"
            ).fillna(0).iloc[0]
            pf_value_missing = bool(
                has_pf_value != 0 and pd.isna(row.get("Train_PF_nm", np.nan))
            )
            record = {
                "Training_Row_ID": row["Training_Row_ID"],
                "SNP_Relative_Path": row["SNP_Relative_Path"],
                "ZIP_Member": member,
                "geometry_fingerprint": fingerprint,
                **labels.as_dict(),
                "fs_outside_shared_band": fs_outside_shared_band,
                "fp_outside_shared_band": fp_outside_shared_band,
                "pf_value_missing": pf_value_missing,
            }
            audit_columns = {
                "Training_Ready",
                "Cross_Source_Status",
                "Diff_Fields",
                "Match_Method",
                "Match_Confidence",
                "Notes_and_Normalization",
            }
            retained_columns = set(STATE_COLUMNS) | audit_columns
            for column in rows.columns:
                if (
                    column.startswith("Train_")
                    or column.startswith("SNP_")
                    or column in retained_columns
                ):
                    record[column] = row[column]
            label_records.append(record)

    labels_table = pd.DataFrame(label_records)
    spectra_matrix = np.stack(spectra)
    response_hashes = response_quantized_hash(
        spectra_matrix, tolerance=config.response_hash_tolerance
    )
    response_ids = response_near_duplicate_clusters(
        spectra_matrix, tolerance=config.response_hash_tolerance
    )
    final_group_ids = connected_group_ids(fingerprints, response_ids)
    fold_input = rows.loc[:, ["Template", "EG_State"]].copy()
    fold_input["geometry_fingerprint"] = fingerprints.to_numpy()
    fold_input["final_group_id"] = final_group_ids.to_numpy()
    fold_input["spurious_dangerous"] = labels_table[
        "spurious_dangerous"
    ].to_numpy()
    fold_values = assign_group_folds(
        fold_input,
        n_splits=config.n_folds,
        random_state=config.random_state,
        balance_columns=["spurious_dangerous"],
    )
    folds_table = pd.DataFrame(
        {
            "Training_Row_ID": rows["Training_Row_ID"].to_numpy(),
            "geometry_fingerprint": fingerprints.to_numpy(),
            "response_cluster_id": response_ids.to_numpy(),
            "response_hash": response_hashes.to_numpy(),
            "final_group_id": final_group_ids.to_numpy(),
            "fold": fold_values.to_numpy(),
        }
    )
    grouped = folds_table.groupby("final_group_id")
    folds_table["group_member_count"] = folds_table["final_group_id"].map(
        grouped.size()
    )
    folds_table["group_geometry_count"] = folds_table["final_group_id"].map(
        grouped["geometry_fingerprint"].nunique()
    )
    folds_table["group_response_count"] = folds_table["final_group_id"].map(
        grouped["response_cluster_id"].nunique()
    )

    labels_path = output_directory / "labels.csv"
    spectra_path = output_directory / "spectra.npz"
    frequency_path = output_directory / "frequency.npy"
    folds_path = output_directory / "folds.csv"
    audit_json_path = output_directory / "audit.json"
    audit_markdown_path = output_directory / "audit.md"
    labels_table.to_csv(labels_path, index=False)
    np.savez_compressed(
        spectra_path,
        s11=spectra_matrix,
        training_row_id=np.asarray(rows["Training_Row_ID"].astype(str).tolist(), dtype="U"),
    )
    np.save(frequency_path, frequency_grid)
    folds_table.to_csv(folds_path, index=False)

    invalid_reasons = Counter(
        labels_table.loc[~labels_table["quality_valid"], "quality_reason"].astype(str)
    )
    status_columns = [
        "Training_Ready",
        "Cross_Source_Status",
        "Diff_Fields",
        "Match_Method",
        "Match_Confidence",
    ]
    audit_status_counts = {
        column: {
            str(key): int(value)
            for key, value in rows[column]
            .fillna("<MISSING>")
            .astype(str)
            .value_counts()
            .items()
        }
        for column in status_columns
        if column in rows
    }
    override_status_counts = {
        key: value
        for key, value in audit_status_counts.get("Training_Ready", {}).items()
        if key != "YES"
    }
    audit = {
        **selection_audit,
        "valid_labels": int(labels_table["quality_valid"].sum()),
        "invalid_labels": int((~labels_table["quality_valid"]).sum()),
        "invalid_reason_counts": dict(sorted(invalid_reasons.items())),
        "boundary_labels": int(labels_table["quality_boundary"].sum()),
        "ambiguity_labels": int(labels_table["quality_ambiguous"].sum()),
        "mode_crossing_labels": int(labels_table["quality_mode_crossing"].sum()),
        "spurious_labels": int(labels_table["spurious_dangerous"].sum()),
        "fs_outside_shared_band_count": int(
            labels_table["fs_outside_shared_band"].sum()
        ),
        "fp_outside_shared_band_count": int(
            labels_table["fp_outside_shared_band"].sum()
        ),
        "pf_value_missing_count": int(labels_table["pf_value_missing"].sum()),
        "audit_status_counts": audit_status_counts,
        "override_status_counts": override_status_counts,
        "frequency_start_hz": float(frequency_grid[0]),
        "frequency_stop_hz": float(frequency_grid[-1]),
        "frequency_points": int(len(frequency_grid)),
        "geometry_groups": int(fingerprints.nunique()),
        "response_groups": int(response_ids.nunique()),
        "final_groups": int(final_group_ids.nunique()),
        "fold_count": int(folds_table["fold"].nunique()),
        "label_algorithm_version": config.label_algorithm_version,
    }
    audit_json_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_audit_markdown(audit_markdown_path, audit)
    return PreparationArtifacts(
        labels_csv=labels_path,
        spectra_npz=spectra_path,
        frequency_npy=frequency_path,
        folds_csv=folds_path,
        audit_json=audit_json_path,
        audit_markdown=audit_markdown_path,
    )


def _validate_preparation_directory(output_directory: Path) -> None:
    expected = {
        "labels.csv",
        "spectra.npz",
        "frequency.npy",
        "folds.csv",
        "audit.json",
        "audit.md",
    }
    actual = {path.name for path in output_directory.iterdir() if path.is_file()}
    if actual != expected:
        raise RuntimeError(
            f"prepared artifact set mismatch: expected {sorted(expected)}, got {sorted(actual)}"
        )
    if any((output_directory / name).stat().st_size == 0 for name in expected):
        raise RuntimeError("prepared artifact validation found an empty file")
    frequency = np.load(output_directory / "frequency.npy", allow_pickle=False)
    with np.load(output_directory / "spectra.npz", allow_pickle=False) as spectra:
        matrix = spectra["s11"]
        row_ids = spectra["training_row_id"]
    labels = pd.read_csv(output_directory / "labels.csv")
    folds = pd.read_csv(output_directory / "folds.csv")
    json.loads((output_directory / "audit.json").read_text(encoding="utf-8"))
    if matrix.shape != (len(labels), len(frequency)) or len(row_ids) != len(labels):
        raise RuntimeError("spectra artifact dimensions do not match labels/frequency")
    if len(folds) != len(labels):
        raise RuntimeError("fold artifact row count does not match labels")


def _atomic_publish_directory(source: Path, target: Path) -> None:
    """Atomically publish *source* without ever replacing *target* on Linux."""
    if sys.platform != "linux":
        raise RuntimeError("atomic no-replace publish requires Linux renameat2")
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (AttributeError, OSError) as exc:
        raise RuntimeError("libc renameat2 is unavailable; refusing unsafe fallback") from exc
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(target),
        rename_noreplace,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number, os.strerror(error_number), str(target)
        )
    if error_number in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
        raise RuntimeError(
            f"renameat2(RENAME_NOREPLACE) unavailable: {os.strerror(error_number)}"
        )
    raise OSError(error_number, os.strerror(error_number), str(target))


def prepare_dataset(
    workbook: str | Path,
    archive: str | Path,
    output_directory: str | Path,
    *,
    config: PreparationConfig | None = None,
) -> PreparationArtifacts:
    output_directory = Path(output_directory)
    if output_directory.exists():
        raise FileExistsError(f"prepared output already exists: {output_directory}")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.tmp-",
            dir=output_directory.parent,
        )
    )
    try:
        _prepare_dataset_in_directory(
            workbook,
            archive,
            temporary_directory,
            config=config,
        )
        _validate_preparation_directory(temporary_directory)
        if output_directory.exists():
            raise FileExistsError(f"prepared output appeared during build: {output_directory}")
        _atomic_publish_directory(temporary_directory, output_directory)
    except BaseException:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise
    return PreparationArtifacts(
        labels_csv=output_directory / "labels.csv",
        spectra_npz=output_directory / "spectra.npz",
        frequency_npy=output_directory / "frequency.npy",
        folds_csv=output_directory / "folds.csv",
        audit_json=output_directory / "audit.json",
        audit_markdown=output_directory / "audit.md",
    )
