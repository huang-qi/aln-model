from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import pandas as pd


REQUIRED_INDEX_COLUMNS = {
    "Training_Row_ID",
    "SNP_Status",
    "Training_Ready",
    "SNP_Relative_Path",
}
TRAINING_READY_ALLOWLIST = {
    "YES",
    "YES_WITH_AUDIT_NOTE",
    "YES_WITH_USER_DECISION",
}


def _status_counts(values: pd.Series) -> dict[str, int]:
    normalized = values.fillna("<MISSING>").astype(str)
    return {str(key): int(value) for key, value in normalized.value_counts().items()}


def _read_training_index(workbook: str | Path) -> pd.DataFrame:
    rows = pd.read_excel(workbook, sheet_name="Training_Index")
    missing = REQUIRED_INDEX_COLUMNS.difference(rows.columns)
    if missing:
        raise ValueError(f"Training_Index missing columns: {sorted(missing)}")
    return rows


def load_training_index(workbook: str | Path) -> pd.DataFrame:
    rows = _read_training_index(workbook)
    ready = rows["Training_Ready"].isin(TRAINING_READY_ALLOWLIST)
    matched = rows["SNP_Status"].eq("MATCHED")
    selected = rows.loc[matched & ready].copy()
    _validate_selected_rows(selected)
    return selected.reset_index(drop=True)


def load_training_index_with_audit(
    workbook: str | Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = _read_training_index(workbook)
    selected_mask = rows["SNP_Status"].eq("MATCHED") & rows["Training_Ready"].isin(
        TRAINING_READY_ALLOWLIST
    )
    selected = rows.loc[selected_mask].copy()
    _validate_selected_rows(selected)
    excluded = rows.loc[~selected_mask]
    audit = {
        "input_rows": int(len(rows)),
        "selected_rows": int(selected_mask.sum()),
        "excluded_rows": int((~selected_mask).sum()),
        "input_snp_status_counts": _status_counts(rows["SNP_Status"]),
        "selected_snp_status_counts": _status_counts(selected["SNP_Status"]),
        "excluded_snp_status_counts": _status_counts(excluded["SNP_Status"]),
        "input_training_ready_counts": _status_counts(rows["Training_Ready"]),
        "selected_training_ready_counts": _status_counts(selected["Training_Ready"]),
        "excluded_training_ready_counts": _status_counts(excluded["Training_Ready"]),
    }
    return selected.reset_index(drop=True), audit


def _normalize_relative_path(value: object) -> str:
    path = str(value).replace("\\", "/").lstrip("/")
    parts = PurePosixPath(path).parts
    if not parts or ".." in parts:
        raise ValueError(f"invalid SNP_Relative_Path: {value}")
    return "/".join(parts)


def _validate_selected_rows(selected: pd.DataFrame) -> None:
    if selected.empty:
        raise ValueError("no training rows selected by matched/ready contract")
    if selected["SNP_Relative_Path"].isna().any():
        raise ValueError("matched training row has no SNP_Relative_Path")
    training_ids = selected["Training_Row_ID"]
    null_ids = training_ids.isna()
    canonical_ids = training_ids.fillna("").astype(str).str.strip()
    invalid_ids = null_ids | canonical_ids.eq("")
    if invalid_ids.any():
        raise ValueError("Training_Row_ID must be non-null and non-blank")
    duplicate_ids = canonical_ids.duplicated(keep=False)
    if duplicate_ids.any():
        values = sorted(canonical_ids.loc[duplicate_ids].unique())
        raise ValueError(f"duplicate canonical Training_Row_ID values: {values}")
    selected["Training_Row_ID"] = pd.Series(
        canonical_ids.to_list(), index=selected.index, dtype=object
    )
    normalized_paths = selected["SNP_Relative_Path"].map(_normalize_relative_path)
    duplicate_paths = normalized_paths.duplicated(keep=False)
    if duplicate_paths.any():
        values = sorted(normalized_paths.loc[duplicate_paths].unique())
        raise ValueError(f"duplicate SNP_Relative_Path values: {values}")


def map_archive_members(rows: pd.DataFrame, archive: str | Path) -> pd.DataFrame:
    with ZipFile(archive) as zf:
        members = [name for name in zf.namelist() if not name.endswith("/")]
    exact_index: dict[str, list[str]] = {}
    suffix_index: dict[str, list[str]] = {}
    for member in members:
        normalized_member = _normalize_relative_path(member)
        exact_index.setdefault(normalized_member, []).append(member)
        parts = PurePosixPath(normalized_member).parts
        for start in range(len(parts)):
            suffix = "/".join(parts[start:])
            suffix_index.setdefault(suffix, []).append(member)
    mapped: list[str] = []
    for raw_path in rows["SNP_Relative_Path"]:
        relative = _normalize_relative_path(raw_path)
        matches = exact_index.get(relative) or suffix_index.get(relative, [])
        if len(matches) != 1:
            raise ValueError(
                f"archive mapping for {relative!r} expected one member, found {len(matches)}"
            )
        mapped.append(matches[0])
    result = rows.copy()
    result["ZIP_Member"] = mapped
    return result
