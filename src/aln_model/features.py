from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd


PHYSICAL_COLUMNS = (
    "Train_A_res_um2",
    "Train_L_top_um",
    "Train_L_bot_um",
    "Train_L_air_um",
    "Train_h_FL_nm",
    "Train_L_FL_um",
    "Train_h_AG_nm",
    "Train_L_AG_um",
    "Train_h_EG_nm",
    "Train_L_EG_um",
    "Train_PF_nm",
)
STATE_COLUMNS = (
    "Template",
    "EG_State",
    "has_FL",
    "has_AG",
    "has_patterned_EG",
    "EG_material_present",
    "has_PF",
    "is_BASIC",
)
TOPOLOGIES = ("T11", "T10", "T01", "T00")


def _normalized_physical_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    normalized = rows.loc[:, PHYSICAL_COLUMNS].astype(float).copy()
    raw_pf = normalized["Train_PF_nm"]
    if "has_PF" in rows:
        has_pf = pd.to_numeric(rows["has_PF"], errors="coerce").fillna(0).ne(0)
    else:
        has_pf = raw_pf.fillna(0).ne(0)
    pf_missing = has_pf & raw_pf.isna()
    normalized["Train_PF_nm"] = raw_pf.where(has_pf, 0.0).fillna(0.0)
    return normalized, pf_missing.astype(float)


def _require_physical_columns(rows: pd.DataFrame) -> None:
    missing = set(PHYSICAL_COLUMNS).difference(rows.columns)
    if missing:
        raise ValueError(f"missing authoritative physical columns: {sorted(missing)}")
    missing_state = set(STATE_COLUMNS).difference(rows.columns)
    if missing_state:
        raise ValueError(f"missing required state columns: {sorted(missing_state)}")


def _canonical_value(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".12g")
    if isinstance(value, (int, np.integer, bool, np.bool_)):
        return int(value)
    return str(value).strip()


def geometry_fingerprint(rows: pd.DataFrame) -> pd.Series:
    _require_physical_columns(rows)
    normalized_physical, pf_missing = _normalized_physical_rows(rows)
    normalized = rows.copy()
    for column in PHYSICAL_COLUMNS:
        normalized[column] = pd.Series(
            normalized_physical[column].to_numpy(dtype=float),
            index=normalized.index,
            dtype=float,
        )
    normalized["pf_value_missing"] = pd.Series(
        pf_missing.to_numpy(dtype=float), index=normalized.index, dtype=float
    )
    columns = [
        column
        for column in (*STATE_COLUMNS, *PHYSICAL_COLUMNS, "pf_value_missing")
        if column in normalized
    ]

    def fingerprint(row: pd.Series) -> str:
        payload = {column: _canonical_value(row[column]) for column in columns}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    return normalized.apply(fingerprint, axis=1).rename("geometry_fingerprint")


def build_physics_features(rows: pd.DataFrame) -> pd.DataFrame:
    _require_physical_columns(rows)
    result, pf_missing = _normalized_physical_rows(rows)
    result["pf_value_missing"] = pf_missing
    for column in STATE_COLUMNS:
        if column not in {"Template", "EG_State"}:
            values = rows[column] if column in rows else pd.Series(0, index=rows.index)
            result[column] = pd.to_numeric(values, errors="coerce").fillna(0).astype(float)
    topology = rows["Template"].fillna("").astype(str).str.upper()
    for name in TOPOLOGIES:
        result[f"topology_{name}"] = topology.eq(name).astype(float)
    result["topology_unknown"] = (~topology.isin(TOPOLOGIES)).astype(float)
    eg_state = rows["EG_State"].fillna("").astype(str).str.lower()
    result["eg_patterned"] = eg_state.eq("patterned").astype(float)
    result["eg_blanket"] = eg_state.eq("blanket").astype(float)
    result["eg_unknown"] = (~eg_state.isin({"patterned", "blanket"})).astype(float)
    def add_safe_ratio(name: str, numerator: str, denominator: str) -> None:
        numerator_values = result[numerator].to_numpy(dtype=float)
        denominator_values = result[denominator].to_numpy(dtype=float)
        valid = (
            np.isfinite(numerator_values)
            & np.isfinite(denominator_values)
            & (denominator_values != 0)
        )
        values = np.zeros(len(result), dtype=float)
        np.divide(
            numerator_values,
            denominator_values,
            out=values,
            where=valid,
        )
        result[name] = values
        result[f"{name}_invalid"] = (~valid).astype(float)

    add_safe_ratio("air_to_bot_ratio", "Train_L_air_um", "Train_L_bot_um")
    add_safe_ratio("ag_to_top_ratio", "Train_L_AG_um", "Train_L_top_um")
    add_safe_ratio("fl_to_top_ratio", "Train_L_FL_um", "Train_L_top_um")
    add_safe_ratio("eg_to_top_ratio", "Train_L_EG_um", "Train_L_top_um")
    return result
