from __future__ import annotations

import hashlib
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd


def response_quantized_hash(
    spectra: np.ndarray,
    *,
    tolerance: float = 1e-6,
) -> pd.Series:
    """Return a quantized audit hash; this does not guarantee near-neighbour grouping."""
    spectra = np.asarray(spectra, dtype=complex)
    if spectra.ndim != 2 or spectra.shape[1] == 0:
        raise ValueError("spectra must be a non-empty two-dimensional matrix")
    if tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError("response hash tolerance must be positive and finite")
    if not np.all(np.isfinite(spectra)):
        raise ValueError("spectra must be finite")
    hashes: list[str] = []
    for spectrum in spectra:
        quantized = np.rint(
            np.column_stack((spectrum.real, spectrum.imag)) / tolerance
        ).astype("<i8", copy=False)
        hashes.append(hashlib.sha256(quantized.tobytes()).hexdigest())
    return pd.Series(hashes, name="response_quantized_hash")


def response_near_duplicate_hash(
    spectra: np.ndarray,
    *,
    tolerance: float = 1e-6,
) -> pd.Series:
    """Deprecated audit hash alias; it does not identify all near neighbours."""
    warnings.warn(
        "response_near_duplicate_hash is only a quantized audit hash; "
        "use response_near_duplicate_clusters for fold grouping",
        DeprecationWarning,
        stacklevel=2,
    )
    return response_quantized_hash(spectra, tolerance=tolerance)


def response_near_duplicate_clusters(
    spectra: np.ndarray,
    *,
    tolerance: float = 1e-6,
) -> pd.Series:
    spectra = np.asarray(spectra, dtype=complex)
    if spectra.ndim != 2 or spectra.shape[0] == 0 or spectra.shape[1] == 0:
        raise ValueError("spectra must be a non-empty two-dimensional matrix")
    if tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError("response cluster tolerance must be positive and finite")
    if not np.all(np.isfinite(spectra)):
        raise ValueError("spectra must be finite")
    vectors = np.concatenate((spectra.real, spectra.imag), axis=1)
    sort_coordinate = int(np.argmax(np.var(vectors, axis=0)))
    order = np.argsort(vectors[:, sort_coordinate], kind="stable")
    parent = list(range(len(vectors)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    window_start = 0
    for position, row_index in enumerate(order):
        coordinate = vectors[row_index, sort_coordinate]
        while (
            window_start < position
            and coordinate - vectors[order[window_start], sort_coordinate] > tolerance
        ):
            window_start += 1
        for candidate_position in range(window_start, position):
            candidate_index = int(order[candidate_position])
            if np.max(np.abs(spectra[row_index] - spectra[candidate_index])) <= tolerance:
                union(int(row_index), candidate_index)

    raw_hashes = [hashlib.sha256(vector.astype("<f8").tobytes()).hexdigest() for vector in vectors]
    components: dict[int, list[int]] = {}
    for index in range(len(vectors)):
        components.setdefault(find(index), []).append(index)
    cluster_ids = {
        root: hashlib.sha256(
            "|".join(sorted(raw_hashes[index] for index in members)).encode()
        ).hexdigest()
        for root, members in components.items()
    }
    return pd.Series(
        [cluster_ids[find(index)] for index in range(len(vectors))],
        name="response_cluster_id",
    )


def connected_group_ids(
    geometry_fingerprints: pd.Series,
    response_cluster_ids: pd.Series,
) -> pd.Series:
    geometry = pd.Series(geometry_fingerprints).astype(str).reset_index(drop=True)
    response = pd.Series(response_cluster_ids).astype(str).reset_index(drop=True)
    if len(geometry) != len(response):
        raise ValueError("geometry and response group arrays must have equal length")
    parent = list(range(len(geometry)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for values in (geometry, response):
        first_seen: dict[str, int] = {}
        for index, value in enumerate(values):
            if value in first_seen:
                union(first_seen[value], index)
            else:
                first_seen[value] = index
    components: dict[int, list[int]] = {}
    for index in range(len(parent)):
        components.setdefault(find(index), []).append(index)
    component_ids: dict[int, str] = {}
    for root, members in components.items():
        signature = "|".join(
            sorted(
                {f"g:{geometry.iloc[i]}" for i in members}
                | {f"r:{response.iloc[i]}" for i in members}
            )
        )
        component_ids[root] = hashlib.sha256(signature.encode()).hexdigest()
    result = [component_ids[find(index)] for index in range(len(parent))]
    return pd.Series(result, index=pd.Series(geometry_fingerprints).index, name="final_group_id")


def assign_group_folds(
    rows: pd.DataFrame,
    *,
    n_splits: int = 5,
    random_state: int = 20260818,
    balance_columns: Sequence[str] | None = None,
) -> pd.Series:
    balance_columns = tuple(dict.fromkeys(balance_columns or ()))
    group_column = (
        "final_group_id" if "final_group_id" in rows.columns else "geometry_fingerprint"
    )
    required = {group_column, "Template", "EG_State", *balance_columns}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"fold input missing columns: {sorted(missing)}")
    groups = rows[group_column].astype(str)
    if groups.nunique() < n_splits:
        raise ValueError(f"need at least {n_splits} distinct geometry groups")
    stratified_rows = rows.assign(
        _group=groups,
        _stratum=(
            rows["Template"].fillna("UNKNOWN").astype(str)
            + "|"
            + rows["EG_State"].fillna("UNKNOWN").astype(str)
        ),
    )
    group_sizes = stratified_rows.groupby("_group", sort=True).size()
    group_strata = pd.crosstab(
        stratified_rows["_group"], stratified_rows["_stratum"]
    ).reindex(group_sizes.index, fill_value=0)
    group_balance = pd.DataFrame(
        {
            column: rows[column].fillna(False).astype(bool).groupby(groups).any()
            for column in balance_columns
        },
        index=group_sizes.index,
    )
    tie_keys = {
        group: hashlib.sha256(f"{random_state}|{group}".encode()).hexdigest()
        for group in group_sizes.index
    }
    ordered_groups = sorted(
        group_sizes.index,
        key=lambda group: (
            -int(group_balance.loc[group].sum()),
            -int(group_sizes[group]),
            tie_keys[group],
        ),
    )
    fold_load = np.zeros(n_splits, dtype=int)
    fold_group_count = np.zeros(n_splits, dtype=int)
    stratum_load = np.zeros((n_splits, len(group_strata.columns)), dtype=int)
    balance_load = np.zeros((n_splits, len(balance_columns)), dtype=int)
    stratum_totals = group_strata.sum(axis=0).to_numpy(dtype=float)
    stratum_targets = np.maximum(stratum_totals / n_splits, 1.0)
    group_to_fold: dict[str, int] = {}
    for group in ordered_groups:
        group_counts = group_strata.loc[group].to_numpy(dtype=int)
        group_balance_counts = group_balance.loc[group].to_numpy(dtype=int)
        scores = []
        for fold in range(n_splits):
            balance_penalty = int(
                np.sum(balance_load[fold] * group_balance_counts)
            )
            stratum_penalty = float(
                np.sum((stratum_load[fold] + group_counts) / stratum_targets)
            )
            scores.append(
                (
                    balance_penalty,
                    int(fold_load[fold]),
                    stratum_penalty,
                    int(fold_group_count[fold]),
                    fold,
                )
            )
        selected_fold = min(scores)[-1]
        group_to_fold[str(group)] = selected_fold
        fold_load[selected_fold] += int(group_sizes[group])
        fold_group_count[selected_fold] += 1
        stratum_load[selected_fold] += group_counts
        balance_load[selected_fold] += group_balance_counts
    folds = groups.map(group_to_fold).to_numpy(dtype=int)
    if np.any(folds < 0):
        raise RuntimeError("not every row received a validation fold")
    result = pd.Series(folds, index=rows.index, name="fold")
    if result.groupby(groups).nunique().gt(1).any():
        raise RuntimeError("geometry fingerprint leaked across folds")
    return result
