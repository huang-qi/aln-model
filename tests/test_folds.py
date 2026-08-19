import numpy as np
import pandas as pd

from aln_model.folds import (
    assign_group_folds,
    connected_group_ids,
    response_quantized_hash,
    response_near_duplicate_clusters,
)


def test_assign_group_folds_is_deterministic_and_isolates_fingerprints():
    rows = pd.DataFrame(
        {
            "geometry_fingerprint": [f"g{i // 2}" for i in range(20)],
            "Template": ["T11", "T10", "T01", "T00"] * 5,
            "EG_State": ["patterned", "blanket"] * 10,
        }
    )

    first = assign_group_folds(rows, n_splits=5, random_state=17)
    second = assign_group_folds(rows, n_splits=5, random_state=17)

    assert first.tolist() == second.tolist()
    assert sorted(first.unique()) == [0, 1, 2, 3, 4]
    assert first.groupby(rows["geometry_fingerprint"]).nunique().eq(1).all()


def test_assign_group_folds_rejects_too_few_groups():
    rows = pd.DataFrame(
        {
            "geometry_fingerprint": ["same"] * 5,
            "Template": ["T11"] * 5,
            "EG_State": ["patterned"] * 5,
        }
    )

    try:
        assign_group_folds(rows, n_splits=5)
    except ValueError as exc:
        assert "distinct geometry groups" in str(exc)
    else:
        raise AssertionError("expected insufficient-group validation")


def test_response_hash_and_connected_groups_prevent_transitive_leakage():
    spectra = np.array(
        [
            [0.1 + 0.2j, 0.2 + 0.3j],
            [0.1 + 0.2j + 1e-8, 0.2 + 0.3j],
            [0.8 + 0.1j, 0.7 + 0.2j],
        ]
    )
    response_ids = response_quantized_hash(spectra, tolerance=1e-6)
    final_ids = connected_group_ids(
        pd.Series(["geometry-a", "geometry-b", "geometry-b"]), response_ids
    )

    assert response_ids.iloc[0] == response_ids.iloc[1]
    assert response_ids.iloc[0] != response_ids.iloc[2]
    assert final_ids.nunique() == 1


def test_actual_distance_clusters_neighbours_across_quantization_boundary():
    spectra = np.array([[0.49e-6 + 0j], [0.51e-6 + 0j], [3e-6 + 0j]])

    hashes = response_quantized_hash(spectra, tolerance=1e-6)
    clusters = response_near_duplicate_clusters(spectra, tolerance=1e-6)

    assert hashes.iloc[0] != hashes.iloc[1]
    assert clusters.iloc[0] == clusters.iloc[1]
    assert clusters.iloc[0] != clusters.iloc[2]


def test_actual_distance_uses_complex_modulus_not_componentwise_distance():
    spectra = np.array([[0 + 0j], [0.9e-6 + 0.9e-6j]])

    clusters = response_near_duplicate_clusters(spectra, tolerance=1e-6)

    assert clusters.iloc[0] != clusters.iloc[1]


def test_fold_greedy_balances_sample_counts_with_one_large_group():
    rows = pd.DataFrame(
        {
            "final_group_id": ["large"] * 100 + [f"small-{i}" for i in range(9)],
            "geometry_fingerprint": ["large"] * 100
            + [f"small-{i}" for i in range(9)],
            "Template": ["T11"] * 109,
            "EG_State": ["patterned"] * 109,
        }
    )

    folds = assign_group_folds(rows, n_splits=5, random_state=17)

    assert sorted(folds.value_counts().tolist()) == [2, 2, 2, 3, 100]
    assert folds.groupby(rows["final_group_id"]).nunique().eq(1).all()


def test_assign_group_folds_spreads_rare_positive_groups_across_every_fold():
    rows = pd.DataFrame(
        {
            "final_group_id": [f"positive-{i}" for i in range(7)]
            + [f"negative-{i}" for i in range(8)],
            "geometry_fingerprint": [f"geometry-{i}" for i in range(15)],
            "Template": ["T11", "T10", "T01", "T00", "T11"] * 3,
            "EG_State": ["patterned", "blanket", "patterned"] * 5,
            "spurious_dangerous": [True] * 7 + [False] * 8,
        }
    )

    first = assign_group_folds(
        rows,
        n_splits=5,
        random_state=17,
        balance_columns=["spurious_dangerous"],
    )
    second = assign_group_folds(
        rows,
        n_splits=5,
        random_state=17,
        balance_columns=["spurious_dangerous"],
    )

    positives_per_fold = rows["spurious_dangerous"].groupby(first).sum()
    assert positives_per_fold.reindex(range(5), fill_value=0).ge(1).all()
    assert first.groupby(rows["final_group_id"]).nunique().eq(1).all()
    assert first.tolist() == second.tolist()


def test_rare_balancing_keeps_unavoidable_large_group_load_reasonable():
    rows = pd.DataFrame(
        {
            "final_group_id": ["large"] * 100
            + [f"positive-{i}" for i in range(5)]
            + [f"negative-{i}" for i in range(5)],
            "geometry_fingerprint": ["large"] * 100
            + [f"positive-{i}" for i in range(5)]
            + [f"negative-{i}" for i in range(5)],
            "Template": ["T11"] * 110,
            "EG_State": ["patterned"] * 110,
            "spurious_dangerous": [False] * 100 + [True] * 5 + [False] * 5,
        }
    )

    folds = assign_group_folds(
        rows,
        n_splits=5,
        random_state=17,
        balance_columns=["spurious_dangerous"],
    )

    loads = sorted(folds.value_counts().tolist())
    assert loads[-1] == 101
    assert loads[-2] - loads[0] <= 1
    assert rows["spurious_dangerous"].groupby(folds).sum().ge(1).all()
