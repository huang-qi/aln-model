from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from aln_model.data import (
    load_training_index,
    load_training_index_with_audit,
    map_archive_members,
)


def test_load_training_index_keeps_matched_positive_ready_rows(tmp_path: Path):
    workbook = tmp_path / "index.xlsx"
    rows = pd.DataFrame(
        {
            "Training_Row_ID": ["keep-a", "keep-b", "cancel", "not-ready", "bogus"],
            "SNP_Status": ["MATCHED", "MATCHED", "CANCELLED", "MATCHED", "MATCHED"],
            "Training_Ready": [
                "YES",
                "YES_WITH_AUDIT_NOTE",
                "CANCELLED_DO_NOT_TRAIN",
                "NO",
                "YES_BUT_UNREVIEWED",
            ],
            "SNP_Relative_Path": [
                "D1\\a.s1p",
                "D2/b.s1p",
                None,
                "D1/c.s1p",
                "D1/d.s1p",
            ],
        }
    )
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        rows.to_excel(writer, sheet_name="Training_Index", index=False)

    selected = load_training_index(workbook)

    assert selected["Training_Row_ID"].tolist() == ["keep-a", "keep-b"]


def test_training_index_audit_counts_input_selected_and_excluded_states(tmp_path: Path):
    workbook = tmp_path / "index.xlsx"
    rows = pd.DataFrame(
        {
            "Training_Row_ID": ["a", "b", "c", "d"],
            "SNP_Status": ["MATCHED", "MATCHED", "CANCELLED", "MATCHED"],
            "Training_Ready": [
                "YES",
                "YES_WITH_USER_DECISION",
                "CANCELLED_DO_NOT_TRAIN",
                "YES_UNKNOWN",
            ],
            "SNP_Relative_Path": ["a.s1p", "b.s1p", None, "d.s1p"],
        }
    )
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        rows.to_excel(writer, sheet_name="Training_Index", index=False)

    selected, audit = load_training_index_with_audit(workbook)

    assert len(selected) == 2
    assert audit["input_rows"] == 4
    assert audit["selected_rows"] == 2
    assert audit["excluded_rows"] == 2
    assert audit["input_training_ready_counts"]["YES_UNKNOWN"] == 1
    assert audit["excluded_training_ready_counts"]["YES_UNKNOWN"] == 1


def test_map_archive_members_uses_relative_path_not_basename(tmp_path: Path):
    archive = tmp_path / "spectra.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr("root/D1/a.s1p", "first")
        zf.writestr("root/D2/a.s1p", "second")
    rows = pd.DataFrame(
        {"SNP_Relative_Path": ["D1\\a.s1p", "D2\\a.s1p"]},
        index=[10, 11],
    )

    mapped = map_archive_members(rows, archive)

    assert mapped["ZIP_Member"].tolist() == ["root/D1/a.s1p", "root/D2/a.s1p"]


def test_map_archive_members_reports_missing_relative_path(tmp_path: Path):
    archive = tmp_path / "spectra.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr("root/other/a.s1p", "data")
    rows = pd.DataFrame({"SNP_Relative_Path": ["D1/a.s1p"]})

    with pytest.raises(ValueError, match="D1/a.s1p"):
        map_archive_members(rows, archive)


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda frame: frame.assign(SNP_Status="CANCELLED"), "no training rows"),
        (
            lambda frame: pd.concat([frame, frame], ignore_index=True),
            "Training_Row_ID",
        ),
        (
            lambda frame: pd.concat(
                [frame, frame.assign(Training_Row_ID="different")], ignore_index=True
            ),
            "SNP_Relative_Path",
        ),
    ],
)
def test_training_index_rejects_empty_or_duplicate_selection(tmp_path, mutator, message):
    workbook = tmp_path / "invalid.xlsx"
    base = pd.DataFrame(
        {
            "Training_Row_ID": ["one"],
            "SNP_Status": ["MATCHED"],
            "Training_Ready": ["YES"],
            "SNP_Relative_Path": ["D1\\one.s1p"],
        }
    )
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        mutator(base).to_excel(writer, sheet_name="Training_Index", index=False)

    with pytest.raises(ValueError, match=message):
        load_training_index(workbook)


@pytest.mark.parametrize("invalid_id", [None, np.nan, "", "   "])
def test_training_index_rejects_missing_or_blank_training_id(tmp_path, invalid_id):
    workbook = tmp_path / "blank-id.xlsx"
    rows = pd.DataFrame(
        {
            "Training_Row_ID": [invalid_id],
            "SNP_Status": ["MATCHED"],
            "Training_Ready": ["YES"],
            "SNP_Relative_Path": ["D1/one.s1p"],
        }
    )
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        rows.to_excel(writer, sheet_name="Training_Index", index=False)

    with pytest.raises(ValueError, match="Training_Row_ID"):
        load_training_index(workbook)


def test_training_index_rejects_mixed_type_ids_with_same_canonical_value(monkeypatch):
    rows = pd.DataFrame(
        {
            "Training_Row_ID": [1, "1"],
            "SNP_Status": ["MATCHED", "MATCHED"],
            "Training_Ready": ["YES", "YES"],
            "SNP_Relative_Path": ["D1/one.s1p", "D1/two.s1p"],
        }
    )
    monkeypatch.setattr("aln_model.data.pd.read_excel", lambda *args, **kwargs: rows)

    with pytest.raises(ValueError, match="canonical Training_Row_ID"):
        load_training_index("mixed-id.xlsx")


def test_training_index_replaces_selected_id_with_trimmed_canonical_string(monkeypatch):
    rows = pd.DataFrame(
        {
            "Training_Row_ID": ["  id-1  "],
            "SNP_Status": ["MATCHED"],
            "Training_Ready": ["YES"],
            "SNP_Relative_Path": ["D1/one.s1p"],
        }
    )
    monkeypatch.setattr("aln_model.data.pd.read_excel", lambda *args, **kwargs: rows)

    selected = load_training_index("canonical-id.xlsx")

    assert selected["Training_Row_ID"].tolist() == ["id-1"]
