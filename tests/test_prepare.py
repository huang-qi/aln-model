import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from aln_model.cli import main
from aln_model.prepare import prepare_dataset
import aln_model.prepare as prepare_module


def _bvd_touchstone(fs_hz: float) -> str:
    frequency = np.linspace(4.2e9, 5.9e9, 341)
    omega = 2 * np.pi * frequency
    cm = 5e-15
    c0 = 100e-15
    lm = 1 / ((2 * np.pi * fs_hz) ** 2 * cm)
    rm = 2 * np.pi * fs_hz * lm / 350
    admittance = 1j * omega * c0 + 1 / (
        rm + 1j * omega * lm + 1 / (1j * omega * cm)
    )
    impedance = 1 / admittance
    s11 = (impedance - 50) / (impedance + 50)
    lines = ["! miniature analytic spectrum", "# Hz S MA R 50"]
    for f, value in zip(frequency, s11):
        lines.append(f"{f:.0f} {abs(value):.16g} {np.angle(value, deg=True):.16g}")
    return "\n".join(lines) + "\n"


def _make_miniature_inputs(tmp_path: Path) -> tuple[Path, Path]:
    workbook = tmp_path / "training.xlsx"
    archive = tmp_path / "spectra.zip"
    rows = []
    with ZipFile(archive, "w") as zf:
        for index in range(10):
            relative_path = f"D1\\group-{index}\\response.s1p"
            rows.append(
                {
                    "Training_Row_ID": (
                        f" TR-{index:04d} " if index == 0 else f"TR-{index:04d}"
                    ),
                    "SNP_Status": "MATCHED",
                    "Training_Ready": "YES",
                    "SNP_Relative_Path": relative_path,
                    "SNP_File": f"response-{index}.s1p",
                    "SNP_A_res_um2": 9000.0 + index,
                    "SNP_h_AG_nm": 81.0,
                    "Template": ["T11", "T10", "T01", "T00"][index % 4],
                    "EG_State": ["patterned", "blanket"][index % 2],
                    "Train_A_res_um2": 1000.0 + index * 100,
                    "Train_L_top_um": 60.0,
                    "Train_L_bot_um": 56.0,
                    "Train_L_air_um": 50.0,
                    "Train_h_FL_nm": 70.0,
                    "Train_L_FL_um": 1.5,
                    "Train_h_AG_nm": 80.0,
                    "Train_L_AG_um": 1.25,
                    "Train_h_EG_nm": 3.0,
                    "Train_L_EG_um": 0.75,
                    "Train_PF_nm": np.nan if index == 8 else 0.0,
                    "has_FL": 1,
                    "has_AG": 1,
                    "has_patterned_EG": int(index % 2 == 0),
                    "EG_material_present": 1,
                    "has_PF": 1 if index == 8 else 0,
                    "is_BASIC": 0,
                    "Training_Ready": "YES" if index < 9 else "YES_WITH_AUDIT_NOTE",
                    "Cross_Source_Status": "AGREED",
                    "Diff_Fields": "",
                    "Match_Method": "EXACT",
                    "Match_Confidence": "HIGH",
                    "Notes_and_Normalization": f"note-{index}",
                }
            )
            member = "archive-root/" + relative_path.replace("\\", "/")
            fs = 4.65e9 if index in {0, 1} else 4.65e9 + index * 5e6
            if index == 9:
                fs = 5.55e9
            zf.writestr(member, _bvd_touchstone(fs))
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Training_Index", index=False)
    return workbook, archive


def test_prepare_cli_writes_auditable_artifact_contract(tmp_path: Path, monkeypatch):
    workbook, archive = _make_miniature_inputs(tmp_path)
    output = tmp_path / "prepared"
    fold_call = {}
    original_assign_group_folds = prepare_module.assign_group_folds

    def capture_fold_input(rows, **kwargs):
        fold_call["rows"] = rows.copy()
        fold_call["kwargs"] = kwargs.copy()
        return original_assign_group_folds(rows, **kwargs)

    monkeypatch.setattr(prepare_module, "assign_group_folds", capture_fold_input)

    exit_code = main(
        [
            "prepare",
            "--workbook",
            str(workbook),
            "--archive",
            str(archive),
            "--output",
            str(output),
            "--frequency-step-hz",
            "5000000",
        ]
    )

    assert exit_code == 0
    expected = {
        "labels.csv",
        "spectra.npz",
        "frequency.npy",
        "folds.csv",
        "audit.json",
        "audit.md",
    }
    assert expected == {path.name for path in output.iterdir()}
    labels = pd.read_csv(output / "labels.csv")
    folds = pd.read_csv(output / "folds.csv")
    frequency = np.load(output / "frequency.npy")
    with np.load(output / "spectra.npz") as spectra:
        assert spectra["s11"].shape == (10, 239)
        assert spectra["training_row_id"].tolist() == labels["Training_Row_ID"].tolist()
    assert folds["Training_Row_ID"].tolist() == labels["Training_Row_ID"].tolist()
    assert labels.loc[0, "Training_Row_ID"] == "TR-0000"
    audit = json.loads((output / "audit.json").read_text())

    assert frequency[[0, -1]].tolist() == [4.2e9, 5.39e9]
    assert {"fs", "fp", "k_eff2", "Qs", "Qp", "spurious"}.issubset(labels)
    assert {"fs_hz", "fp_hz", "qs", "qp"}.issubset(labels)
    assert {
        "spurious_dangerous",
        "quality_valid",
        "quality_boundary",
        "quality_ambiguous",
        "quality_grid_ok",
        "quality_reason",
        "label_algorithm_version",
    }.issubset(labels)
    assert labels["quality_valid"].all()
    assert labels["fs"].max() > 5.39e9
    outside = labels.loc[labels["fs"].idxmax()]
    assert bool(outside["fs_outside_shared_band"])
    assert bool(outside["fp_outside_shared_band"])
    assert labels["label_algorithm_version"].nunique() == 1
    assert labels["label_algorithm_version"].iloc[0] == audit["label_algorithm_version"]
    assert fold_call["kwargs"]["balance_columns"] == ["spurious_dangerous"]
    assert fold_call["rows"]["spurious_dangerous"].tolist() == labels[
        "spurious_dangerous"
    ].tolist()
    assert {
        "Training_Ready",
        "Cross_Source_Status",
        "Diff_Fields",
        "Match_Method",
        "Match_Confidence",
        "Notes_and_Normalization",
        "has_FL",
        "has_AG",
        "has_patterned_EG",
        "EG_material_present",
        "has_PF",
        "is_BASIC",
    }.issubset(labels)
    assert {"SNP_Status", "SNP_File", "SNP_A_res_um2", "SNP_h_AG_nm"}.issubset(labels)
    assert labels.loc[8, "SNP_File"] == "response-8.s1p"
    assert bool(labels.loc[8, "pf_value_missing"])
    assert not bool(labels.loc[0, "pf_value_missing"])
    assert {
        "response_cluster_id",
        "final_group_id",
        "group_member_count",
        "group_geometry_count",
        "group_response_count",
    }.issubset(folds)
    assert folds.loc[0, "response_cluster_id"] == folds.loc[1, "response_cluster_id"]
    assert folds.loc[0, "final_group_id"] == folds.loc[1, "final_group_id"]
    assert folds.loc[0, "fold"] == folds.loc[1, "fold"]
    assert folds.groupby("geometry_fingerprint")["fold"].nunique().eq(1).all()
    assert sorted(folds["fold"].unique()) == [0, 1, 2, 3, 4]
    assert audit["selected_rows"] == 10
    assert audit["valid_labels"] == 10
    assert audit["input_rows"] == 10
    assert audit["selected_training_ready_counts"]["YES_WITH_AUDIT_NOTE"] == 1
    assert audit["boundary_labels"] == 0
    assert audit["ambiguity_labels"] >= 0
    assert audit["spurious_labels"] >= 0
    assert audit["response_groups"] == 9
    assert audit["final_groups"] <= audit["geometry_groups"]
    assert audit["fs_outside_shared_band_count"] == 1
    assert audit["fp_outside_shared_band_count"] == 1
    assert audit["pf_value_missing_count"] == 1


def test_prepare_rejects_existing_output_directory(tmp_path: Path):
    workbook, archive = _make_miniature_inputs(tmp_path)
    output = tmp_path / "already-exists"
    output.mkdir()

    with pytest.raises(FileExistsError):
        prepare_dataset(workbook, archive, output)


def test_prepare_failure_leaves_no_partial_or_temporary_directory(tmp_path, monkeypatch):
    workbook, archive = _make_miniature_inputs(tmp_path)
    output = tmp_path / "prepared-failure"
    monkeypatch.setattr(
        prepare_module,
        "parse_touchstone_s11",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("injected failure")),
    )

    with pytest.raises(ValueError, match="injected failure"):
        prepare_dataset(workbook, archive, output)

    assert not output.exists()
    assert not list(tmp_path.glob(".prepared-failure.tmp-*"))


def test_atomic_publish_never_replaces_existing_empty_directory(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "marker").write_text("source")

    with pytest.raises(FileExistsError):
        prepare_module._atomic_publish_directory(source, target)

    assert source.exists()
    assert target.exists()
    assert not (target / "marker").exists()


def test_atomic_publish_reports_missing_renameat2_without_fallback(tmp_path, monkeypatch):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    monkeypatch.setattr(prepare_module.ctypes, "CDLL", lambda *args, **kwargs: object())

    with pytest.raises(RuntimeError, match="renameat2"):
        prepare_module._atomic_publish_directory(source, target)

    assert source.exists()
    assert not target.exists()
