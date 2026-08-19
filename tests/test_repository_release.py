import hashlib
import json
import re
import subprocess
import tomllib
from pathlib import Path

import pandas as pd
import yaml

from aln_model.features import PHYSICAL_COLUMNS, STATE_COLUMNS, TOPOLOGIES


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tracked_paths() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    return [
        PROJECT_ROOT / path.decode("utf-8", errors="surrogateescape")
        for path in result.stdout.split(b"\0")
        if path
    ]


def _is_ignored(path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", "--", path],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode in {0, 1}, result.stderr
    return result.returncode == 0


def test_gitignore_protects_private_inputs_and_generated_outputs() -> None:
    gitignore = PROJECT_ROOT / ".gitignore"

    assert gitignore.is_file(), ".gitignore must define the repository data boundary"

    ignored_paths = {
        "private/archive.zip",
        "private/archive.ZIP",
        "private/archive.tar",
        "private/archive.TAR",
        "private/archive.tar.gz",
        "private/archive.TAR.GZ",
        "private/archive.tgz",
        "private/archive.TGZ",
        "private/archive.7z",
        "private/archive.7Z",
        "private/archive.rar",
        "private/archive.RAR",
        "private/payload.gz",
        "private/payload.GZ",
        "private/payload.bz2",
        "private/payload.BZ2",
        "private/payload.xz",
        "private/payload.XZ",
        "private/archive.tar.bz2",
        "private/archive.TAR.BZ2",
        "private/archive.tar.xz",
        "private/archive.TAR.XZ",
        "private/workbook.xls",
        "private/workbook.XLS",
        "private/workbook.xlsx",
        "private/workbook.XLSX",
        "private/workbook.xlsm",
        "private/workbook.XLSM",
        "private/workbook.xlsb",
        "private/workbook.XLSB",
        "private/device.s1p",
        "private/device.s2p",
        "private/device.s10p",
        "private/device.S1P",
        "private/device.S2P",
        "private/device.S10P",
        "private/device.snp",
        "private/device.SNP",
        "artifacts/run/metrics.json",
        "reports/internal/report.md",
        ".venv/lib/python/site.py",
        "venv/lib/python/site.py",
        "env/lib/python/site.py",
        "src/__pycache__/module.pyc",
        ".pytest_cache/v/cache/nodeids",
        ".ruff_cache/content",
        "trained/model.joblib",
        "trained/model.JOBLIB",
        "trained/model.pkl",
        "trained/model.PKL",
        "trained/model.pickle",
        "trained/model.PICKLE",
        "trained/model.onnx",
        "trained/model.ONNX",
        "generated/array.npy",
        "generated/array.NPY",
        "generated/array.npz",
        "generated/array.NPZ",
        "private/design_technical_report.pdf",
        "private/design_TECHNICAL_REPORT.PDF",
        "private/器件技术报告-final.pdf",
        "private/notes.doc",
        "private/notes.DOC",
        "private/notes.docx",
        "private/notes.DOCX",
        "internal-process-report.md",
        "release-review-notes.md",
        "tests/fixtures/archive.zip",
        "tests/fixtures/archive.tar.gz",
        "tests/fixtures/archive.rar",
        "tests/fixtures/workbook.xlsx",
        "tests/fixtures/device.s2p",
        "tests/fixtures/model.joblib",
        "tests/fixtures/array.npz",
        "tests/fixtures/report.docx",
        "predictions/result.csv",
        "generated_predictions/result.csv",
        "src/aln_model.egg-info/PKG-INFO",
        "build/lib/aln_model/__init__.py",
        "dist/aln_model-0.1.0.whl",
    }
    allowed_root_documents = {
        "README.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "CODE_OF_CONDUCT.md",
    }

    unexpectedly_allowed = sorted(path for path in ignored_paths if not _is_ignored(path))
    unexpectedly_ignored = sorted(
        path for path in allowed_root_documents if _is_ignored(path)
    )

    assert not unexpectedly_allowed, (
        f"private/generated paths are allowed: {unexpectedly_allowed}"
    )
    assert not unexpectedly_ignored, (
        f"approved root documents are ignored: {unexpectedly_ignored}"
    )


def test_tracked_release_files_exclude_private_data_artifacts_and_secrets() -> None:
    tracked_paths = _tracked_paths()
    approved_root_files = {
        ".gitattributes",
        ".gitignore",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "MANIFEST.in",
        "README.md",
        "SECURITY.md",
        "pyproject.toml",
        "uv.lock",
    }
    approved_exact_paths = {
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/feature_request.yml",
        ".github/workflows/ci.yml",
        "docs/MODEL_CARD.md",
        "examples/geometry.csv",
    }
    forbidden_suffixes = {
        ".7z",
        ".bz2",
        ".csv",
        ".doc",
        ".docx",
        ".gz",
        ".joblib",
        ".json",
        ".md",
        ".npy",
        ".npz",
        ".onnx",
        ".pdf",
        ".pickle",
        ".pkl",
        ".rar",
        ".snp",
        ".tar",
        ".tgz",
        ".xls",
        ".xlsb",
        ".xlsm",
        ".xlsx",
        ".xz",
        ".zip",
    }
    approved_suffix_exceptions = {
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "README.md",
        "SECURITY.md",
        "docs/MODEL_CARD.md",
        "examples/geometry.csv",
    }

    forbidden_files: list[str] = []
    for path in tracked_paths:
        relative = path.relative_to(PROJECT_ROOT)
        relative_name = relative.as_posix()
        lower_name = path.name.lower()
        is_approved_source = (
            len(relative.parts) >= 3
            and relative.parts[:2] == ("src", "aln_model")
            and path.suffix == ".py"
        )
        is_approved_test = (
            len(relative.parts) == 2
            and relative.parts[0] == "tests"
            and path.name.startswith("test_")
            and path.suffix == ".py"
        )
        is_approved_path = (
            relative_name in approved_root_files
            or relative_name in approved_exact_paths
            or is_approved_source
            or is_approved_test
        )
        has_forbidden_suffix = (
            path.suffix.lower() in forbidden_suffixes
            or re.search(r"\.s\d+p$", lower_name) is not None
        ) and relative_name not in approved_suffix_exceptions
        is_forbidden_report = (
            "_technical_report." in lower_name or "技术报告" in path.name
        )
        if not is_approved_path or has_forbidden_suffix or is_forbidden_report:
            forbidden_files.append(relative_name)

    assert not forbidden_files, f"unapproved tracked release files: {forbidden_files}"

    sensitive_patterns = {
        "private key": re.compile(
            "-----BEGIN (?:RSA |EC |OPENSSH )?" + "PRIVATE KEY-----"
        ),
        "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
        "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
        "GitHub fine-grained token": re.compile(
            r"\bgithub_" + r"pat_[A-Za-z0-9_]{20,}\b"
        ),
        "GitLab token": re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),
        "Google API key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
        "OpenAI API key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
        "Slack token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
        "assigned credential": re.compile(
            r"(?i)\b(?:api[_-]?key|access[_-]?token|password|secret)\b"
            r"\s*[:=]\s*['\"][^'\"\s]{8,}['\"]"
        ),
        "POSIX machine path": re.compile("/" + r"(?:home|Users)/[^/\s]+/"),
        "Windows machine path": re.compile(
            r"[A-Za-z]:\\" + r"Users\\[^\\\s]+\\"
        ),
    }
    decoding_errors: list[str] = []
    sensitive_matches: list[str] = []
    for path in tracked_paths:
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            decoding_errors.append(
                f"{path.relative_to(PROJECT_ROOT).as_posix()}: {error}"
            )
            continue
        for label, pattern in sensitive_patterns.items():
            if pattern.search(content):
                sensitive_matches.append(
                    f"{path.relative_to(PROJECT_ROOT).as_posix()}: {label}"
                )

    assert not decoding_errors, f"tracked files are not strict UTF-8: {decoding_errors}"
    assert not sensitive_matches, (
        f"tracked files contain credentials or machine paths: {sensitive_matches}"
    )


def test_sdist_manifest_includes_release_documentation_and_example() -> None:
    manifest = (PROJECT_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    included_paths = {
        line.removeprefix("include ")
        for line in manifest.splitlines()
        if line.startswith("include ")
    }

    assert {
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "README.md",
        "SECURITY.md",
        "docs/MODEL_CARD.md",
        "examples/geometry.csv",
    } <= included_paths


def test_pyproject_declares_release_metadata_and_tooling() -> None:
    pyproject = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    project = pyproject["project"]

    assert project["readme"] == "README.md"
    assert project["license"] == "Apache-2.0"
    assert project["authors"] == [{"name": "Huang Qi"}]
    assert project["requires-python"] == ">=3.11"
    assert project["dependencies"] == [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "joblib",
        "openpyxl",
    ]
    assert {
        "Development Status :: 3 - Alpha",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
    } <= set(project["classifiers"])
    assert "Operating System :: OS Independent" not in project["classifiers"]
    assert not any(
        classifier.startswith("License ::") for classifier in project["classifiers"]
    ), "PEP 639 license expressions supersede deprecated license classifiers"

    repository = "https://github.com/huang-qi/aln-model"
    assert project["urls"] == {
        "Homepage": repository,
        "Repository": repository,
        "Issues": f"{repository}/issues",
    }
    assert {"build", "pytest", "pyyaml", "ruff"} <= set(
        project["optional-dependencies"]["dev"]
    )

    assert pyproject["build-system"] == {
        "requires": ["setuptools>=77.0.3"],
        "build-backend": "setuptools.build_meta",
    }
    assert pyproject["tool"]["setuptools"]["package-dir"] == {"": "src"}
    assert pyproject["tool"]["setuptools"]["packages"]["find"] == {
        "where": ["src"],
        "include": ["aln_model*"],
    }
    assert pyproject["tool"]["ruff"]["target-version"] == "py311"
    assert pyproject["tool"]["ruff"]["lint"]["select"] == ["E4", "E7", "E9"]


def test_license_is_the_official_apache_2_text() -> None:
    license_bytes = (PROJECT_ROOT / "LICENSE").read_bytes()
    normalized_license_bytes = license_bytes.replace(b"\r\n", b"\n").replace(
        b"\r", b"\n"
    )
    attributes = (PROJECT_ROOT / ".gitattributes").read_text(encoding="utf-8")

    assert "LICENSE text eol=lf" in attributes.splitlines()
    assert hashlib.sha256(normalized_license_bytes).hexdigest() == (
        "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
    )


def test_community_health_documents_define_safe_contribution_routes() -> None:
    contributing = (PROJECT_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    security = (PROJECT_ROOT / "SECURITY.md").read_text(encoding="utf-8")
    conduct = (PROJECT_ROOT / "CODE_OF_CONDUCT.md").read_text(encoding="utf-8")
    normalized_security = " ".join(security.split())
    normalized_conduct = " ".join(conduct.split())

    assert "pytest" in contributing
    assert "ruff" in contributing
    assert "pull request" in contributing.lower()
    assert "never load untrusted joblib" in security.lower()
    assert "GitHub Security Advisories" in security
    assert "https://github.com/huang-qi/aln-model/security/advisories/new" in security
    assert (
        "private channel through which repository access was granted"
        in normalized_security
    )
    assert "repository owner or administrator" in normalized_security
    assert "normal GitHub issue" in normalized_security
    assert "Contributor Covenant" in conduct
    assert "version 2.1" in conduct
    assert (
        "private channel through which repository access was granted"
        in normalized_conduct
    )
    assert "repository owner or administrator" in normalized_conduct
    assert "security/advisories" not in normalized_conduct
    assert "normal GitHub issue" in normalized_conduct


def test_github_issue_forms_cover_bug_reports_and_feature_requests() -> None:
    forms = PROJECT_ROOT / ".github" / "ISSUE_TEMPLATE"
    expected_fields = {
        "bug_report.yml": {
            "name": "Bug report",
            "required": {
                "description": "textarea",
                "reproduction": "textarea",
                "environment": "textarea",
            },
        },
        "feature_request.yml": {
            "name": "Feature request",
            "required": {
                "problem": "textarea",
                "proposal": "textarea",
            },
        },
    }

    for filename, expected in expected_fields.items():
        form = json.loads((forms / filename).read_text(encoding="utf-8"))
        assert form["name"] == expected["name"]
        assert isinstance(form["description"], str) and form["description"].strip()
        assert isinstance(form["body"], list) and form["body"]

        fields = [item for item in form["body"] if item["type"] != "markdown"]
        field_ids = [item["id"] for item in fields]
        assert all(
            isinstance(field_id, str) and field_id.strip() for field_id in field_ids
        )
        assert len(field_ids) == len(set(field_ids))

        fields_by_id = {item["id"]: item for item in fields}
        for field_id, field_type in expected["required"].items():
            field = fields_by_id[field_id]
            assert field["type"] == field_type
            assert field["validations"]["required"] is True

        for item in form["body"]:
            assert item["type"] in {
                "checkboxes",
                "dropdown",
                "input",
                "markdown",
                "textarea",
            }
            if item["type"] == "dropdown":
                options = item["attributes"]["options"]
                assert isinstance(options, list) and options


def test_ci_tests_supported_pythons_and_smokes_the_built_wheel() -> None:
    workflow_path = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"

    assert workflow_path.is_file(), "CI workflow must be committed"

    workflow_text = workflow_path.read_text(encoding="utf-8")
    workflow = yaml.load(workflow_text, Loader=yaml.BaseLoader)
    checkout_action = (
        "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"
    )
    setup_python_action = (
        "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97"
    )

    assert "on" in workflow, "BaseLoader must preserve the GitHub Actions 'on' key"
    assert workflow["permissions"] == {"contents": "read"}
    jobs = workflow["jobs"]
    test_job = jobs["test"]
    smoke_job = jobs["distribution-smoke"]

    assert test_job["runs-on"] == "ubuntu-latest"
    assert test_job["strategy"]["matrix"]["python-version"] == ["3.11", "3.12"]
    assert smoke_job["runs-on"] == "ubuntu-latest"
    assert smoke_job["needs"] == "test"
    assert smoke_job.get("strategy") == {
        "matrix": {"python-version": ["3.11", "3.12"]}
    }

    for job in (test_job, smoke_job):
        action_steps = {
            step["uses"]: step for step in job["steps"] if "uses" in step
        }
        assert set(action_steps) == {checkout_action, setup_python_action}
        assert action_steps[checkout_action]["with"] == {
            "persist-credentials": "false"
        }

    test_action_steps = {
        step["uses"]: step for step in test_job["steps"] if "uses" in step
    }
    assert test_action_steps[setup_python_action]["with"] == {
        "python-version": "${{ matrix.python-version }}",
        "cache": "pip",
        "cache-dependency-path": "pyproject.toml\nuv.lock\n",
    }
    smoke_action_steps = {
        step["uses"]: step for step in smoke_job["steps"] if "uses" in step
    }
    assert smoke_action_steps[setup_python_action]["with"] == {
        "python-version": "${{ matrix.python-version }}"
    }
    assert workflow_text.count(f"uses: {checkout_action} # v7.0.1") == 2
    assert workflow_text.count(f"uses: {setup_python_action} # v7.0.0") == 2

    test_runs = "\n".join(step.get("run", "") for step in test_job["steps"])
    assert 'python -m pip install ".[dev]"' in test_runs
    assert "ruff check ." in test_runs
    assert "python -m pytest" in test_runs

    smoke_runs = "\n".join(step.get("run", "") for step in smoke_job["steps"])
    assert smoke_runs.count("python -m build") == 1
    assert 'python -m venv "$smoke_venv"' in smoke_runs
    assert (
        'smoke_venv="${RUNNER_TEMP}/aln-wheel-smoke-${{ matrix.python-version }}"'
        in smoke_runs
    )
    assert "-name '*.whl'" in smoke_runs
    assert "!= 1" in smoke_runs, "smoke job must require exactly one wheel"
    assert "dist/*.whl" not in smoke_runs, "never pass an ambiguous glob to pip"
    assert '"$smoke_venv/bin/python" -m pip install "$wheel_path"' in smoke_runs
    assert 'cd "$RUNNER_TEMP"' in smoke_runs
    assert "import aln_model" in smoke_runs
    assert "aln_model.__version__" in smoke_runs
    assert '"$smoke_venv/bin/aln-model" --help' in smoke_runs


def test_readme_documents_private_library_usage_and_boundaries() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(readme.lower().split())

    assert readme.startswith("# ALN/XBAW surrogate")
    assert "## 中文概览" in readme
    assert "linux is required" in normalized
    assert (
        "all four cli workflows require libc `renameat2(rename_noreplace)`"
        in normalized
    )
    assert "macos and windows are unsupported" in normalized
    assert "private GitHub repository" in readme
    assert "git clone <repository-url>" in readme
    assert 'python -m pip install -e ".[dev]"' in readme
    assert "python -m pip install ." in readme
    assert "aln-model prepare" in readme
    assert "aln-model benchmark" in readme
    assert "aln-model train" in readme
    assert "aln-model predict" in readme
    assert "path/to/training-map.xlsx" in readme
    assert "path/to/touchstone-archive.zip" in readme
    assert "shared preparation" in normalized
    assert "group-aware" in normalized and "oof" in normalized
    assert all(
        route in normalized
        for route in ("extra trees", "physics boosting", "rff kernel", "functional pca")
    )
    assert "scalar" in normalized and "spectrum" in normalized
    assert "selected scalar and spectrum routes" in normalized
    assert "authorized" in normalized
    assert "never commit" in normalized
    assert "no independent d3-v validation" in normalized
    assert "trusted" in normalized and "joblib" in normalized
    assert "python -m pytest" in readme
    assert "ruff check ." in readme
    assert "python -m build" in readme
    assert "[Model card](docs/MODEL_CARD.md)" in readme
    assert "[Contributing](CONTRIBUTING.md)" in readme
    assert "[Security](SECURITY.md)" in readme
    assert "[License](LICENSE)" in readme

    private_names = (
        "model-v3",
        "prepared-v2",
        "benchmark-v2",
    )
    assert not any(name in readme for name in private_names)


def test_model_card_scopes_internal_metrics_and_undistributed_artifacts() -> None:
    model_card = (PROJECT_ROOT / "docs" / "MODEL_CARD.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(model_card.lower().split())

    assert "one internal run" in normalized
    assert "data and artifacts are not distributed" in normalized
    assert "not universal performance" in normalized
    assert "group-aware oof" in normalized
    assert "not external validation" in normalized
    assert "no d3-v" in normalized
    assert "download" not in normalized
    assert not any(
        path in model_card
        for path in (
            "artifacts/prepared-v2",
            "artifacts/benchmark-v2",
            "artifacts/model-v3",
            "artifacts/prediction-smoke-v3",
        )
    )


def test_synthetic_geometry_example_has_exact_prediction_schema_and_safe_values() -> None:
    example_path = PROJECT_ROOT / "examples" / "geometry.csv"
    example = pd.read_csv(example_path)

    assert example.columns.tolist() == [
        "Training_Row_ID",
        *PHYSICAL_COLUMNS,
        *STATE_COLUMNS,
    ]
    assert len(example) >= 2
    assert example["Training_Row_ID"].str.fullmatch(r"synthetic-\d{3}").all()
    assert example["Training_Row_ID"].is_unique
    assert all(
        pd.api.types.is_float_dtype(example[column]) for column in PHYSICAL_COLUMNS
    )
    assert all(
        pd.api.types.is_integer_dtype(example[column])
        for column in STATE_COLUMNS
        if column not in {"Template", "EG_State"}
    )
    assert set(example["Template"]) <= set(TOPOLOGIES)
    assert set(example["EG_State"]) <= {"patterned", "blanket"}
    assert pd.api.types.is_string_dtype(example["Training_Row_ID"])
    assert pd.api.types.is_string_dtype(example["Template"])
    assert pd.api.types.is_string_dtype(example["EG_State"])
    assert set(
        example.loc[
            :,
            [
                column
                for column in STATE_COLUMNS
                if column not in {"Template", "EG_State"}
            ],
        ]
        .to_numpy()
        .ravel()
    ) <= {0, 1}

    forbidden_column_fragments = (
        "target",
        "result",
        "prediction",
        "fs_hz",
        "fp_hz",
        "k_eff2",
        "spurious",
        "s11",
    )
    assert not any(
        fragment in column.lower()
        for column in example.columns
        for fragment in forbidden_column_fragments
    )
    serialized = example_path.read_text(encoding="utf-8").lower()
    assert not any(
        token in serialized
        for token in ("d1_", "d2_", ".s1p", ".s2p", ".snp", ".xlsx", ".zip")
    )
