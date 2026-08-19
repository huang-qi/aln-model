from pathlib import Path


def test_project_requires_python_311_or_newer():
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()

    assert 'requires-python = ">=3.11"' in pyproject
