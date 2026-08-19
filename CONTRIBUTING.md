# Contributing to aln-model

Thank you for helping improve `aln-model`. Please keep contributions focused,
reproducible, and free of private measurement data or generated artifacts.

## Development setup

Create a Python 3.11 or newer environment and install the project with its
development tools:

```console
python -m pip install -e ".[dev]"
```

Before opening a pull request, run the test suite and lint checks:

```console
pytest
ruff check .
python -m build
```

Add or update tests for behavior changes. Use only synthetic fixtures that are
safe to publish, and keep commits small enough to review.

## Reporting problems

Use the GitHub issue forms for public bug reports and feature requests. Do not
put vulnerabilities or sensitive data in a public issue; follow `SECURITY.md`
instead.
