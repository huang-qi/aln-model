# ALN/XBAW surrogate

An auditable Python library for preparing simulation data, comparing surrogate
routes with group-aware out-of-fold (OOF) evaluation, training a selected model,
and predicting engineering scalars plus a complex S11 spectrum. The repository
contains source code and a synthetic schema example; it is not a data or model
registry.

**Platform requirement:** Linux is required. All four CLI workflows require
libc `renameat2(RENAME_NOREPLACE)` for atomic, no-overwrite publication. macOS
and Windows are unsupported.

## Installation

Python 3.11 or newer is required. Clone the private GitHub repository using the
access method authorized for your account, then install it:

```bash
git clone <repository-url>
cd aln-model
python -m pip install .
```

For a local editable development install, include the development tools:

```bash
python -m pip install -e ".[dev]"
```

## Command-line workflow

All paths below are placeholders. Supply your own authorized inputs and choose
local output directories that are not committed to the repository.

1. `prepare` validates and joins a training-map workbook with a Touchstone
   archive, extracts scalar labels and spectra, and writes the shared prepared
   dataset and fixed group folds.

   ```bash
   aln-model prepare \
     --workbook path/to/training-map.xlsx \
     --archive path/to/touchstone-archive.zip \
     --output path/to/prepared-output
   ```

2. `benchmark` evaluates all four routes on the same group-aware OOF folds and
   records the comparison used for model selection.

   ```bash
   aln-model benchmark \
     --prepared path/to/prepared-output \
     --output path/to/benchmark-output
   ```

3. `train` refits the selected scalar and spectrum routes on all prepared rows
   and writes a local model bundle.

   ```bash
   aln-model train \
     --prepared path/to/prepared-output \
     --benchmark path/to/benchmark-output \
     --output path/to/model-output
   ```

4. `predict` loads a trained bundle and applies it to a geometry CSV. The
   synthetic file at `examples/geometry.csv` demonstrates the exact input
   columns. Output contains `prediction_labels.csv` and
   `prediction_spectra.npz`.

   ```bash
   aln-model predict \
     --model path/to/model-output/model.joblib \
     --input examples/geometry.csv \
     --output path/to/prediction-output
   ```

Output directories are published atomically and are never overwritten. If
`Training_Row_ID` is omitted from an input CSV, the CLI creates deterministic
prediction IDs.

> **Security:** joblib files can execute code while loading. Only use a trained
> bundle from a trusted source; never load an untrusted joblib file.

## Architecture

Shared preparation creates one immutable, row-aligned contract containing
features, full-native-sweep engineering labels, spectra on the shared frequency
grid, and group folds. Those folds keep related geometries and near-identical
responses together during group-aware OOF evaluation.

The benchmark compares four routes: Extra Trees, physics boosting, an RFF
kernel model, and functional PCA. The final bundle deliberately has dual heads:
a scalar model predicts `fs`, `fp`, `k_eff2`, `Qs`, `Qp`, and dangerous-spurious
probability, while a spectrum model predicts complex S11 on the shared grid.
Keeping the scalar and spectrum models separate avoids deriving full-sweep
engineering labels from a narrower spectral band.

## Data privacy boundary

Bring an authorized workbook and Touchstone archive at runtime. Raw or prepared
data, archives, workbooks, reports, predictions, and trained bundles remain
local: never commit them to this source repository. The committed example is
fully synthetic and contains neither measured/simulated results nor proprietary
identifiers.

## Applicability and limitations

Predictions are intended for the feature ranges and categories represented by
the training data. Out-of-domain inputs are retained with field-level warnings,
not made reliable by extrapolation. Group-aware OOF measures internal
cross-validation performance; it is not independent external validation. There
is no independent D3-V validation: D3-V data were not evaluated, so there is no
D3-V performance or generalization claim. See the [Model card](docs/MODEL_CARD.md)
for the complete limitations and the clearly scoped record of one internal run.

## Development

```bash
python -m pytest
ruff check .
python -m build
```

Project policies: [Contributing](CONTRIBUTING.md), [Security](SECURITY.md), and
[License](LICENSE).

## 中文概览

本仓库提供可审计的 ALN/XBAW 代理模型流程，包括数据准备、分组 OOF 评测、训练和预测。
代码仓库不分发原始数据、训练产物或模型；获授权的用户需在本地提供工作簿和 Touchstone
压缩包，并且绝不能将私有输入或生成结果提交到仓库。模型仅适用于训练数据覆盖的范围，
当前没有 D3-V 外部验证结论。加载 joblib 模型前必须确认其来源可信。
