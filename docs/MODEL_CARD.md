# ALN/XBAW surrogate model card

This card records one internal run completed on 2026-08-18
(Asia/Singapore). Its values are retained for traceability, not as a packaged
benchmark. The underlying data and artifacts are not distributed with this
repository, and these results are not universal performance guarantees.

## Model design

- Prepared rows: 1,339 matched simulations from an authorized internal dataset.
- Split: five connected-group folds; geometry and near-identical response groups
  never cross folds.
- Rare-event balance: 15 dangerous-spurious positives, exactly 3 per validation
  fold.
- Scalar head: Extra Trees on raw features, fitted on all prepared rows after OOF
  selection.
- Spectrum head: functional PCA with 24 train-fold components and complex
  reconstruction projected to the passive one-port unit disk.
- Spectrum grid: 1,191 points, 4.20–5.39 GHz at 1 MHz spacing.

The scalar and spectrum heads intentionally differ. Engineering labels were
extracted from each simulation's complete native sweep, while the complex-output
grid covers only the common band. In this run, 1,207 `fp` labels lay above
5.39 GHz, so deriving them from the shared-band prediction would truncate the
target. The pipeline reconstructs `k_eff2` as `1-(fs/fp)^2` and enforces
`fp>fs` and positive Q values. BVD/MBVD parameters are diagnostic only, not
acceptance targets.

## One internal run: five-fold OOF results

| Route | fs MAE (MHz) | fp MAE (MHz) | k_eff2 MAE (pp) | Qs log-MAE | Qp log-MAE | Worst 5% freq. (MHz) | Spurious AUCPR | Brier | Complex S11 RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Extra Trees raw | 0.100 | 1.372 | 0.0445 | 0.0214 | 0.0716 | 3.736 | 0.559 | 0.00736 | — |
| Physics boosting | 0.662 | 1.414 | 0.0369 | 0.0268 | 0.0826 | 6.878 | 0.695 | 0.00655 | — |
| RFF kernel | 12.228 | 13.395 | 0.0527 | 0.0305 | 0.1294 | 87.357 | 0.456 | 0.00963 | — |
| Functional PCA | 0.694 | 1.944 | 0.0477 | 0.0325 | 0.1321 | 11.005 | 0.607 | 0.0200 | 0.1076 |

All routes had zero physical-constraint violations in this run. Each route beat
its fold-local target-permutation control on 8 of 9 frozen metrics. Extra Trees
had the best equal-weight average metric rank (1.611 versus 1.722 for physics
boosting). A 10,000-resample paired bootstrap on per-row mean `fs`/`fp` error
gave Extra Trees minus physics boosting = -0.302 MHz, with a 95% interval of
[-0.385, -0.228] MHz.

## Applicability and limitations

The trained bundle records min/max bounds for physical inputs, allowed
`Template` and `EG_State` values, and training-supported missingness.
Predictions outside that domain are retained but marked `ood` with field-level
reasons.

- These are group-aware OOF results, not external validation on an independent
  unseen-geometry dataset. Related samples are isolated by group, but that does
  not establish performance for a new process, device family, or data source.
- There is no D3-V validation or D3-V generalization claim because D3-V data were
  not supplied to this run.
- Dangerous-spurious evaluation contains only 15 positives, so AUCPR is
  statistically fragile.
- Complex RMSE is dimensionless S11 error on the shared band and does not
  validate native frequencies above 5.39 GHz.
- The label extractor reports mode-crossing, boundary, and native-grid
  diagnostics. Downstream users should retain those audit columns when adding
  authorized data.

## Reproducibility boundary

Raw simulations, mapping workbooks, prepared datasets, OOF predictions,
serialized bundles, and generated reports remain in authorized local storage.
Data and artifacts are not distributed through this source repository. To
reproduce the workflow, an authorized user supplies local inputs and runs the
documented `prepare`, `benchmark`, `train`, and `predict` commands. Any joblib
bundle must come from a trusted source.
