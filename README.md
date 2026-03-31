# Search Ads Optimization

Config-driven hierarchical Bayesian search-ads modeling project.

## What is in this stable refactor

- Package-style imports across `src/*`
- Conservative semantic clustering defaults for VM stability
- Runtime diagnostics for segmentation, clustering, and fallback levels
- Cleaner artifact export with diagnostic CSV/XLSX sheets
- Basic input validation for load, aggregate, and backtest stages

## Implemented design

- Raw input is always daily keyword data.
- Aggregation is configurable: `daily` or `weekly`.
- Weekly aggregation supports partial-week policies: `normalize`, `drop`, `keep`.
- Split is fully config-driven by date ranges. No hardcoded month logic.
- RFM is always applied first and only produces `head` / `long_tail`.
- Monetary threshold uses cumulative click-share cutoff.
- Optional semantic clustering uses `SBERT + optional UMAP + HDBSCAN`.
- Hierarchy depth:
  - clustering OFF: `global -> keyword`
  - clustering ON: `global -> cluster -> keyword`
- Prediction fallback order:
  - dense keyword -> `keyword`
  - sparse keyword with valid cluster -> `cluster`
  - otherwise -> `global`
- Curves: `log`, `hill`
- Likelihoods: `gaussian`, `nb`, `zinb`

## Recommended runtime defaults

- `embedding_batch_size=32`
- `umap_n_components=5`
- `min_cluster_size=8`
- `min_train_rows_for_keyword_prediction=3`
- `max_noise_share_to_accept=0.85`

## Running locally

Use the repository root on `PYTHONPATH` and import through `src`.

```bash
export PYTHONPATH=.
pytest -q
```

Example:

```python
from src.api import BacktestConfig, DataConfig, SplitConfig, run_backtest_suite
```

## Diagnostics exported

- `diagnostics_segmentation.csv`
- `diagnostics_clusters.csv`
- `diagnostics_semantic_runtime.csv`
- `diagnostics_fallback.csv`

## Notes

For short histories such as ~2 months, keep `aggregation_level='daily'` as the default baseline.
