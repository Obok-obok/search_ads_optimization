from __future__ import annotations

from typing import Any

import numpy as np

from ..config import HierarchyConfig
from ..curves.base import BaseCurve
from ..distributions.base import BaseDistribution
from .hierarchy import HierarchyInputs


def extract_posterior_means(trace) -> dict[str, Any]:
    means: dict[str, Any] = {}
    for var_name in trace.posterior.data_vars:
        value = trace.posterior[var_name].mean(dim=('chain', 'draw')).values
        means[var_name] = float(value) if np.ndim(value) == 0 else np.asarray(value)
    return means


def _predict_by_source(
    curve: BaseCurve,
    distribution: BaseDistribution,
    x_scaled: np.ndarray,
    posterior_means: dict[str, Any],
    source: str,
    indices: np.ndarray | None = None,
) -> np.ndarray:
    mu = curve.predict_level_numpy(
        np.asarray(x_scaled, dtype=float),
        posterior_means,
        source,
        None if indices is None else np.asarray(indices, dtype=int),
    )
    return np.asarray(distribution.postprocess_prediction(np.asarray(mu, dtype=float), posterior_means), dtype=float)


def predict_hierarchical_keyword(
    curve: BaseCurve,
    distribution: BaseDistribution,
    x_scaled: np.ndarray,
    posterior_means: dict[str, Any],
    hierarchy_inputs: HierarchyInputs,
    test_df,
    hierarchy_config: HierarchyConfig | None = None,
):
    cfg = hierarchy_config or HierarchyConfig()
    pred = np.zeros(len(x_scaled), dtype=float)
    posterior_source = np.full(len(x_scaled), '', dtype=object)

    keyword_valid = np.isfinite(hierarchy_inputs.test_keyword_idx) & (hierarchy_inputs.test_keyword_idx >= 0)
    min_rows = max(int(cfg.min_train_rows_for_keyword_prediction), 0)
    if min_rows > 0:
        keyword_valid = keyword_valid & (np.asarray(hierarchy_inputs.test_keyword_train_count, dtype=int) >= min_rows)
    cluster_valid = (
        hierarchy_inputs.test_cluster_idx is not None
        and np.isfinite(hierarchy_inputs.test_cluster_idx)
        & (hierarchy_inputs.test_cluster_idx >= 0)
    )
    if hierarchy_inputs.test_cluster_idx is None:
        cluster_valid = np.zeros(len(x_scaled), dtype=bool)

    if keyword_valid.any():
        pred[keyword_valid] = _predict_by_source(
            curve=curve,
            distribution=distribution,
            x_scaled=x_scaled[keyword_valid],
            posterior_means=posterior_means,
            source='keyword',
            indices=hierarchy_inputs.test_keyword_idx[keyword_valid],
        )
        posterior_source[keyword_valid] = 'keyword'

    unseen_mask = ~keyword_valid
    if unseen_mask.any():
        if bool(cfg.use_cluster_surrogate_for_unseen) and np.any(cluster_valid & unseen_mask):
            mask = cluster_valid & unseen_mask
            pred[mask] = _predict_by_source(
                curve=curve,
                distribution=distribution,
                x_scaled=x_scaled[mask],
                posterior_means=posterior_means,
                source='cluster',
                indices=hierarchy_inputs.test_cluster_idx[mask],
            )
            posterior_source[mask] = 'cluster_surrogate'

        remaining = unseen_mask & (posterior_source == '')
        if remaining.any():
            if not bool(cfg.use_global_surrogate_for_unseen):
                raise ValueError('Encountered unseen test keywords with no allowed surrogate prediction source.')
            pred[remaining] = _predict_by_source(
                curve=curve,
                distribution=distribution,
                x_scaled=x_scaled[remaining],
                posterior_means=posterior_means,
                source='global',
                indices=None,
            )
            posterior_source[remaining] = 'global_surrogate'

    out = test_df.copy()
    out['predicted'] = pred
    out['posterior_source'] = posterior_source
    return out
