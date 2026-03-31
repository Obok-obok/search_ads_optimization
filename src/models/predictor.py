from __future__ import annotations

from typing import Any

import numpy as np

from ..config import HierarchyConfig
from ..curves.base import BaseCurve
from ..distributions.base import BaseDistribution


def extract_posterior_means(trace) -> dict[str, Any]:
    means: dict[str, Any] = {}
    for var_name in trace.posterior.data_vars:
        value = trace.posterior[var_name].mean(dim=('chain', 'draw')).values
        means[var_name] = float(value) if np.ndim(value) == 0 else np.asarray(value)
    return means


def choose_prediction_level(
    keyword_idx: float | int | None,
    cluster_idx: float | int | None,
    use_semantic_clustering: bool,
    keyword_train_count: int | None = None,
    hierarchy_config: HierarchyConfig | None = None,
) -> str:
    cfg = hierarchy_config or HierarchyConfig()
    has_keyword = (
        keyword_idx is not None
        and np.isfinite(keyword_idx)
        and keyword_idx >= 0
    )
    has_cluster = (
        use_semantic_clustering
        and cluster_idx is not None
        and np.isfinite(cluster_idx)
        and cluster_idx >= 0
    )

    if has_keyword:
        count = int(keyword_train_count or 0)
        if count >= int(cfg.min_train_rows_for_keyword_prediction):
            return 'keyword'
        if has_cluster and cfg.prefer_cluster_for_sparse_keywords:
            return 'cluster'
        return 'keyword'

    if has_cluster:
        return 'cluster'
    return 'global'


def predict_with_fallback(
    curve: BaseCurve,
    distribution: BaseDistribution,
    x_scaled: np.ndarray,
    keyword_idx: np.ndarray,
    posterior_means: dict[str, Any],
    use_semantic_clustering: bool,
    cluster_idx: np.ndarray | None = None,
    keyword_train_counts: np.ndarray | None = None,
    hierarchy_config: HierarchyConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.zeros_like(x_scaled, dtype=float)
    levels = np.empty(len(x_scaled), dtype=object)
    cfg = hierarchy_config or HierarchyConfig()

    for i in range(len(x_scaled)):
        k = keyword_idx[i] if keyword_idx is not None else None
        c = None if cluster_idx is None else cluster_idx[i]
        train_count = None if keyword_train_counts is None else int(keyword_train_counts[i])
        level = choose_prediction_level(
            keyword_idx=k,
            cluster_idx=c,
            use_semantic_clustering=use_semantic_clustering,
            keyword_train_count=train_count,
            hierarchy_config=cfg,
        )
        levels[i] = level
        if level == 'global':
            mu = curve.predict_level_numpy(
                np.asarray([x_scaled[i]], dtype=float),
                posterior_means,
                'global',
            )
        elif level == 'cluster':
            mu = curve.predict_level_numpy(
                np.asarray([x_scaled[i]], dtype=float),
                posterior_means,
                'cluster',
                np.asarray([int(c)], dtype=int),
            )
        else:
            mu = curve.predict_level_numpy(
                np.asarray([x_scaled[i]], dtype=float),
                posterior_means,
                'keyword',
                np.asarray([int(k)], dtype=int),
            )
        pred[i] = float(distribution.postprocess_prediction(np.asarray(mu, dtype=float), posterior_means)[0])

    return pred, levels
