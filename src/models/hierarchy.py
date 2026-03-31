from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class HierarchyInputs:
    keyword_idx_train: np.ndarray
    keyword_idx_test: np.ndarray
    n_keywords: int
    cluster_idx_train: np.ndarray | None
    cluster_idx_test: np.ndarray | None
    n_clusters: int
    keyword_to_cluster_idx: np.ndarray | None
    keyword_train_counts: np.ndarray
    keyword_train_counts_test: np.ndarray


def build_hierarchy_inputs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    segment_table: pd.DataFrame,
    use_semantic_clustering: bool,
    noise_label: int = -1,
) -> tuple[pd.DataFrame, pd.DataFrame, HierarchyInputs]:
    train_out = train_df.copy()
    test_out = test_df.copy()

    all_keywords = sorted(train_out['keyword'].astype(str).unique())
    keyword_to_idx = {kw: idx for idx, kw in enumerate(all_keywords)}
    keyword_counts = train_out['keyword'].astype(str).value_counts()

    train_out['keyword_idx'] = train_out['keyword'].map(keyword_to_idx).astype(int)
    test_out['keyword_idx'] = test_out['keyword'].map(keyword_to_idx)

    n_keywords = len(all_keywords)
    keyword_train_counts = np.zeros(n_keywords, dtype=np.int32)
    for kw, idx in keyword_to_idx.items():
        keyword_train_counts[idx] = int(keyword_counts.get(kw, 0))

    keyword_train_counts_test = np.zeros(len(test_out), dtype=np.int32)
    if len(test_out) > 0:
        valid_mask = test_out['keyword_idx'].notna()
        if valid_mask.any():
            valid_indices = test_out.loc[valid_mask, 'keyword_idx'].astype(int).to_numpy()
            keyword_train_counts_test[valid_mask.to_numpy()] = keyword_train_counts[valid_indices]

    cluster_idx_train = None
    cluster_idx_test = None
    keyword_to_cluster_idx = None
    n_clusters = 0

    if use_semantic_clustering:
        valid_clusters = segment_table[['keyword', 'cluster_id']].dropna().copy()
        if len(valid_clusters) > 0:
            valid_clusters['cluster_id'] = valid_clusters['cluster_id'].astype(int)
            valid_clusters = valid_clusters.loc[valid_clusters['cluster_id'] != int(noise_label)].copy()

        if len(valid_clusters) > 0:
            cluster_values = sorted(valid_clusters['cluster_id'].unique().tolist())
            cluster_to_idx = {cluster_id: idx for idx, cluster_id in enumerate(cluster_values)}
            train_out = train_out.merge(valid_clusters, on='keyword', how='left')
            test_out = test_out.merge(valid_clusters, on='keyword', how='left')
            train_out['cluster_idx'] = train_out['cluster_id'].map(cluster_to_idx).astype(float)
            test_out['cluster_idx'] = test_out['cluster_id'].map(cluster_to_idx).astype(float)
            cluster_idx_train = train_out['cluster_idx'].to_numpy(dtype=float)
            cluster_idx_test = test_out['cluster_idx'].to_numpy(dtype=float)
            n_clusters = len(cluster_to_idx)

            kw_cluster = (
                train_out[['keyword_idx', 'cluster_idx']]
                .dropna(subset=['cluster_idx'])
                .drop_duplicates('keyword_idx')
                .sort_values('keyword_idx')
            )
            keyword_to_cluster_idx = np.full(n_keywords, -1, dtype=int)
            if len(kw_cluster) > 0:
                keyword_to_cluster_idx[kw_cluster['keyword_idx'].to_numpy(dtype=int)] = kw_cluster['cluster_idx'].to_numpy(dtype=int)

    hierarchy = HierarchyInputs(
        keyword_idx_train=train_out['keyword_idx'].to_numpy(dtype=np.int32),
        keyword_idx_test=test_out['keyword_idx'].to_numpy(dtype=float),
        n_keywords=n_keywords,
        cluster_idx_train=cluster_idx_train,
        cluster_idx_test=cluster_idx_test,
        n_clusters=n_clusters,
        keyword_to_cluster_idx=keyword_to_cluster_idx,
        keyword_train_counts=keyword_train_counts,
        keyword_train_counts_test=keyword_train_counts_test,
    )
    return train_out, test_out, hierarchy
