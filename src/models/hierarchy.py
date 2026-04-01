from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import HierarchyConfig
from .pooling import compute_keyword_prior_scale


@dataclass(slots=True)
class HierarchyInputs:
    train_keyword_idx: np.ndarray
    test_keyword_idx: np.ndarray
    n_keywords: int
    train_cluster_idx: np.ndarray | None
    test_cluster_idx: np.ndarray | None
    n_clusters: int
    keyword_idx_to_cluster_idx: np.ndarray | None
    keyword_train_count: np.ndarray
    test_keyword_train_count: np.ndarray
    keyword_is_long_tail: np.ndarray
    keyword_prior_scale: np.ndarray
    keyword_to_idx: dict[str, int]
    cluster_id_to_idx: dict[int, int]
    keyword_to_cluster_id: dict[str, int]


def _ensure_cluster_id_column(df: pd.DataFrame, noise_label: int) -> pd.DataFrame:
    out = df.copy()
    if 'cluster_id' not in out.columns:
        out['cluster_id'] = int(noise_label)
    out['cluster_id'] = out['cluster_id'].fillna(int(noise_label)).astype(int)
    return out



def build_hierarchy_inputs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    segment_table: pd.DataFrame,
    use_semantic_clustering: bool,
    noise_label: int = -1,
    hierarchy_config: HierarchyConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, HierarchyInputs]:
    train_out = train_df.copy()
    test_out = test_df.copy()

    all_keywords = sorted(train_out['keyword'].astype(str).unique())
    keyword_to_idx = {keyword: idx for idx, keyword in enumerate(all_keywords)}
    keyword_counts = train_out['keyword'].astype(str).value_counts()

    train_out['keyword_idx'] = train_out['keyword'].map(keyword_to_idx).astype(int)
    test_out['keyword_idx'] = test_out['keyword'].map(keyword_to_idx)

    n_keywords = len(all_keywords)
    keyword_train_count = np.zeros(n_keywords, dtype=np.int32)
    for keyword, keyword_idx in keyword_to_idx.items():
        keyword_train_count[keyword_idx] = int(keyword_counts.get(keyword, 0))

    segment_lookup = (
        segment_table[['keyword', 'segment']]
        .dropna(subset=['keyword'])
        .drop_duplicates(subset=['keyword'])
        .copy()
        if 'segment' in segment_table.columns
        else pd.DataFrame(columns=['keyword', 'segment'])
    )
    keyword_is_long_tail = np.zeros(n_keywords, dtype=bool)
    if len(segment_lookup) > 0:
        keyword_to_segment = dict(zip(segment_lookup['keyword'].astype(str), segment_lookup['segment'], strict=False))
        for keyword, keyword_idx in keyword_to_idx.items():
            keyword_is_long_tail[keyword_idx] = str(keyword_to_segment.get(keyword, 'head')) == 'long_tail'

    test_keyword_train_count = np.zeros(len(test_out), dtype=np.int32)
    if len(test_out) > 0:
        valid_mask = test_out['keyword_idx'].notna()
        if valid_mask.any():
            valid_indices = test_out.loc[valid_mask, 'keyword_idx'].astype(int).to_numpy()
            test_keyword_train_count[valid_mask.to_numpy()] = keyword_train_count[valid_indices]

    train_cluster_idx = None
    test_cluster_idx = None
    keyword_idx_to_cluster_idx = None
    cluster_id_to_idx: dict[int, int] = {}
    keyword_to_cluster_id: dict[str, int] = {}
    n_clusters = 0

    if use_semantic_clustering:
        valid_cluster_lookup = segment_table[['keyword', 'cluster_id']].dropna().copy()
        if len(valid_cluster_lookup) > 0:
            valid_cluster_lookup['cluster_id'] = valid_cluster_lookup['cluster_id'].astype(int)
            valid_cluster_lookup = valid_cluster_lookup.loc[valid_cluster_lookup['cluster_id'] != int(noise_label)].copy()

        train_out = _ensure_cluster_id_column(train_out, noise_label)
        test_out = _ensure_cluster_id_column(test_out, noise_label)

        if len(valid_cluster_lookup) > 0:
            cluster_ids = sorted(valid_cluster_lookup['cluster_id'].unique().tolist())
            cluster_id_to_idx = {cluster_id: cluster_idx for cluster_idx, cluster_id in enumerate(cluster_ids)}
            lookup = valid_cluster_lookup.drop_duplicates(subset=['keyword'])
            keyword_to_cluster_id = dict(zip(lookup['keyword'].astype(str), lookup['cluster_id'].astype(int), strict=False))

            if 'cluster_id' not in train_out.columns:
                train_out = train_out.merge(lookup, on='keyword', how='left')
            else:
                train_out = train_out.merge(lookup, on='keyword', how='left', suffixes=('', '_seg'))
                if 'cluster_id_seg' in train_out.columns:
                    train_out['cluster_id'] = train_out['cluster_id'].where(
                        train_out['cluster_id'].ne(int(noise_label)),
                        train_out['cluster_id_seg'],
                    )
                    train_out = train_out.drop(columns=['cluster_id_seg'])

            if 'cluster_id' not in test_out.columns:
                test_out = test_out.merge(lookup, on='keyword', how='left')
            else:
                test_out = test_out.merge(lookup, on='keyword', how='left', suffixes=('', '_seg'))
                if 'cluster_id_seg' in test_out.columns:
                    test_out['cluster_id'] = test_out['cluster_id'].where(
                        test_out['cluster_id'].ne(int(noise_label)),
                        test_out['cluster_id_seg'],
                    )
                    test_out = test_out.drop(columns=['cluster_id_seg'])

            train_out = _ensure_cluster_id_column(train_out, noise_label)
            test_out = _ensure_cluster_id_column(test_out, noise_label)
            train_out['cluster_idx'] = train_out['cluster_id'].map(cluster_id_to_idx).astype(float)
            test_out['cluster_idx'] = test_out['cluster_id'].map(cluster_id_to_idx).astype(float)
            train_cluster_idx = train_out['cluster_idx'].to_numpy(dtype=float)
            test_cluster_idx = test_out['cluster_idx'].to_numpy(dtype=float)
            n_clusters = len(cluster_id_to_idx)

            keyword_cluster_df = (
                train_out[['keyword_idx', 'cluster_idx']]
                .dropna(subset=['cluster_idx'])
                .drop_duplicates('keyword_idx')
                .sort_values('keyword_idx')
            )
            keyword_idx_to_cluster_idx = np.full(n_keywords, -1, dtype=int)
            if len(keyword_cluster_df) > 0:
                keyword_idx_to_cluster_idx[keyword_cluster_df['keyword_idx'].to_numpy(dtype=int)] = keyword_cluster_df['cluster_idx'].to_numpy(dtype=int)

    keyword_prior_scale = compute_keyword_prior_scale(
        keyword_train_count=keyword_train_count,
        keyword_is_long_tail=keyword_is_long_tail,
        hierarchy_config=hierarchy_config or HierarchyConfig(),
        n_keywords=n_keywords,
    )

    hierarchy_inputs = HierarchyInputs(
        train_keyword_idx=train_out['keyword_idx'].to_numpy(dtype=np.int32),
        test_keyword_idx=test_out['keyword_idx'].to_numpy(dtype=float),
        n_keywords=n_keywords,
        train_cluster_idx=train_cluster_idx,
        test_cluster_idx=test_cluster_idx,
        n_clusters=n_clusters,
        keyword_idx_to_cluster_idx=keyword_idx_to_cluster_idx,
        keyword_train_count=keyword_train_count,
        test_keyword_train_count=test_keyword_train_count,
        keyword_is_long_tail=keyword_is_long_tail,
        keyword_prior_scale=keyword_prior_scale,
        keyword_to_idx=keyword_to_idx,
        cluster_id_to_idx=cluster_id_to_idx,
        keyword_to_cluster_id=keyword_to_cluster_id,
    )
    return train_out, test_out, hierarchy_inputs
