from __future__ import annotations

import pandas as pd
import numpy as np

from ..config import BacktestConfig
from .rfm import apply_rfm_head_tail, build_rfm_table
from .semantic import (
    build_keyword_embeddings,
    cluster_keywords,
    postprocess_clusters,
    reduce_embedding_dimensions,
)
from .topic_intent import build_topic_intent_frame


_SEGMENT_COLUMNS = [
    'keyword', 'segment', 'recency', 'frequency', 'monetary', 'M_share',
    'M_cum_share', 'is_top_click_group', 'topic', 'intent', 'routing_key',
    'cluster_id', 'cluster_size', 'is_clustered', 'cluster_representative_keyword',
]


def _empty_cluster_frame(noise_label: int) -> pd.DataFrame:
    return pd.DataFrame(columns=['keyword', 'cluster_id', 'cluster_size', 'is_clustered', 'cluster_representative_keyword']).assign(
        cluster_id=pd.Series(dtype='int64'),
        cluster_size=pd.Series(dtype='int64'),
        is_clustered=pd.Series(dtype='bool'),
        cluster_representative_keyword=pd.Series(dtype='object'),
    )


def _compute_cluster_representatives(cluster_df: pd.DataFrame, reduced_embeddings: np.ndarray, noise_label: int) -> dict[int, str]:
    reps: dict[int, str] = {}
    if cluster_df.empty or reduced_embeddings.size == 0:
        return reps

    labels = cluster_df['cluster_id'].to_numpy(dtype=int)
    keywords = cluster_df['keyword'].astype(str).tolist()
    for cluster_id in sorted(set(labels.tolist())):
        if cluster_id == int(noise_label):
            continue
        idx = np.where(labels == cluster_id)[0]
        if idx.size == 0:
            continue
        if idx.size == 1:
            reps[int(cluster_id)] = keywords[int(idx[0])]
            continue
        sub = reduced_embeddings[idx]
        centroid = sub.mean(axis=0)
        dist = np.sum((sub - centroid) ** 2, axis=1)
        reps[int(cluster_id)] = keywords[int(idx[int(np.argmin(dist))])]
    return reps


def _cluster_one_group(target_df: pd.DataFrame, config: BacktestConfig) -> tuple[pd.DataFrame, dict]:
    noise_label = int(config.semantic.cluster_noise_label)
    segment_name = str(target_df['segment'].iloc[0]) if not target_df.empty and 'segment' in target_df.columns else 'unknown'
    routing_key = str(target_df['routing_key'].iloc[0]) if not target_df.empty and 'routing_key' in target_df.columns else f'{segment_name}__unknown__general'
    keywords = target_df['keyword'].astype(str).dropna().drop_duplicates().tolist() if not target_df.empty else []
    diag = {
        'segment': segment_name,
        'routing_key': routing_key,
        'n_keywords': len(keywords),
        'embedding_dim': 0,
        'reduced_dim': 0,
        'n_clusters': 0,
        'clustered_keywords': 0,
        'noise_keywords': len(keywords),
        'noise_share': 1.0 if keywords else 0.0,
        'status': 'empty',
    }
    if target_df.empty:
        return _empty_cluster_frame(noise_label), diag

    min_group = max(2, int(config.segmentation.min_keywords_per_routing_group))
    min_cluster_size = max(2, int(config.semantic.min_cluster_size))
    if len(keywords) < max(min_group, min_cluster_size):
        diag['status'] = 'skipped_too_few_keywords'
        return pd.DataFrame(
            {
                'keyword': keywords,
                'cluster_id': [noise_label] * len(keywords),
                'cluster_size': [0] * len(keywords),
                'is_clustered': [False] * len(keywords),
                'cluster_representative_keyword': [None] * len(keywords),
            }
        ), diag

    embeddings = build_keyword_embeddings(keywords, config.semantic)
    diag['embedding_dim'] = int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.size else 0
    reduced = reduce_embedding_dimensions(embeddings, config.semantic)
    diag['reduced_dim'] = int(reduced.shape[1]) if reduced.ndim == 2 and reduced.size else 0

    cluster_df = cluster_keywords(keywords, reduced, config.semantic)
    cluster_df = postprocess_clusters(cluster_df, config.segmentation, config.semantic)

    reps = _compute_cluster_representatives(cluster_df, reduced, noise_label)
    cluster_df['cluster_representative_keyword'] = cluster_df['cluster_id'].map(reps)
    cluster_df.loc[~cluster_df['is_clustered'], 'cluster_representative_keyword'] = None

    valid_clusters = int(cluster_df.loc[cluster_df['is_clustered'], 'cluster_id'].nunique())
    noise_share = 1.0 if cluster_df.empty else float((~cluster_df['is_clustered']).mean())
    diag.update({
        'n_clusters': valid_clusters,
        'clustered_keywords': int(cluster_df['is_clustered'].sum()) if not cluster_df.empty else 0,
        'noise_keywords': int((~cluster_df['is_clustered']).sum()) if not cluster_df.empty else len(keywords),
        'noise_share': noise_share,
        'status': 'ok',
    })

    if (
        valid_clusters < int(config.semantic.min_valid_clusters)
        or noise_share > float(config.semantic.max_noise_share_to_accept)
    ):
        cluster_df['cluster_id'] = noise_label
        cluster_df['cluster_size'] = 0
        cluster_df['is_clustered'] = False
        cluster_df['cluster_representative_keyword'] = None
        diag.update({
            'n_clusters': 0,
            'clustered_keywords': 0,
            'noise_keywords': len(cluster_df),
            'noise_share': 1.0 if len(cluster_df) else 0.0,
            'status': 'disabled_noise_threshold',
        })
    return cluster_df, diag


def build_segment_table(train_df: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    rfm = build_rfm_table(train_df, config.segmentation)
    segment_table = apply_rfm_head_tail(rfm, config.segmentation)
    segment_table['topic'] = 'unknown'
    segment_table['intent'] = 'general'
    segment_table['routing_key'] = segment_table['segment'].astype(str) + '__default'
    segment_table['cluster_id'] = int(config.semantic.cluster_noise_label)
    segment_table['cluster_size'] = 0
    segment_table['is_clustered'] = False
    segment_table['cluster_representative_keyword'] = None
    semantic_diags: list[dict] = []

    semantic_segments = [
        segment for segment in config.segmentation.semantic_apply_to_segments
        if segment in set(segment_table['segment'].dropna().astype(str))
    ]

    if bool(config.segmentation.use_topic_intent_routing):
        for segment_name in semantic_segments:
            mask = segment_table['segment'].eq(segment_name)
            routing_frame = build_topic_intent_frame(segment_table.loc[mask, 'keyword'].tolist(), segment=segment_name)
            if len(routing_frame) > 0:
                route_map = routing_frame.set_index('keyword')
                for col in ['topic', 'intent', 'routing_key']:
                    segment_table.loc[mask, col] = segment_table.loc[mask, 'keyword'].map(route_map[col])

    if config.segmentation.use_semantic_clustering:
        cluster_frames: list[pd.DataFrame] = []
        next_cluster_base = 0

        for segment_name in semantic_segments:
            mask = segment_table['segment'].eq(segment_name)
            target = segment_table.loc[mask, ['keyword', 'segment', 'topic', 'intent', 'routing_key']].copy()
            if not bool(config.segmentation.use_topic_intent_routing):
                target['routing_key'] = f'{segment_name}__all'
                target['topic'] = 'unknown'
                target['intent'] = 'general'
                segment_table.loc[mask, 'routing_key'] = target.set_index('keyword')['routing_key'].reindex(segment_table.loc[mask, 'keyword']).to_numpy()

            for _, route_group in target.groupby('routing_key', dropna=False):
                cluster_df, diag = _cluster_one_group(route_group[['keyword', 'segment', 'routing_key']], config)
                semantic_diags.append(diag)
                if cluster_df.empty:
                    continue
                valid_mask = cluster_df['cluster_id'].ne(int(config.semantic.cluster_noise_label))
                if valid_mask.any():
                    cluster_df.loc[valid_mask, 'cluster_id'] = cluster_df.loc[valid_mask, 'cluster_id'].astype(int) + next_cluster_base
                    next_cluster_base = int(cluster_df.loc[valid_mask, 'cluster_id'].max()) + 1
                cluster_frames.append(cluster_df)

        if cluster_frames:
            merged_clusters = pd.concat(cluster_frames, ignore_index=True).drop_duplicates(subset=['keyword'], keep='first')
            segment_table = segment_table.merge(merged_clusters, on='keyword', how='left', suffixes=('', '_semantic'))
            for col in ['cluster_id', 'cluster_size', 'is_clustered', 'cluster_representative_keyword']:
                segment_table[col] = segment_table[f'{col}_semantic'].combine_first(segment_table[col])
                segment_table = segment_table.drop(columns=[f'{col}_semantic'])
            segment_table['cluster_id'] = segment_table['cluster_id'].fillna(int(config.semantic.cluster_noise_label)).astype(int)
            segment_table['cluster_size'] = segment_table['cluster_size'].fillna(0).astype(int)
            segment_table['is_clustered'] = segment_table['is_clustered'].fillna(False).astype(bool)

    segment_table = segment_table[_SEGMENT_COLUMNS].copy()
    segment_table.attrs['semantic_diagnostics'] = pd.DataFrame(semantic_diags)
    return segment_table
