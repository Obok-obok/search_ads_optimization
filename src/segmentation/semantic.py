from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np
import pandas as pd

from ..config import SemanticClusteringConfig, SegmentationConfig


@lru_cache(maxsize=2)
def _load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            'sentence-transformers is required for semantic clustering. '
            'Install with: pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu '
            '&& pip install --no-cache-dir sentence-transformers hdbscan umap-learn'
        ) from exc

    return SentenceTransformer(model_name)



def _deduplicate_keywords(keywords: Iterable[str]) -> list[str]:
    series = pd.Series(list(keywords), dtype='object').dropna().astype(str).str.strip()
    if series.empty:
        return []
    series = series.loc[series.ne('')]
    return series.drop_duplicates().tolist()



def build_keyword_embeddings(keywords: list[str], config: SemanticClusteringConfig) -> np.ndarray:
    unique_keywords = _deduplicate_keywords(keywords)
    if not unique_keywords:
        return np.empty((0, 0), dtype=np.float32)

    model = _load_sentence_transformer(config.embedding_model_name)
    embeddings = model.encode(
        unique_keywords,
        batch_size=config.embedding_batch_size,
        show_progress_bar=config.show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=config.normalize_embeddings,
    )
    return np.asarray(embeddings, dtype=np.float32)



def reduce_embedding_dimensions(embeddings: np.ndarray, config: SemanticClusteringConfig) -> np.ndarray:
    if embeddings.size == 0:
        return embeddings
    if not config.use_umap:
        return np.asarray(embeddings, dtype=np.float32)

    n_samples, n_features = embeddings.shape
    if n_samples < 3:
        return np.asarray(embeddings, dtype=np.float32)

    target_components = min(config.umap_n_components, max(2, n_features), n_samples - 1)
    if target_components >= n_features:
        return np.asarray(embeddings, dtype=np.float32)

    try:
        import umap
    except ImportError:
        # graceful fallback for constrained environments
        return np.asarray(embeddings, dtype=np.float32)

    reducer = umap.UMAP(
        n_neighbors=min(config.umap_n_neighbors, max(2, n_samples - 1)),
        n_components=target_components,
        metric=config.umap_metric,
        min_dist=config.umap_min_dist,
        random_state=config.random_state,
        low_memory=config.umap_low_memory,
    )
    reduced = reducer.fit_transform(embeddings)
    return np.asarray(reduced, dtype=np.float32)



def cluster_keywords(keywords: list[str], embeddings: np.ndarray, config: SemanticClusteringConfig) -> pd.DataFrame:
    try:
        import hdbscan
    except ImportError as exc:
        raise ImportError('hdbscan is required for semantic clustering. Install with: pip install --no-cache-dir hdbscan') from exc

    unique_keywords = _deduplicate_keywords(keywords)
    if not unique_keywords:
        return pd.DataFrame(columns=['keyword', 'cluster_id'])

    if len(unique_keywords) < max(2, config.min_cluster_size):
        return pd.DataFrame({'keyword': unique_keywords, 'cluster_id': [int(config.cluster_noise_label)] * len(unique_keywords)})

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        metric=config.cluster_metric,
        core_dist_n_jobs=1,
    )
    labels = clusterer.fit_predict(embeddings).astype(int)
    return pd.DataFrame({'keyword': unique_keywords, 'cluster_id': labels})



def postprocess_clusters(cluster_df: pd.DataFrame, segmentation_config: SegmentationConfig, semantic_config: SemanticClusteringConfig) -> pd.DataFrame:
    out = cluster_df.copy()
    if out.empty:
        out['cluster_size'] = pd.Series(dtype='int64')
        out['is_clustered'] = pd.Series(dtype='bool')
        return out

    noise = int(semantic_config.cluster_noise_label)
    counts = out.loc[out['cluster_id'] != noise, 'cluster_id'].value_counts()
    small_clusters = counts.loc[counts < segmentation_config.min_keywords_per_cluster].index.tolist()
    if small_clusters:
        out.loc[out['cluster_id'].isin(small_clusters), 'cluster_id'] = noise

    counts_final = out['cluster_id'].value_counts(dropna=False).to_dict()
    out['cluster_size'] = out['cluster_id'].map(counts_final).fillna(0).astype(int)
    out['is_clustered'] = out['cluster_id'].fillna(noise).astype(int) != noise
    out.loc[~out['is_clustered'], 'cluster_size'] = 0
    return out
