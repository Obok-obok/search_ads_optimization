import numpy as np
import pandas as pd

from src.config import BacktestConfig, DataConfig, SegmentationConfig, SemanticClusteringConfig, SplitConfig
from src.segmentation.pipeline import build_segment_table
from src.segmentation.semantic import reduce_embedding_dimensions


def test_reduce_embedding_dimensions_falls_back_when_target_not_smaller():
    emb = np.random.default_rng(42).normal(size=(4, 3)).astype(np.float32)
    cfg = SemanticClusteringConfig(use_umap=True, umap_n_components=8)
    out = reduce_embedding_dimensions(emb, cfg)
    assert out.shape == emb.shape


def test_build_segment_table_semantic_runs_per_segment_without_duplicate_keywords():
    df = pd.DataFrame(
        {
            'keyword': ['a', 'a', 'b', 'c', 'd', 'e'],
            'period_start': pd.to_datetime([
                '2026-01-01', '2026-01-08', '2026-01-01', '2026-01-01', '2026-01-08', '2026-01-08'
            ]),
            'spend': [10, 10, 3, 2, 1, 1],
            'click': [10, 8, 2, 1, 1, 1],
        }
    )
    cfg = BacktestConfig(
        data=DataConfig(),
        split=SplitConfig(train_start='2026-01-01', train_end='2026-01-31', test_start='2026-02-01', test_end='2026-02-28'),
        segmentation=SegmentationConfig(
            use_semantic_clustering=True,
            semantic_apply_to_segments=('head', 'long_tail'),
            min_keywords_per_cluster=2,
        ),
        semantic=SemanticClusteringConfig(use_umap=False, min_cluster_size=10),
    )
    out = build_segment_table(df, cfg)
    diag = out.attrs.get('semantic_diagnostics')
    assert out['keyword'].nunique() == len(out)
    assert set(out['segment'].unique()).issubset({'head', 'long_tail'})
    assert out['cluster_id'].eq(-1).all()
    assert diag is not None and set(diag.columns) >= {'segment', 'n_keywords', 'status'}
