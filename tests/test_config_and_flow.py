import numpy as np
import pandas as pd

from src.config import BacktestConfig, DataConfig, HierarchyConfig, SegmentationConfig, SemanticClusteringConfig, SplitConfig, TrainingConfig
from src.data.aggregators import aggregate_data
from src.data.splitters import split_train_test
from src.models.hierarchy import build_hierarchy_inputs
from src.segmentation.pipeline import build_segment_table


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'date': pd.to_datetime([
                '2026-01-01', '2026-01-02', '2026-01-08', '2026-01-09',
                '2026-02-01', '2026-02-02', '2026-02-08', '2026-02-09',
            ]),
            'keyword': ['a', 'b', 'a', 'c', 'a', 'b', 'a', 'd'],
            'spend': [10, 5, 12, 3, 20, 4, 18, 2],
            'click': [5, 1, 6, 1, 9, 1, 8, 1],
        }
    )


def test_aggregate_then_split_and_rfm():
    df = _sample_df()
    cfg = BacktestConfig(
        data=DataConfig(aggregation_level='weekly', weekly_partial_policy='normalize'),
        split=SplitConfig(
            train_start='2025-12-29', train_end='2026-01-31',
            test_start='2026-02-01', test_end='2026-02-28',
        ),
        segmentation=SegmentationConfig(use_semantic_clustering=False),
        semantic=SemanticClusteringConfig(),
        training=TrainingConfig(advi_steps=10, posterior_draws=5, progressbar=False),
        curves=('log',),
        likelihoods=('nb',),
    )
    agg = aggregate_data(df, cfg.data)
    train_df, test_df = split_train_test(agg, cfg.split)
    assert 'period_start' in agg.columns
    assert len(train_df) > 0 and len(test_df) > 0
    seg = build_segment_table(train_df, cfg)
    assert set(seg['segment'].unique()).issubset({'head', 'long_tail'})


def test_hierarchy_inputs_noise_cluster_is_ignored_and_counts_preserved():
    train_df = pd.DataFrame({'keyword': ['a', 'a', 'b'], 'spend': [1, 2, 2], 'click': [1, 2, 2]})
    test_df = pd.DataFrame({'keyword': ['a', 'x'], 'spend': [1, 2], 'click': [1, 2]})
    segment_table = pd.DataFrame({'keyword': ['a', 'b'], 'cluster_id': [0, -1]})
    train_out, test_out, h = build_hierarchy_inputs(train_df, test_df, segment_table, True, noise_label=-1)
    assert h.n_keywords == 2
    assert h.n_clusters == 1
    assert h.keyword_idx_to_cluster_idx is not None
    assert h.keyword_idx_to_cluster_idx[0] == 0
    assert h.keyword_idx_to_cluster_idx[1] == -1
    assert h.keyword_train_count.tolist() == [2, 1]
    assert h.test_keyword_train_count.tolist() == [2, 0]
    assert pd.isna(test_out.loc[test_out['keyword'] == 'x', 'keyword_idx']).all()
    assert h.train_cluster_idx is not None and h.train_cluster_idx.dtype.kind == 'f'
    assert np.isnan(h.train_cluster_idx[-1])



def test_hierarchy_inputs_build_keyword_prior_scale_with_long_tail_penalty():
    train_df = pd.DataFrame({'keyword': ['a', 'a', 'a', 'b'], 'spend': [1, 2, 2, 1], 'click': [1, 2, 2, 1]})
    test_df = pd.DataFrame({'keyword': ['a', 'b'], 'spend': [1, 2], 'click': [1, 2]})
    segment_table = pd.DataFrame({
        'keyword': ['a', 'b'],
        'segment': ['head', 'long_tail'],
        'cluster_id': [-1, -1],
    })
    _, _, h = build_hierarchy_inputs(
        train_df,
        test_df,
        segment_table,
        False,
        hierarchy_config=HierarchyConfig(keyword_pooling_strength=1.0, long_tail_pooling_multiplier=2.0),
    )
    assert h.keyword_train_count.tolist() == [3, 1]
    assert h.keyword_is_long_tail.tolist() == [False, True]
    assert h.keyword_prior_scale[0] > h.keyword_prior_scale[1]
