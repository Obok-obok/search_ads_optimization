import pandas as pd

from src.config import BacktestConfig, DataConfig, SegmentationConfig, SemanticClusteringConfig, SplitConfig, TrainingConfig
from src.data.aggregators import aggregate_data
from src.data.splitters import split_train_test
from src.segmentation.pipeline import build_segment_table
from src.segmentation.topic_intent import build_topic_intent_frame


def test_topic_intent_frame_separates_topic_and_price_intent():
    frame = build_topic_intent_frame(['간병보험', '간병인보험비용', '치아보험비용'], segment='long_tail')
    by_keyword = frame.set_index('keyword')
    assert by_keyword.loc['간병보험', 'topic'] == '간병'
    assert by_keyword.loc['간병인보험비용', 'intent'] == 'price'
    assert by_keyword.loc['치아보험비용', 'topic'] == '치아'
    assert by_keyword.loc['간병인보험비용', 'routing_key'] != by_keyword.loc['치아보험비용', 'routing_key']


def test_segment_table_includes_routing_columns_when_enabled():
    raw = pd.DataFrame(
        {
            'date': pd.to_datetime([
                '2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04',
                '2026-01-05', '2026-01-06', '2026-01-07', '2026-01-08',
            ]),
            'keyword': ['간병보험', '간병인보험비용', '치아보험', '치아보험비용', '암보험', '암보험비용', '간병보험', '치아보험'],
            'spend': [10, 8, 5, 4, 6, 5, 11, 6],
            'click': [5, 3, 2, 1, 3, 1, 5, 2],
        }
    )
    cfg = BacktestConfig(
        data=DataConfig(aggregation_level='daily', weekly_partial_policy='normalize'),
        split=SplitConfig(train_start='2026-01-01', train_end='2026-01-31', test_start='2026-02-01', test_end='2026-02-28'),
        segmentation=SegmentationConfig(use_semantic_clustering=True, use_topic_intent_routing=True, semantic_apply_to_segments=('long_tail',), min_keywords_per_cluster=2, min_keywords_per_routing_group=2),
        semantic=SemanticClusteringConfig(min_cluster_size=2, min_samples=1, show_progress_bar=False, use_umap=False),
        training=TrainingConfig(advi_steps=5, posterior_draws=5, progressbar=False),
        curves=('log',),
        likelihoods=('gaussian',),
    )
    agg = aggregate_data(raw, cfg.data)
    train_df, _ = split_train_test(agg, cfg.split)
    seg = build_segment_table(train_df, cfg)
    assert {'topic', 'intent', 'routing_key', 'cluster_representative_keyword'}.issubset(seg.columns)
