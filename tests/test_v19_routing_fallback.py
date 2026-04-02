import pandas as pd

from src.config import BacktestConfig, DataConfig, SplitConfig, SegmentationConfig, SemanticClusteringConfig
from src.data.aggregators import aggregate_data
from src.segmentation.pipeline import build_segment_table


def test_small_routing_groups_fall_back_to_all_when_group_support_is_too_low():
    raw = pd.DataFrame(
        {
            'date': pd.to_datetime([
                '2026-01-01','2026-01-02','2026-01-03','2026-01-04','2026-01-05','2026-01-06',
                '2026-01-01','2026-01-02','2026-01-03','2026-01-04','2026-01-05','2026-01-06',
                '2026-01-01','2026-01-02',
            ]),
            'keyword': [
                '간병보험','간병보험','간병보험료','간병보험료','간병비용','간병비용',
                '치아보험','치아보험','치아보험료','치아보험료','치아비용','치아비용',
                '희귀보험료','희귀보험료',
            ],
            'spend': [10]*14,
            'click': [1]*14,
        }
    )
    cfg = BacktestConfig(
        data=DataConfig(),
        split=SplitConfig(train_start='2026-01-01', train_end='2026-01-31', test_start='2026-02-01', test_end='2026-02-28'),
        segmentation=SegmentationConfig(
            use_semantic_clustering=False,
            use_topic_intent_routing=True,
            routing_mode='topic',
            min_keywords_per_routing_group=3,
            min_train_rows_per_routing_group=5,
        ),
        semantic=SemanticClusteringConfig(min_cluster_size=2, min_samples=1),
    )
    agg = aggregate_data(raw, cfg.data)
    seg = build_segment_table(agg, cfg)
    routing = dict(zip(seg['keyword'], seg['routing_key']))
    assert routing['희귀보험료'].endswith('__all')
