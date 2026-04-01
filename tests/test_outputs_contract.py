import json
from pathlib import Path

import pandas as pd

from src.artifacts.writer import save_backtest_suite
from src.config import BacktestConfig, DataConfig, SegmentationConfig, SemanticClusteringConfig, SplitConfig, TrainingConfig
from src.data.aggregators import aggregate_data
from src.data.splitters import split_train_test
from src.models.hierarchy import build_hierarchy_inputs
from src.segmentation.pipeline import build_segment_table


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'date': pd.to_datetime([
                '2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04',
                '2026-02-01', '2026-02-02', '2026-02-03', '2026-02-04',
            ]),
            'keyword': ['암보험', '치아보험', '간병보험', '암보험', '암보험', '치아보험', '간병보험', '실손보험'],
            'spend': [10, 8, 3, 12, 20, 4, 5, 2],
            'click': [5, 3, 1, 6, 9, 1, 2, 1],
        }
    )


def _config() -> BacktestConfig:
    return BacktestConfig(
        data=DataConfig(aggregation_level='daily', weekly_partial_policy='normalize'),
        split=SplitConfig(
            train_start='2026-01-01', train_end='2026-01-31',
            test_start='2026-02-01', test_end='2026-02-28',
        ),
        segmentation=SegmentationConfig(use_semantic_clustering=False, semantic_apply_to_segments=('head', 'long_tail'), min_keywords_per_cluster=2),
        semantic=SemanticClusteringConfig(min_cluster_size=2, min_samples=1, show_progress_bar=False),
        training=TrainingConfig(advi_steps=10, posterior_draws=5, progressbar=False),
        curves=('log',),
        likelihoods=('gaussian',),
    )


def test_hierarchy_works_when_cluster_id_already_present():
    cfg = _config()
    aggregated_df = aggregate_data(_sample_df(), cfg.data)
    train_aggregated, test_aggregated = split_train_test(aggregated_df, cfg.split)
    segment_table = build_segment_table(train_aggregated, cfg)
    train_with_cluster = train_aggregated.merge(segment_table[['keyword', 'cluster_id']], on='keyword', how='left')
    test_with_cluster = test_aggregated.merge(segment_table[['keyword', 'cluster_id']], on='keyword', how='left')
    train_out, test_out, hierarchy_inputs = build_hierarchy_inputs(
        train_df=train_with_cluster,
        test_df=test_with_cluster,
        segment_table=segment_table,
        use_semantic_clustering=True,
        noise_label=-1,
    )
    assert 'cluster_id' in train_out.columns
    assert 'cluster_id' in test_out.columns
    assert 'cluster_id_seg' not in train_out.columns
    assert hierarchy_inputs.train_keyword_idx.shape[0] == len(train_out)


def test_save_backtest_suite_writes_contract(tmp_path: Path):
    segment_table = pd.DataFrame({
        'keyword': ['a', 'b'],
        'segment': ['head', 'long_tail'],
        'cluster_id': [1, -1],
        'cluster_size': [2, 0],
        'is_clustered': [True, False],
    })
    result = {
        'summary': pd.DataFrame([{'run_name': 'r1', 'rmse': 1.0, 'mae': 0.5}]),
        'segment_table': segment_table,
        'train': pd.DataFrame({'keyword': ['a'], 'cluster_id': [1]}),
        'test': pd.DataFrame({'keyword': ['b'], 'cluster_id': [-1]}),
        'train_aggregated': pd.DataFrame({'keyword': ['a'], 'cluster_id': [1]}),
        'test_aggregated': pd.DataFrame({'keyword': ['b'], 'cluster_id': [-1]}),
        'predictions_all': pd.DataFrame({'keyword': ['b'], 'cluster_id': [-1], 'pred_click': [1.0]}),
        'keyword_level': pd.DataFrame({'keyword': ['b'], 'cluster_id': [-1], 'actual_click': [1], 'pred_click': [1.0]}),
        'cluster_level': pd.DataFrame({'cluster_id': [-1], 'actual_click': [1], 'pred_click': [1.0]}),
        'diagnostics': {'pooling': pd.DataFrame([{'n_keywords': 1, 'keyword_rows': 1}])},
        'config_snapshot': {'hello': 'world'},
    }
    save_backtest_suite(result, str(tmp_path))
    expected = {
        'summary.csv', 'segment_table.csv', 'train.csv', 'test.csv',
        'train_aggregated.csv', 'test_aggregated.csv', 'predictions_all.csv', 'keyword_level.csv',
        'cluster_level.csv', 'config_snapshot.json', 'backtest_outputs.xlsx', 'diagnostics_pooling.csv',
    }
    produced = {p.name for p in tmp_path.iterdir()}
    assert expected.issubset(produced)
    cfg = json.loads((tmp_path / 'config_snapshot.json').read_text())
    assert cfg['hello'] == 'world'
