import pandas as pd

from src.config import (
    BacktestConfig, DataConfig, SplitConfig, SegmentationConfig,
    SemanticClusteringConfig, TrainingConfig,
)
from src.services import backtest_service as bt
from src.artifacts.writer import save_backtest_suite


def _sample_df() -> pd.DataFrame:
    rows = []
    kws = ['alpha 보험', 'alpha 암보험', 'beta 건강보험', 'beta 간병보험']
    dates = pd.date_range('2026-01-01', periods=12, freq='D').tolist() + pd.date_range('2026-02-01', periods=8, freq='D').tolist()
    for i, dt in enumerate(dates):
        kw = kws[i % len(kws)]
        rows.append({'date': dt, 'keyword': kw, 'spend': float(10 + (i % 5)), 'click': int(1 + (i % 3))})
    return pd.DataFrame(rows)


def test_run_backtest_suite_end_to_end_with_stub(monkeypatch, tmp_path):
    cfg = BacktestConfig(
        data=DataConfig(),
        split=SplitConfig('2026-01-01', '2026-01-31', '2026-02-01', '2026-02-28'),
        segmentation=SegmentationConfig(use_semantic_clustering=False, semantic_apply_to_segments=('head', 'long_tail'), min_keywords_per_cluster=2),
        semantic=SemanticClusteringConfig(min_cluster_size=2, min_samples=1),
        training=TrainingConfig(advi_steps=5, posterior_draws=5, progressbar=False),
        curves=('log',),
        likelihoods=('gaussian',),
    )

    def _stub_run_single_model(train_df, test_df, segment_table, curve_name, likelihood_name, config):
        pred_df = test_df.copy()
        pred_df['pred_click'] = pred_df['click'].astype(float)
        pred_df['prediction_level'] = 'keyword'
        pred_df['keyword_train_count'] = 3
        pred_df['spend_scale'] = 1.0
        metrics = {'rmse': 0.0, 'mae': 0.0, 'mape': 0.0, 'r2': 1.0}
        fallback = {
            'prediction_level_keyword': len(pred_df),
            'prediction_level_cluster': 0,
            'prediction_level_global': 0,
            'prediction_level_keyword_share': 1.0,
            'prediction_level_cluster_share': 0.0,
            'prediction_level_global_share': 0.0,
            'n_test_rows': len(pred_df),
        }
        return pred_df, metrics, fallback

    monkeypatch.setattr(bt, '_run_single_model', _stub_run_single_model)

    result = bt.run_backtest_suite(_sample_df(), cfg)

    assert 'summary' in result and not result['summary'].empty
    assert 'segments' in result and 'cluster_id' in result['segments'].columns
    for pred in result['predictions'].values():
        assert 'cluster_id' in pred.columns

    save_backtest_suite(result, str(tmp_path / 'out'))
    assert (tmp_path / 'out' / 'summary.csv').exists()
    assert (tmp_path / 'out' / 'segments.csv').exists()
    assert (tmp_path / 'out' / 'segment_table.csv').exists()


def test_writer_recovers_missing_cluster_columns(tmp_path):
    segment_table = pd.DataFrame({'keyword': ['a'], 'cluster_id': [2], 'cluster_size': [1], 'is_clustered': [True]})
    result = {
        'summary': pd.DataFrame([{'run_name': 'x', 'rmse': 0.0, 'mae': 0.0}]),
        'segments': segment_table,
        'train_aggregated': pd.DataFrame([{'period_start': pd.Timestamp('2026-01-01'), 'keyword': 'a', 'spend': 1.0, 'click': 1}]),
        'test_aggregated': pd.DataFrame([{'period_start': pd.Timestamp('2026-02-01'), 'keyword': 'a', 'spend': 2.0, 'click': 2}]),
        'predictions': {'run1': pd.DataFrame([{'period_start': pd.Timestamp('2026-02-01'), 'keyword': 'a', 'pred_click': 2.0}])},
        'diagnostics': {},
        'config_snapshot': {},
    }
    save_backtest_suite(result, str(tmp_path / 'out2'))
    pred = pd.read_csv(tmp_path / 'out2' / 'run1.csv')
    assert 'cluster_id' in pred.columns
