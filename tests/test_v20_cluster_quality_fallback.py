import numpy as np
import pandas as pd

from src.config import (
    BacktestConfig,
    DataConfig,
    SplitConfig,
    SegmentationConfig,
    SemanticClusteringConfig,
)
from src.services.backtest_service import _apply_cluster_quality_fallback


class _DummyCurve:
    def predict_level_numpy(self, x_scaled, posterior_means, level, indices=None):
        x_scaled = np.asarray(x_scaled, dtype=float)
        if level == 'global':
            return np.full_like(x_scaled, float(posterior_means['alpha_global']), dtype=float) + x_scaled
        raise ValueError(level)


class _DummyDistribution:
    def postprocess_prediction(self, mu, posterior_means):
        return mu



def test_bad_cluster_surrogate_rows_fall_back_to_global():
    pred_df = pd.DataFrame(
        {
            'keyword': ['a', 'b', 'c', 'd'],
            'click': [1.0, 2.0, 1.0, 2.0],
            'pred_click': [10.0, 10.0, 1.0, 2.0],
            'predicted': [10.0, 10.0, 1.0, 2.0],
            'cluster_id': [7, 7, -1, -1],
            'posterior_source': ['cluster_surrogate', 'cluster_surrogate', 'global_surrogate', 'global_surrogate'],
            'spend': [1.0, 1.0, 1.0, 1.0],
        }
    )
    cfg = BacktestConfig(
        data=DataConfig(),
        split=SplitConfig(train_start='2026-01-01', train_end='2026-01-31', test_start='2026-02-01', test_end='2026-02-28'),
        segmentation=SegmentationConfig(
            use_cluster_quality_fallback=True,
            cluster_quality_r2_threshold=0.0,
            min_test_rows_per_cluster_quality=2,
            min_keywords_per_cluster_quality=2,
        ),
        semantic=SemanticClusteringConfig(min_cluster_size=2, min_samples=1),
    )

    out, diag = _apply_cluster_quality_fallback(
        pred_df,
        curve=_DummyCurve(),
        distribution=_DummyDistribution(),
        x_test_scaled=np.array([1.0, 1.0, 1.0, 1.0]),
        posterior_means={'alpha_global': 100.0},
        config=cfg,
    )

    assert diag.loc[diag['cluster_id'] == 7, 'fallback_applied'].iloc[0]
    assert out.loc[:1, 'posterior_source'].tolist() == ['global_surrogate_quality_fallback', 'global_surrogate_quality_fallback']
    assert out.loc[:1, 'pred_click'].tolist() == [101.0, 101.0]
