import os
import sys

PROJECT_ROOT = os.path.abspath('..')
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from api import (  # noqa: E402
    BacktestConfig,
    CurvePriorsConfig,
    DataConfig,
    DistributionPriorsConfig,
    SegmentationConfig,
    SemanticClusteringConfig,
    SplitConfig,
    TrainingConfig,
    load_keyword_data,
    run_backtest_suite,
    save_backtest_suite,
)

config = BacktestConfig(
    data=DataConfig(
        aggregation_level='daily',
        weekly_partial_policy='normalize',
    ),
    split=SplitConfig(
        train_start='2025-12-01',
        train_end='2026-01-31',
        test_start='2026-02-01',
        test_end='2026-02-28',
    ),
    segmentation=SegmentationConfig(
        recency_quantile=0.30,
        frequency_quantile=0.70,
        monetary_click_share_cutoff=0.80,
        use_semantic_clustering=False,
    ),
    semantic=SemanticClusteringConfig(
        min_cluster_size=5,
        min_samples=2,
    ),
    training=TrainingConfig(
        inference_method='advi',
        advi_steps=4000,
        posterior_draws=300,
        random_seed=42,
    ),
    curves=('log', 'hill'),
    likelihoods=('gaussian', 'nb', 'zinb'),
    curve_priors=CurvePriorsConfig(
        use_data_driven_alpha_center=True,
    ),
    distribution_priors=DistributionPriorsConfig(),
)

raw_df = load_keyword_data('../data/source/keyword.csv', config.data)
result = run_backtest_suite(raw_df=raw_df, config=config)
save_backtest_suite(result, '../outputs/backtest/run_001')
print(result['summary'].head())
