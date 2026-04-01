import os
import sys

PROJECT_ROOT = os.path.abspath('..')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.api import (  # noqa: E402
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
        use_semantic_clustering=True,
        semantic_apply_to_segments=('head', 'long_tail'),
        min_keywords_per_cluster=2,
    ),
    semantic=SemanticClusteringConfig(
        min_cluster_size=2,
        min_samples=1,
        show_progress_bar=False,
    ),
    training=TrainingConfig(
        inference_method='advi',
        advi_steps=4000,
        posterior_draws=300,
        random_seed=42,
    ),
    curves=('log',),
    likelihoods=('gaussian',),
    curve_priors=CurvePriorsConfig(
        use_data_driven_alpha_center=True,
    ),
    distribution_priors=DistributionPriorsConfig(),
)

candidate_paths = [
    '../data/source/keyword.csv',
    'keyword.csv',
    './keyword.csv',
]

data_path = next((path for path in candidate_paths if os.path.exists(path)), None)
if data_path is None:
    raise FileNotFoundError(
        "keyword.csv not found. Place the file in ../data/source/keyword.csv or current working directory."
    )

raw_df = load_keyword_data(data_path, config.data)
result = run_backtest_suite(raw_df=raw_df, config=config)
save_backtest_suite(result, '../outputs/backtest/run_001')
print(result['summary'].head())
print(result['cluster_level'].head())
print(result['diagnostics']['pooling'].head())
