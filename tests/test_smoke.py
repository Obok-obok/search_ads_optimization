from src.config import BacktestConfig, DataConfig, SegmentationConfig, SemanticClusteringConfig, SplitConfig, TrainingConfig


def test_config_instantiation() -> None:
    cfg = BacktestConfig(
        data=DataConfig(),
        split=SplitConfig(
            train_start='2025-12-01',
            train_end='2026-01-31',
            test_start='2026-02-01',
            test_end='2026-02-28',
        ),
        segmentation=SegmentationConfig(),
        semantic=SemanticClusteringConfig(),
        training=TrainingConfig(),
    )
    assert cfg.data.aggregation_level in {'daily', 'weekly'}
