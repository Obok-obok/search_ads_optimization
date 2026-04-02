
import pandas as pd

from src.services.backtest_service import _prepare_segment_frames


def test_prepare_segment_frames_keeps_unseen_test_keywords_in_long_tail():
    train_aggregated = pd.DataFrame({
        'period_start': pd.to_datetime(['2026-01-01']),
        'keyword': ['seen'],
        'spend': [1.0],
        'click': [1.0],
    })
    test_aggregated = pd.DataFrame({
        'period_start': pd.to_datetime(['2026-02-01', '2026-02-02']),
        'keyword': ['seen', 'unseen'],
        'spend': [1.0, 2.0],
        'click': [1.0, 2.0],
    })
    segment_table = pd.DataFrame({
        'keyword': ['seen'],
        'segment': ['head'],
        'cluster_id': [-1],
        'cluster_size': [0],
        'is_clustered': [False],
    })

    _, head_test, _ = _prepare_segment_frames(train_aggregated, test_aggregated, segment_table, 'head', noise_label=-1)
    _, tail_test, _ = _prepare_segment_frames(train_aggregated, test_aggregated, segment_table, 'long_tail', noise_label=-1)

    assert head_test['keyword'].tolist() == ['seen']
    assert tail_test['keyword'].tolist() == ['unseen']
