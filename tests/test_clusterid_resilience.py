import pandas as pd

from src.artifacts.writer import _ensure_cluster_columns_for_export
from src.services.backtest_service import _ensure_cluster_columns


def test_ensure_cluster_columns_merges_defaults_from_segment_table():
    df = pd.DataFrame({'keyword': ['a', 'b', 'x'], 'spend': [1, 2, 3], 'click': [1, 2, 3]})
    segment_table = pd.DataFrame({
        'keyword': ['a', 'b'],
        'cluster_id': [10, -1],
        'cluster_size': [2, 0],
        'is_clustered': [True, False],
    })

    out = _ensure_cluster_columns(df, segment_table)

    assert out['cluster_id'].tolist() == [10, -1, -1]
    assert out['cluster_size'].tolist() == [2, 0, 0]
    assert out['is_clustered'].tolist() == [True, False, False]


def test_writer_export_helper_backfills_cluster_columns():
    df = pd.DataFrame({'keyword': ['a', 'x'], 'pred_click': [1.5, 0.5]})
    segment_table = pd.DataFrame({
        'keyword': ['a'],
        'cluster_id': [3],
        'cluster_size': [4],
        'is_clustered': [True],
    })

    out = _ensure_cluster_columns_for_export(df, segment_table)

    assert list(out.columns) == ['keyword', 'pred_click', 'cluster_id', 'cluster_size', 'is_clustered']
    assert out['cluster_id'].tolist() == [3, -1]
    assert out['cluster_size'].tolist() == [4, 0]
    assert out['is_clustered'].tolist() == [True, False]
