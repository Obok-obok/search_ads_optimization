import numpy as np

from src.config import HierarchyConfig
from src.models.predictor import choose_prediction_level


def test_noise_cluster_goes_global():
    assert choose_prediction_level(np.nan, -1, True) == 'global'


def test_seen_keyword_wins_over_cluster_when_dense():
    cfg = HierarchyConfig(min_train_rows_for_keyword_prediction=2, prefer_cluster_for_sparse_keywords=True)
    assert choose_prediction_level(3, 1, True, keyword_train_count=5, hierarchy_config=cfg) == 'keyword'


def test_valid_cluster_used_when_keyword_missing():
    assert choose_prediction_level(np.nan, 0, True) == 'cluster'


def test_sparse_keyword_can_fall_back_to_cluster():
    cfg = HierarchyConfig(min_train_rows_for_keyword_prediction=3, prefer_cluster_for_sparse_keywords=True)
    assert choose_prediction_level(3, 1, True, keyword_train_count=1, hierarchy_config=cfg) == 'cluster'
