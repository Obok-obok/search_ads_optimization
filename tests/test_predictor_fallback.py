import numpy as np
import pandas as pd

from src.config import HierarchyConfig
from src.models.hierarchy import HierarchyInputs
from src.models.predictor import predict_hierarchical_keyword


class _DummyCurve:
    def predict_level_numpy(self, x_scaled, posterior_means, level, indices=None):
        x_scaled = np.asarray(x_scaled, dtype=float)
        if level == 'keyword':
            return np.asarray(posterior_means['alpha_k'])[np.asarray(indices, dtype=int)] + x_scaled
        if level == 'cluster':
            return np.asarray(posterior_means['alpha_cluster'])[np.asarray(indices, dtype=int)] + x_scaled
        if level == 'global':
            return np.full_like(x_scaled, float(posterior_means['alpha_global']), dtype=float) + x_scaled
        raise ValueError(level)


class _DummyDistribution:
    def postprocess_prediction(self, mu, posterior_means):
        return mu



def _hierarchy_inputs():
    return HierarchyInputs(
        train_keyword_idx=np.array([0, 1], dtype=np.int32),
        test_keyword_idx=np.array([0.0, np.nan, np.nan]),
        n_keywords=2,
        train_cluster_idx=np.array([0.0, 1.0]),
        test_cluster_idx=np.array([0.0, 1.0, np.nan]),
        n_clusters=2,
        keyword_idx_to_cluster_idx=np.array([0, 1], dtype=int),
        keyword_train_count=np.array([5, 3], dtype=np.int32),
        test_keyword_train_count=np.array([5, 0, 0], dtype=np.int32),
        keyword_to_idx={'a': 0, 'b': 1},
        cluster_id_to_idx={10: 0, 20: 1},
        keyword_to_cluster_id={'a': 10, 'b': 20},
    )



def test_predict_hierarchical_keyword_prefers_keyword_and_surrogates_unseen():
    pred_df = predict_hierarchical_keyword(
        curve=_DummyCurve(),
        distribution=_DummyDistribution(),
        x_scaled=np.array([1.0, 1.0, 1.0]),
        posterior_means={
            'alpha_k': np.array([10.0, 20.0]),
            'alpha_cluster': np.array([100.0, 200.0]),
            'alpha_global': 1000.0,
        },
        hierarchy_inputs=_hierarchy_inputs(),
        test_df=pd.DataFrame({'click': [1, 2, 3]}),
        hierarchy_config=HierarchyConfig(
            use_cluster_surrogate_for_unseen=True,
            use_global_surrogate_for_unseen=True,
        ),
    )
    assert pred_df['posterior_source'].tolist() == ['keyword', 'cluster_surrogate', 'global_surrogate']
    assert pred_df['predicted'].tolist() == [11.0, 201.0, 1001.0]
