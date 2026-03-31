from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal, Sequence


AggregationLevel = Literal['daily', 'weekly']
WeeklyPartialPolicy = Literal['normalize', 'drop', 'keep']
InferenceMethod = Literal['advi']
CurveName = Literal['log', 'hill']
LikelihoodName = Literal['gaussian', 'nb', 'zinb']


@dataclass(slots=True)
class DataConfig:
    """Raw input and aggregation settings."""

    date_col: str = 'date'
    keyword_col: str = 'keyword'
    spend_col: str = 'spend'
    click_col: str = 'click'
    aggregation_level: AggregationLevel = 'daily'
    week_anchor: str = 'MON'
    weekly_partial_policy: WeeklyPartialPolicy = 'normalize'
    weekly_expected_days: int = 7


@dataclass(slots=True)
class SplitConfig:
    """Date range split settings without hardcoded month logic."""

    train_start: str
    train_end: str
    test_start: str
    test_end: str


@dataclass(slots=True)
class SegmentationConfig:
    """RFM-first segmentation settings."""

    recency_quantile: float = 0.30
    frequency_quantile: float = 0.70
    monetary_click_share_cutoff: float = 0.80
    use_semantic_clustering: bool = False
    semantic_apply_to_segments: tuple[str, ...] = ('long_tail',)
    min_keywords_per_cluster: int = 3


@dataclass(slots=True)
class SemanticClusteringConfig:
    """Semantic clustering using SBERT + optional UMAP + HDBSCAN.

    Defaults are set conservatively for VM stability.
    """

    embedding_model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    embedding_batch_size: int = 32
    normalize_embeddings: bool = True
    show_progress_bar: bool = False

    use_umap: bool = True
    umap_n_neighbors: int = 15
    umap_n_components: int = 5
    umap_min_dist: float = 0.0
    umap_metric: str = 'cosine'
    umap_low_memory: bool = True

    min_cluster_size: int = 8
    min_samples: int = 2
    cluster_metric: str = 'euclidean'
    cluster_noise_label: int = -1
    random_state: int = 42

    min_valid_clusters: int = 1
    max_noise_share_to_accept: float = 0.85


@dataclass(slots=True)
class HierarchyConfig:
    """Keyword/cluster/global fallback policy for stable prediction."""

    min_train_rows_for_keyword_prediction: int = 3
    prefer_cluster_for_sparse_keywords: bool = True


@dataclass(slots=True)
class CurvePriorsConfig:
    """Prior settings for hierarchical curve parameters."""

    use_data_driven_alpha_center: bool = True
    mu_log_alpha_value: float | None = None
    mu_log_alpha_scale: float = 2.0
    sigma_log_alpha_scale: float = 1.5

    mu_log_beta_value: float = math.log(0.3)
    mu_log_beta_scale: float = 1.5
    sigma_log_beta_scale: float = 1.0

    mu_log_gamma_value: float = 0.0
    mu_log_gamma_scale: float = 1.5
    sigma_log_gamma_scale: float = 1.0

    sigma_log_alpha_cluster_scale: float = 1.0
    sigma_log_beta_cluster_scale: float = 0.8
    sigma_log_gamma_cluster_scale: float = 0.8


@dataclass(slots=True)
class DistributionPriorsConfig:
    """Prior settings for likelihood-specific parameters."""

    gaussian_sigma_scale: float = 20.0
    nb_alpha_sigma: float = 30.0
    zinb_alpha_sigma: float = 20.0
    zinb_logit_psi_mu: float = 0.0
    zinb_logit_psi_sigma: float = 1.5


@dataclass(slots=True)
class TrainingConfig:
    """Training / inference settings."""

    inference_method: InferenceMethod = 'advi'
    advi_steps: int = 15000
    posterior_draws: int = 1000
    progressbar: bool = True
    random_seed: int = 42


@dataclass(slots=True)
class BacktestConfig:
    """Top-level experiment settings."""

    data: DataConfig
    split: SplitConfig
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    semantic: SemanticClusteringConfig = field(default_factory=SemanticClusteringConfig)
    hierarchy: HierarchyConfig = field(default_factory=HierarchyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    curve_priors: CurvePriorsConfig = field(default_factory=CurvePriorsConfig)
    distribution_priors: DistributionPriorsConfig = field(default_factory=DistributionPriorsConfig)
    curves: Sequence[CurveName] = ('log', 'hill')
    likelihoods: Sequence[LikelihoodName] = ('gaussian', 'nb', 'zinb')
