from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymc as pm

from ..config import BacktestConfig
from ..curves.base import BaseCurve
from ..distributions.base import BaseDistribution


@dataclass(slots=True)
class BuiltModel:
    model: pm.Model
    curve_name: str
    distribution_name: str


def build_model(
    x_scaled: np.ndarray,
    y_obs: np.ndarray,
    keyword_idx: np.ndarray,
    n_keywords: int,
    curve: BaseCurve,
    distribution: BaseDistribution,
    config: BacktestConfig,
    cluster_idx: np.ndarray | None = None,
    n_clusters: int = 0,
    keyword_to_cluster_idx: np.ndarray | None = None,
) -> BuiltModel:
    with pm.Model() as model:
        mu = curve.build_mu(
            x_scaled=x_scaled,
            keyword_idx=keyword_idx,
            n_keywords=n_keywords,
            y_obs=y_obs,
            config=config,
            cluster_idx=cluster_idx,
            n_clusters=n_clusters,
            keyword_to_cluster_idx=keyword_to_cluster_idx,
        )
        distribution.add_likelihood('y_obs', mu=mu, y_obs=y_obs, config=config)
    return BuiltModel(model=model, curve_name=curve.name, distribution_name=distribution.name)
