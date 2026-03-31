from __future__ import annotations

from typing import Any

import numpy as np
import pymc as pm

from ..distributions.base import BaseDistribution


class GaussianDistribution(BaseDistribution):
    name = 'gaussian'

    def add_likelihood(self, name: str, mu, y_obs, config: Any):
        sigma = pm.HalfNormal('sigma', sigma=config.distribution_priors.gaussian_sigma_scale)
        return pm.Normal(name, mu=mu, sigma=sigma, observed=y_obs)

    def postprocess_prediction(self, mu_pred: np.ndarray, posterior_means: dict[str, Any]) -> np.ndarray:
        return np.maximum(mu_pred, 0.0)
