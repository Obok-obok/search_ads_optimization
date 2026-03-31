from __future__ import annotations

from typing import Any

import numpy as np
import pymc as pm

from ..distributions.base import BaseDistribution


class NegativeBinomialDistribution(BaseDistribution):
    name = 'nb'

    def add_likelihood(self, name: str, mu, y_obs, config: Any):
        phi = pm.HalfNormal('phi', sigma=config.distribution_priors.nb_alpha_sigma)
        return pm.NegativeBinomial(name, mu=mu, alpha=phi, observed=y_obs)

    def postprocess_prediction(self, mu_pred: np.ndarray, posterior_means: dict[str, Any]) -> np.ndarray:
        return np.maximum(mu_pred, 0.0)
