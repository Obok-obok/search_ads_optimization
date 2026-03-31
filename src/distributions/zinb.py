from __future__ import annotations

from typing import Any

import numpy as np
import pymc as pm

from ..distributions.base import BaseDistribution


class ZINBDistribution(BaseDistribution):
    name = 'zinb'

    def add_likelihood(self, name: str, mu, y_obs, config: Any):
        priors = config.distribution_priors
        phi = pm.HalfNormal('phi', sigma=priors.zinb_alpha_sigma)
        logit_psi = pm.Normal('logit_psi', mu=priors.zinb_logit_psi_mu, sigma=priors.zinb_logit_psi_sigma)
        psi = pm.Deterministic('psi', pm.math.sigmoid(logit_psi))
        return pm.ZeroInflatedNegativeBinomial(name, psi=psi, mu=mu, alpha=phi, observed=y_obs)

    def postprocess_prediction(self, mu_pred: np.ndarray, posterior_means: dict[str, Any]) -> np.ndarray:
        psi = float(posterior_means.get('psi', 1.0))
        return np.maximum(psi * mu_pred, 0.0)
