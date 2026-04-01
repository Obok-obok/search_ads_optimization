from __future__ import annotations

from typing import Any

import numpy as np
import pymc as pm

from ..curves.base import BaseCurve


class LogCurve(BaseCurve):
    name = 'log'

    def build_mu(
        self,
        x_scaled: np.ndarray,
        keyword_idx: np.ndarray,
        n_keywords: int,
        y_obs: np.ndarray,
        config: Any,
        cluster_idx: np.ndarray | None = None,
        n_clusters: int = 0,
        keyword_idx_to_cluster_idx: np.ndarray | None = None,
        keyword_prior_scale: np.ndarray | None = None,
    ):
        priors = config.curve_priors
        alpha_center = (
            float(np.log(max(np.percentile(y_obs, 75) + 1.0, 1.0)))
            if priors.use_data_driven_alpha_center or priors.mu_log_alpha_value is None
            else float(priors.mu_log_alpha_value)
        )

        mu_log_alpha_global = pm.Normal('mu_log_alpha_global', mu=alpha_center, sigma=priors.mu_log_alpha_scale)
        sigma_log_alpha_global = pm.HalfNormal('sigma_log_alpha_global', sigma=priors.sigma_log_alpha_scale)
        mu_log_beta_global = pm.Normal('mu_log_beta_global', mu=priors.mu_log_beta_value, sigma=priors.mu_log_beta_scale)
        sigma_log_beta_global = pm.HalfNormal('sigma_log_beta_global', sigma=priors.sigma_log_beta_scale)

        has_valid_cluster = (
            keyword_idx_to_cluster_idx is not None
            and n_clusters > 0
            and np.any(keyword_idx_to_cluster_idx >= 0)
        )

        keyword_prior_scale = np.ones(n_keywords, dtype=float) if keyword_prior_scale is None else np.asarray(keyword_prior_scale, dtype=float)
        alpha_keyword_sigma = priors.alpha_keyword_scale() * keyword_prior_scale
        beta_keyword_sigma = priors.beta_keyword_scale() * keyword_prior_scale

        if has_valid_cluster:
            sigma_log_alpha_cluster = pm.HalfNormal('sigma_log_alpha_cluster', sigma=priors.alpha_global_to_cluster_scale())
            sigma_log_beta_cluster = pm.HalfNormal('sigma_log_beta_cluster', sigma=priors.beta_global_to_cluster_scale())

            log_alpha_cluster = pm.Normal('log_alpha_cluster', mu=mu_log_alpha_global, sigma=sigma_log_alpha_cluster, shape=n_clusters)
            log_beta_cluster = pm.Normal('log_beta_cluster', mu=mu_log_beta_global, sigma=sigma_log_beta_cluster, shape=n_clusters)

            safe_cluster_idx = np.where(keyword_idx_to_cluster_idx >= 0, keyword_idx_to_cluster_idx, 0).astype(int)
            valid_cluster_mask = (keyword_idx_to_cluster_idx >= 0)

            parent_alpha_cluster = log_alpha_cluster[safe_cluster_idx]
            parent_beta_cluster = log_beta_cluster[safe_cluster_idx]

            parent_alpha = pm.math.switch(valid_cluster_mask, parent_alpha_cluster, mu_log_alpha_global)
            parent_beta = pm.math.switch(valid_cluster_mask, parent_beta_cluster, mu_log_beta_global)

            log_alpha_k = pm.Normal('log_alpha_k', mu=parent_alpha, sigma=alpha_keyword_sigma, shape=n_keywords)
            log_beta_k = pm.Normal('log_beta_k', mu=parent_beta, sigma=beta_keyword_sigma, shape=n_keywords)

            pm.Deterministic('alpha_cluster', pm.math.exp(log_alpha_cluster))
            pm.Deterministic('beta_cluster', pm.math.exp(log_beta_cluster))
        else:
            log_alpha_k = pm.Normal('log_alpha_k', mu=mu_log_alpha_global, sigma=alpha_keyword_sigma, shape=n_keywords)
            log_beta_k = pm.Normal('log_beta_k', mu=mu_log_beta_global, sigma=beta_keyword_sigma, shape=n_keywords)

        alpha_k = pm.Deterministic('alpha_k', pm.math.exp(log_alpha_k))
        beta_k = pm.Deterministic('beta_k', pm.math.exp(log_beta_k))
        pm.Deterministic('alpha_global', pm.math.exp(mu_log_alpha_global))
        pm.Deterministic('beta_global', pm.math.exp(mu_log_beta_global))

        mu = alpha_k[keyword_idx] * pm.math.log(1.0 + beta_k[keyword_idx] * x_scaled)
        return pm.math.maximum(mu, 1e-6)

    def predict_level_numpy(
        self,
        x_scaled: np.ndarray,
        posterior_means: dict[str, Any],
        level: str,
        indices: np.ndarray | None = None,
    ) -> np.ndarray:
        if level == 'global':
            alpha = float(posterior_means['alpha_global'])
            beta = float(posterior_means['beta_global'])
            return np.maximum(alpha * np.log1p(beta * x_scaled), 0.0)
        if level == 'cluster':
            if indices is None:
                raise ValueError('cluster prediction requires indices')
            alpha = np.asarray(posterior_means['alpha_cluster'])[indices]
            beta = np.asarray(posterior_means['beta_cluster'])[indices]
            return np.maximum(alpha * np.log1p(beta * x_scaled), 0.0)
        if level == 'keyword':
            if indices is None:
                raise ValueError('keyword prediction requires indices')
            alpha = np.asarray(posterior_means['alpha_k'])[indices]
            beta = np.asarray(posterior_means['beta_k'])[indices]
            return np.maximum(alpha * np.log1p(beta * x_scaled), 0.0)
        raise ValueError(f'Unknown prediction level: {level}')
