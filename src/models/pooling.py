from __future__ import annotations

import numpy as np

from ..config import HierarchyConfig


_MIN_SIGMA_MULTIPLIER = 0.35
_MAX_SIGMA_MULTIPLIER = 3.0


def compute_keyword_prior_scale(
    keyword_train_count: np.ndarray | None,
    keyword_is_long_tail: np.ndarray | None,
    hierarchy_config: HierarchyConfig,
    n_keywords: int,
) -> np.ndarray:
    """Return per-keyword prior sigma multipliers.

    Larger multipliers mean *less* pooling because keyword-level priors are allowed
    to move farther from their parent level. Smaller multipliers mean stronger
    pooling toward the parent.

    The schedule is intentionally conservative:
    - sparse keywords shrink more strongly,
    - dense keywords keep more freedom,
    - long-tail keywords can be pooled more strongly via the configured multiplier.
    """
    strength = max(float(hierarchy_config.keyword_pooling_strength), 1e-6)
    long_tail_multiplier = max(float(hierarchy_config.long_tail_pooling_multiplier), 1e-6)

    if keyword_train_count is None or len(keyword_train_count) == 0:
        keyword_train_count = np.ones(n_keywords, dtype=float)
    else:
        keyword_train_count = np.maximum(np.asarray(keyword_train_count, dtype=float), 1.0)

    median_count = max(float(np.median(keyword_train_count)), 1.0)
    density_scale = np.sqrt(keyword_train_count / median_count)
    density_scale = np.clip(density_scale, _MIN_SIGMA_MULTIPLIER, _MAX_SIGMA_MULTIPLIER)

    sigma_multiplier = density_scale / strength

    if keyword_is_long_tail is not None and len(keyword_is_long_tail) == n_keywords:
        long_tail_mask = np.asarray(keyword_is_long_tail, dtype=bool)
        sigma_multiplier = np.where(long_tail_mask, sigma_multiplier / long_tail_multiplier, sigma_multiplier)

    return np.clip(sigma_multiplier, _MIN_SIGMA_MULTIPLIER, _MAX_SIGMA_MULTIPLIER).astype(float)
