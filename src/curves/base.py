from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseCurve(ABC):
    name: str

    @abstractmethod
    def build_mu(
        self,
        x_scaled: np.ndarray,
        keyword_idx: np.ndarray,
        n_keywords: int,
        y_obs: np.ndarray,
        config: Any,
        cluster_idx: np.ndarray | None = None,
        n_clusters: int = 0,
        keyword_to_cluster_idx: np.ndarray | None = None,
    ):
        raise NotImplementedError

    @abstractmethod
    def predict_level_numpy(
        self,
        x_scaled: np.ndarray,
        posterior_means: dict[str, Any],
        level: str,
        indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute predictions using one explicit posterior level only.

        Args:
            x_scaled: Scaled spend values.
            posterior_means: Posterior mean dictionary.
            level: One of ``global``, ``cluster`` or ``keyword``.
            indices: Integer indices for cluster/keyword levels. Omit for global.
        """
        raise NotImplementedError
