from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseDistribution(ABC):
    name: str

    @abstractmethod
    def add_likelihood(self, name: str, mu, y_obs, config: Any):
        raise NotImplementedError

    @abstractmethod
    def postprocess_prediction(self, mu_pred: np.ndarray, posterior_means: dict[str, Any]) -> np.ndarray:
        raise NotImplementedError
