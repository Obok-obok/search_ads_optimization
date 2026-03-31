from .gaussian import GaussianDistribution
from .negative_binomial import NegativeBinomialDistribution
from .zinb import ZINBDistribution

LIKELIHOOD_REGISTRY = {
    'gaussian': GaussianDistribution,
    'nb': NegativeBinomialDistribution,
    'zinb': ZINBDistribution,
}


def get_distribution(name: str):
    if name not in LIKELIHOOD_REGISTRY:
        raise ValueError(f"Unknown likelihood '{name}'. Available: {list(LIKELIHOOD_REGISTRY.keys())}")
    return LIKELIHOOD_REGISTRY[name]()
