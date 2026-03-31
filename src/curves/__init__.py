from .hill_curve import HillCurve
from .log_curve import LogCurve

CURVE_REGISTRY = {
    'log': LogCurve,
    'hill': HillCurve,
}


def get_curve(name: str):
    if name not in CURVE_REGISTRY:
        raise ValueError(f"Unknown curve '{name}'. Available curves: {list(CURVE_REGISTRY.keys())}")
    return CURVE_REGISTRY[name]()
