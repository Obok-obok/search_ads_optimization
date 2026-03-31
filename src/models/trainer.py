from __future__ import annotations

import pymc as pm

from ..config import TrainingConfig


def fit_model(model: pm.Model, training_config: TrainingConfig):
    if training_config.inference_method != 'advi':
        raise ValueError(f"Unsupported inference_method: {training_config.inference_method}")

    with model:
        approx = pm.fit(
            n=training_config.advi_steps,
            method='advi',
            progressbar=training_config.progressbar,
            random_seed=training_config.random_seed,
        )
        trace = approx.sample(training_config.posterior_draws, random_seed=training_config.random_seed)
    return trace
