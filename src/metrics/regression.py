
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _coerce_eval_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    return y_true_arr[mask], y_pred_arr[mask]


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true_arr, y_pred_arr = _coerce_eval_arrays(y_true, y_pred)
    if y_true_arr.size == 0:
        return {
            'mae': float('nan'),
            'rmse': float('nan'),
            'r2': float('nan'),
            'mean_error_rate': float('nan'),
        }

    abs_err = np.abs(y_true_arr - y_pred_arr)
    denom = np.maximum(np.asarray(y_true_arr, dtype=float), 1.0)
    unique_true = np.unique(y_true_arr)
    r2 = float('nan') if unique_true.size < 2 else float(r2_score(y_true_arr, y_pred_arr))
    return {
        'mae': float(mean_absolute_error(y_true_arr, y_pred_arr)),
        'rmse': float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        'r2': r2,
        'mean_error_rate': float(np.mean(abs_err / denom)),
    }
