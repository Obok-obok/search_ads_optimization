from __future__ import annotations

import pandas as pd

from ..config import SplitConfig


def _resolve_time_col(df: pd.DataFrame) -> str:
    if 'period_start' in df.columns:
        return 'period_start'
    if 'date' in df.columns:
        return 'date'
    raise KeyError("Expected either 'period_start' or 'date' in dataframe.")


def split_train_test(df: pd.DataFrame, split_config: SplitConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split aggregated or raw data by configurable date ranges."""
    time_col = _resolve_time_col(df)
    train_start = pd.Timestamp(split_config.train_start)
    train_end = pd.Timestamp(split_config.train_end)
    test_start = pd.Timestamp(split_config.test_start)
    test_end = pd.Timestamp(split_config.test_end)

    train_mask = (df[time_col] >= train_start) & (df[time_col] <= train_end)
    test_mask = (df[time_col] >= test_start) & (df[time_col] <= test_end)
    return df.loc[train_mask].copy(), df.loc[test_mask].copy()
