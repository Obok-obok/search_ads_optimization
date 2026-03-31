from __future__ import annotations

import pandas as pd

from ..config import DataConfig


_WEEKDAY_MAP = {
    'MON': 'W-MON',
    'TUE': 'W-TUE',
    'WED': 'W-WED',
    'THU': 'W-THU',
    'FRI': 'W-FRI',
    'SAT': 'W-SAT',
    'SUN': 'W-SUN',
}


REQUIRED_INPUT_COLUMNS = {'date', 'keyword', 'spend', 'click'}


def _validate_input(df: pd.DataFrame) -> None:
    missing = REQUIRED_INPUT_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f'Raw dataframe is missing required columns: {sorted(missing)}')
    if df['date'].isna().any():
        raise ValueError('Raw dataframe contains null date values.')
    if df['keyword'].isna().any():
        raise ValueError('Raw dataframe contains null keyword values.')


def aggregate_daily(df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    out = df.copy().sort_values(['date', 'keyword']).reset_index(drop=True)
    out['period_start'] = out['date']
    out['days_observed'] = 1
    out['coverage_ratio'] = 1.0
    return out[['period_start', 'keyword', 'spend', 'click', 'days_observed', 'coverage_ratio']]


def aggregate_weekly(df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    rule = _WEEKDAY_MAP[config.week_anchor]
    tmp = df.copy()
    tmp['period_start'] = tmp['date'].dt.to_period(rule).dt.start_time

    weekly = (
        tmp.groupby(['period_start', 'keyword'], as_index=False)
        .agg(
            spend=('spend', 'sum'),
            click=('click', 'sum'),
            days_observed=('date', 'nunique'),
        )
        .sort_values(['period_start', 'keyword'])
        .reset_index(drop=True)
    )
    weekly['coverage_ratio'] = weekly['days_observed'] / float(config.weekly_expected_days)

    policy = config.weekly_partial_policy
    if policy == 'drop':
        weekly = weekly.loc[weekly['days_observed'] >= config.weekly_expected_days].copy()
    elif policy == 'normalize':
        valid = weekly['days_observed'] > 0
        weekly.loc[valid, 'spend'] = weekly.loc[valid, 'spend'] / weekly.loc[valid, 'days_observed'] * config.weekly_expected_days
        weekly.loc[valid, 'click'] = weekly.loc[valid, 'click'] / weekly.loc[valid, 'days_observed'] * config.weekly_expected_days
    elif policy == 'keep':
        pass
    else:
        raise ValueError(f'Unknown weekly_partial_policy: {policy}')

    return weekly[['period_start', 'keyword', 'spend', 'click', 'days_observed', 'coverage_ratio']]


def aggregate_data(df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    _validate_input(df)
    if config.aggregation_level == 'daily':
        return aggregate_daily(df, config)
    if config.aggregation_level == 'weekly':
        return aggregate_weekly(df, config)
    raise ValueError(f'Unknown aggregation_level: {config.aggregation_level}')
