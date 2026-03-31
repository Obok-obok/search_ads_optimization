from __future__ import annotations

import pandas as pd

from ..config import SegmentationConfig


def build_rfm_table(df: pd.DataFrame, config: SegmentationConfig) -> pd.DataFrame:
    max_period = df['period_start'].max()
    rfm = (
        df.groupby('keyword', as_index=False)
        .agg(
            total_click=('click', 'sum'),
            n_periods=('period_start', 'nunique'),
            last_period=('period_start', 'max'),
            avg_spend=('spend', 'mean'),
        )
        .sort_values('total_click', ascending=False)
        .reset_index(drop=True)
    )
    total_click = float(rfm['total_click'].sum()) if len(rfm) else 1.0
    total_periods = max(int(df['period_start'].nunique()), 1)
    rfm['recency_periods'] = ((max_period - rfm['last_period']).dt.days).astype(int)
    rfm['frequency_periods'] = rfm['n_periods']
    rfm['monetary'] = rfm['total_click']
    rfm['M_share'] = rfm['total_click'] / total_click
    rfm['M_cum_share'] = rfm['M_share'].cumsum()
    cutoff = config.monetary_click_share_cutoff
    rfm['is_top_click_group'] = rfm['M_cum_share'] <= cutoff
    over_idx = rfm.index[rfm['M_cum_share'] > cutoff]
    if len(over_idx) > 0:
        rfm.loc[over_idx.min(), 'is_top_click_group'] = True
    # display-friendly aliases
    rfm['recency'] = rfm['recency_periods']
    rfm['frequency'] = rfm['frequency_periods'] / total_periods
    return rfm


def apply_rfm_head_tail(rfm: pd.DataFrame, config: SegmentationConfig) -> pd.DataFrame:
    out = rfm.copy()
    recency_th = out['recency'].quantile(config.recency_quantile)
    freq_th = out['frequency'].quantile(config.frequency_quantile)
    out['segment'] = 'long_tail'
    out.loc[
        (out['recency'] <= recency_th) &
        (out['frequency'] >= freq_th) &
        (out['is_top_click_group']),
        'segment',
    ] = 'head'
    return out
