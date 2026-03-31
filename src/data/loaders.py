from __future__ import annotations

import pandas as pd

from ..config import DataConfig


REQUIRED_RAW_COLUMNS = {'date', 'keyword', 'spend', 'click'}


def load_keyword_data(path: str, config: DataConfig) -> pd.DataFrame:
    """Load raw daily keyword data with safe encoding fallback."""
    for enc in ('cp949', 'euc-kr', 'utf-8-sig', 'utf-8'):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise ValueError(f'Unable to read file: {path}')

    df.columns = df.columns.str.strip()
    date_col = config.date_col
    keyword_col = config.keyword_col
    spend_col = config.spend_col
    click_col = config.click_col
    missing = {date_col, keyword_col, spend_col, click_col}.difference(df.columns)
    if missing:
        raise ValueError(f'Input file is missing required columns: {sorted(missing)}')

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[keyword_col] = df[keyword_col].astype(str).str.strip()
    df[spend_col] = pd.to_numeric(df[spend_col], errors='coerce')
    df[click_col] = pd.to_numeric(df[click_col], errors='coerce')

    df = df.dropna(subset=[date_col, keyword_col, spend_col, click_col]).copy()
    df = df[df[keyword_col].ne('')].copy()
    df = df[(df[spend_col] > 0) & (df[click_col] >= 0)].copy()
    df = df.rename(columns={
        date_col: 'date',
        keyword_col: 'keyword',
        spend_col: 'spend',
        click_col: 'click',
    })
    out = df[['date', 'keyword', 'spend', 'click']].sort_values(['date', 'keyword']).reset_index(drop=True)
    missing_standard = REQUIRED_RAW_COLUMNS.difference(out.columns)
    if missing_standard:
        raise ValueError(f'Standardized output is missing required columns: {sorted(missing_standard)}')
    return out
