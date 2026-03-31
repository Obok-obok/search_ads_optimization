from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DIAGNOSTIC_SHEETS = {
    'segmentation': 'diag_segmentation',
    'clusters': 'diag_clusters',
    'semantic_runtime': 'diag_semantic_runtime',
    'fallback': 'diag_fallback',
}
_CLUSTER_DEFAULTS = {
    'cluster_id': -1,
    'cluster_size': 0,
    'is_clustered': False,
}


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _ensure_cluster_columns_for_export(df: pd.DataFrame | None, segment_table: pd.DataFrame | None = None) -> pd.DataFrame | None:
    if df is None:
        return None
    out = df.copy()
    keyword_col = 'keyword'

    for col, default in _CLUSTER_DEFAULTS.items():
        if col not in out.columns:
            if segment_table is not None and keyword_col in out.columns and keyword_col in segment_table.columns and col in segment_table.columns:
                lookup = segment_table[[keyword_col, col]].drop_duplicates(subset=[keyword_col])
                out = out.merge(lookup, on=keyword_col, how='left')
            else:
                out[col] = default

    out['cluster_id'] = out['cluster_id'].fillna(_CLUSTER_DEFAULTS['cluster_id']).astype(int)
    out['cluster_size'] = out['cluster_size'].fillna(_CLUSTER_DEFAULTS['cluster_size']).astype(int)
    out['is_clustered'] = out['is_clustered'].where(out['is_clustered'].notna(), _CLUSTER_DEFAULTS['is_clustered']).astype(bool)
    return out


def save_backtest_suite(result: dict, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    segment_table = _ensure_cluster_columns_for_export(result.get('segments') or result.get('segment_table'))
    summary = result['summary']
    train_aggregated = _ensure_cluster_columns_for_export(result['train_aggregated'], segment_table)
    test_aggregated = _ensure_cluster_columns_for_export(result['test_aggregated'], segment_table)

    summary.to_csv(output_path / 'summary.csv', index=False, encoding='utf-8-sig')
    if segment_table is not None:
        segment_table.to_csv(output_path / 'segments.csv', index=False, encoding='utf-8-sig')
        segment_table.to_csv(output_path / 'segment_table.csv', index=False, encoding='utf-8-sig')
    if train_aggregated is not None:
        train_aggregated.to_csv(output_path / 'train_aggregated.csv', index=False, encoding='utf-8-sig')
    if test_aggregated is not None:
        test_aggregated.to_csv(output_path / 'test_aggregated.csv', index=False, encoding='utf-8-sig')

    for name, pred_df in result['predictions'].items():
        export_df = _ensure_cluster_columns_for_export(pred_df, segment_table)
        export_df.to_csv(output_path / f'{name}.csv', index=False, encoding='utf-8-sig')

    diagnostics = result.get('diagnostics', {})
    for diag_name, diag_df in diagnostics.items():
        if isinstance(diag_df, pd.DataFrame) and len(diag_df) > 0:
            diag_df.to_csv(output_path / f'diagnostics_{diag_name}.csv', index=False, encoding='utf-8-sig')

    config_snapshot = _to_jsonable(result.get('config_snapshot', {}))
    with open(output_path / 'config_snapshot.json', 'w', encoding='utf-8') as fp:
        json.dump(config_snapshot, fp, ensure_ascii=False, indent=2)

    with pd.ExcelWriter(output_path / 'backtest_outputs.xlsx', engine='openpyxl') as writer:
        summary.to_excel(writer, sheet_name='summary', index=False)
        if segment_table is not None:
            segment_table.to_excel(writer, sheet_name='segments', index=False)
        if train_aggregated is not None:
            train_aggregated.to_excel(writer, sheet_name='train_aggregated', index=False)
        if test_aggregated is not None:
            test_aggregated.to_excel(writer, sheet_name='test_aggregated', index=False)
        for diag_name, sheet_name in DIAGNOSTIC_SHEETS.items():
            diag_df = diagnostics.get(diag_name)
            if isinstance(diag_df, pd.DataFrame) and len(diag_df) > 0:
                diag_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        for name, pred_df in result['predictions'].items():
            export_df = _ensure_cluster_columns_for_export(pred_df, segment_table)
            export_df.to_excel(writer, sheet_name=name[:31], index=False)
