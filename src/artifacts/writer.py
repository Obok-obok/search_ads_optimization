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


def _resolve_segment_table(result: dict) -> pd.DataFrame | None:
    segment_table = result.get('segment_table')
    if segment_table is None:
        segment_table = result.get('segments')
    return segment_table


def _ensure_cluster_columns_for_export(
    df: pd.DataFrame | None,
    segment_table: pd.DataFrame | None = None,
    keyword_col: str = 'keyword',
    noise_label: int = -1,
) -> pd.DataFrame | None:
    if df is None:
        return None

    out = df.copy()

    if segment_table is not None and keyword_col in out.columns and keyword_col in segment_table.columns:
        merge_cols = [keyword_col] + [col for col in ('cluster_id', 'cluster_size', 'is_clustered') if col in segment_table.columns]
        if len(merge_cols) > 1:
            lookup = segment_table[merge_cols].drop_duplicates(subset=[keyword_col])
            missing = [col for col in ('cluster_id', 'cluster_size', 'is_clustered') if col not in out.columns]
            if missing:
                out = out.merge(lookup, on=keyword_col, how='left')

    if 'cluster_id' not in out.columns:
        out['cluster_id'] = noise_label
    if 'cluster_size' not in out.columns:
        out['cluster_size'] = 0
    if 'is_clustered' not in out.columns:
        out['is_clustered'] = False

    out['cluster_id'] = out['cluster_id'].fillna(noise_label).astype(int)
    out['cluster_size'] = out['cluster_size'].fillna(0).astype(int)
    out['is_clustered'] = out['is_clustered'].fillna(False).astype(bool)
    return out


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def save_backtest_suite(result: dict, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_df = result['summary'].copy()
    segment_table = _resolve_segment_table(result)
    segment_table = _ensure_cluster_columns_for_export(segment_table, segment_table)
    train_df = _ensure_cluster_columns_for_export(result['train_aggregated'], segment_table)
    test_df = _ensure_cluster_columns_for_export(result['test_aggregated'], segment_table)

    summary_df.to_csv(output_path / 'summary.csv', index=False, encoding='utf-8-sig')
    if segment_table is not None:
        segment_table.to_csv(output_path / 'segments.csv', index=False, encoding='utf-8-sig')
        segment_table.to_csv(output_path / 'segment_table.csv', index=False, encoding='utf-8-sig')
    train_df.to_csv(output_path / 'train_aggregated.csv', index=False, encoding='utf-8-sig')
    test_df.to_csv(output_path / 'test_aggregated.csv', index=False, encoding='utf-8-sig')

    for name, pred_df in result['predictions'].items():
        pred_df = _ensure_cluster_columns_for_export(pred_df, segment_table)
        pred_df.to_csv(output_path / f'{name}.csv', index=False, encoding='utf-8-sig')

    diagnostics = result.get('diagnostics', {})
    for diag_name, diag_df in diagnostics.items():
        if isinstance(diag_df, pd.DataFrame) and len(diag_df) > 0:
            diag_df.to_csv(output_path / f'diagnostics_{diag_name}.csv', index=False, encoding='utf-8-sig')

    config_snapshot = _to_jsonable(result.get('config_snapshot', {}))
    with open(output_path / 'config_snapshot.json', 'w', encoding='utf-8') as fp:
        json.dump(config_snapshot, fp, ensure_ascii=False, indent=2)

    with pd.ExcelWriter(output_path / 'backtest_outputs.xlsx', engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='summary', index=False)
        if segment_table is not None:
            segment_table.to_excel(writer, sheet_name='segments', index=False)
        train_df.to_excel(writer, sheet_name='train_aggregated', index=False)
        test_df.to_excel(writer, sheet_name='test_aggregated', index=False)
        for diag_name, sheet_name in DIAGNOSTIC_SHEETS.items():
            diag_df = diagnostics.get(diag_name)
            if isinstance(diag_df, pd.DataFrame) and len(diag_df) > 0:
                diag_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        for name, pred_df in result['predictions'].items():
            pred_df = _ensure_cluster_columns_for_export(pred_df, segment_table)
            pred_df.to_excel(writer, sheet_name=name[:31], index=False)
