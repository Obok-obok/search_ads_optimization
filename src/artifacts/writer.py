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
    'pooling': 'diag_pooling',
    'hierarchy': 'diag_hierarchy',
}


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj



def _ensure_export_frame(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    out = df.copy()
    if 'cluster_id' in out.columns:
        out['cluster_id'] = out['cluster_id'].fillna(-1).astype(int)
    if 'cluster_size' in out.columns:
        out['cluster_size'] = out['cluster_size'].fillna(0).astype(int)
    if 'is_clustered' in out.columns:
        out['is_clustered'] = out['is_clustered'].fillna(False).astype(bool)
    return out



def save_backtest_suite(result: dict, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = _ensure_export_frame(result.get('summary'))
    segment_table = _ensure_export_frame(result.get('segment_table'))
    train_aggregated = _ensure_export_frame(result.get('train_aggregated'))
    test_aggregated = _ensure_export_frame(result.get('test_aggregated'))
    train_df = _ensure_export_frame(result.get('train'))
    test_df = _ensure_export_frame(result.get('test'))
    predictions_all = _ensure_export_frame(result.get('predictions_all'))
    keyword_level = _ensure_export_frame(result.get('keyword_level'))
    cluster_level = _ensure_export_frame(result.get('cluster_level'))

    if summary is not None:
        summary.to_csv(output_path / 'summary.csv', index=False, encoding='utf-8-sig')
    if segment_table is not None:
        segment_table.to_csv(output_path / 'segment_table.csv', index=False, encoding='utf-8-sig')
    if train_aggregated is not None:
        train_aggregated.to_csv(output_path / 'train_aggregated.csv', index=False, encoding='utf-8-sig')
    if test_aggregated is not None:
        test_aggregated.to_csv(output_path / 'test_aggregated.csv', index=False, encoding='utf-8-sig')
    if train_df is not None:
        train_df.to_csv(output_path / 'train.csv', index=False, encoding='utf-8-sig')
    if test_df is not None:
        test_df.to_csv(output_path / 'test.csv', index=False, encoding='utf-8-sig')
    if predictions_all is not None and len(predictions_all) > 0:
        predictions_all.to_csv(output_path / 'predictions_all.csv', index=False, encoding='utf-8-sig')
    if keyword_level is not None and len(keyword_level) > 0:
        keyword_level.to_csv(output_path / 'keyword_level.csv', index=False, encoding='utf-8-sig')
    if cluster_level is not None and len(cluster_level) > 0:
        cluster_level.to_csv(output_path / 'cluster_level.csv', index=False, encoding='utf-8-sig')

    diagnostics = result.get('diagnostics', {}) or {}
    diagnostics_dir = output_path / 'diagnostics'
    diagnostics_dir.mkdir(exist_ok=True)
    for diag_name, diag_df in diagnostics.items():
        if isinstance(diag_df, pd.DataFrame):
            diag_df = _ensure_export_frame(diag_df)
            diag_df.to_csv(diagnostics_dir / f'{diag_name}.csv', index=False, encoding='utf-8-sig')
            if len(diag_df) > 0:
                diag_df.to_csv(output_path / f'diagnostics_{diag_name}.csv', index=False, encoding='utf-8-sig')

    config_snapshot = _to_jsonable(result.get('config_snapshot', {}))
    with open(output_path / 'config_snapshot.json', 'w', encoding='utf-8') as fp:
        json.dump(config_snapshot, fp, ensure_ascii=False, indent=2)

    with pd.ExcelWriter(output_path / 'backtest_outputs.xlsx', engine='openpyxl') as writer:
        if summary is not None:
            summary.to_excel(writer, sheet_name='summary', index=False)
        if segment_table is not None:
            segment_table.to_excel(writer, sheet_name='segment_table', index=False)
        if train_aggregated is not None:
            train_aggregated.to_excel(writer, sheet_name='train_aggregated', index=False)
        if test_aggregated is not None:
            test_aggregated.to_excel(writer, sheet_name='test_aggregated', index=False)
        if train_df is not None:
            train_df.to_excel(writer, sheet_name='train', index=False)
        if test_df is not None:
            test_df.to_excel(writer, sheet_name='test', index=False)
        if predictions_all is not None and len(predictions_all) > 0:
            predictions_all.to_excel(writer, sheet_name='predictions_all', index=False)
        if keyword_level is not None and len(keyword_level) > 0:
            keyword_level.to_excel(writer, sheet_name='keyword_level', index=False)
        if cluster_level is not None and len(cluster_level) > 0:
            cluster_level.to_excel(writer, sheet_name='cluster_level', index=False)
        for diag_name, sheet_name in DIAGNOSTIC_SHEETS.items():
            diag_df = diagnostics.get(diag_name)
            if isinstance(diag_df, pd.DataFrame) and len(diag_df) > 0:
                diag_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
