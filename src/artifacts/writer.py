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



def _pick_frame(result: dict[str, Any], *keys: str) -> pd.DataFrame | None:
    for key in keys:
        value = result.get(key)
        if isinstance(value, pd.DataFrame):
            return _ensure_export_frame(value)
    return None



def _save_csv(df: pd.DataFrame | None, path: Path) -> None:
    if df is not None:
        df.to_csv(path, index=False, encoding='utf-8-sig')



def save_backtest_suite(result: dict[str, Any], output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    diagnostics_dir = output_path / 'diagnostics'
    diagnostics_dir.mkdir(exist_ok=True)
    predictions_dir = output_path / 'predictions'
    predictions_dir.mkdir(exist_ok=True)

    summary = _pick_frame(result, 'summary')
    train_df = _pick_frame(result, 'train')
    test_df = _pick_frame(result, 'test')
    train_aggregated = _pick_frame(result, 'train_aggregated')
    test_aggregated = _pick_frame(result, 'test_aggregated')
    segment_table = _pick_frame(result, 'segment_table', 'segments')
    predictions_all = _pick_frame(result, 'predictions_all')
    keyword_level = _pick_frame(result, 'keyword_level')
    cluster_level = _pick_frame(result, 'cluster_level')

    _save_csv(summary, output_path / 'summary.csv')
    _save_csv(segment_table, output_path / 'segment_table.csv')
    _save_csv(train_aggregated, output_path / 'train_aggregated.csv')
    _save_csv(test_aggregated, output_path / 'test_aggregated.csv')
    _save_csv(train_df, output_path / 'train.csv')
    _save_csv(test_df, output_path / 'test.csv')
    _save_csv(predictions_all, output_path / 'predictions_all.csv')
    _save_csv(keyword_level, output_path / 'keyword_level.csv')
    _save_csv(cluster_level, output_path / 'cluster_level.csv')

    if predictions_all is not None and not predictions_all.empty:
        _save_csv(predictions_all, predictions_dir / 'predictions_all.csv')

    diagnostics_obj = result.get('diagnostics')
    diagnostics: dict[str, pd.DataFrame] = {}
    if isinstance(diagnostics_obj, dict):
        for diag_name, diag_value in diagnostics_obj.items():
            if isinstance(diag_value, pd.DataFrame):
                diagnostics[diag_name] = _ensure_export_frame(diag_value)

    for diag_name, diag_df in diagnostics.items():
        _save_csv(diag_df, diagnostics_dir / f'{diag_name}.csv')
        if diag_df is not None and not diag_df.empty:
            _save_csv(diag_df, output_path / f'diagnostics_{diag_name}.csv')

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
        if predictions_all is not None and not predictions_all.empty:
            predictions_all.to_excel(writer, sheet_name='predictions_all', index=False)
        if keyword_level is not None and not keyword_level.empty:
            keyword_level.to_excel(writer, sheet_name='keyword_level', index=False)
        if cluster_level is not None and not cluster_level.empty:
            cluster_level.to_excel(writer, sheet_name='cluster_level', index=False)

        for diag_name, sheet_name in DIAGNOSTIC_SHEETS.items():
            diag_df = diagnostics.get(diag_name)
            if diag_df is not None and not diag_df.empty:
                diag_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
