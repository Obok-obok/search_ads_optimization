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

    result['summary'].to_csv(output_path / 'summary.csv', index=False, encoding='utf-8-sig')
    result['segments'].to_csv(output_path / 'segments.csv', index=False, encoding='utf-8-sig')
    result['train_aggregated'].to_csv(output_path / 'train_aggregated.csv', index=False, encoding='utf-8-sig')
    result['test_aggregated'].to_csv(output_path / 'test_aggregated.csv', index=False, encoding='utf-8-sig')
    for name, pred_df in result['predictions'].items():
        pred_df.to_csv(output_path / f'{name}.csv', index=False, encoding='utf-8-sig')

    diagnostics = result.get('diagnostics', {})
    for diag_name, diag_df in diagnostics.items():
        if isinstance(diag_df, pd.DataFrame) and len(diag_df) > 0:
            diag_df.to_csv(output_path / f'diagnostics_{diag_name}.csv', index=False, encoding='utf-8-sig')

    config_snapshot = _to_jsonable(result.get('config_snapshot', {}))
    with open(output_path / 'config_snapshot.json', 'w', encoding='utf-8') as fp:
        json.dump(config_snapshot, fp, ensure_ascii=False, indent=2)

    with pd.ExcelWriter(output_path / 'backtest_outputs.xlsx', engine='openpyxl') as writer:
        result['summary'].to_excel(writer, sheet_name='summary', index=False)
        result['segments'].to_excel(writer, sheet_name='segments', index=False)
        result['train_aggregated'].to_excel(writer, sheet_name='train_aggregated', index=False)
        result['test_aggregated'].to_excel(writer, sheet_name='test_aggregated', index=False)
        for diag_name, sheet_name in DIAGNOSTIC_SHEETS.items():
            diag_df = diagnostics.get(diag_name)
            if isinstance(diag_df, pd.DataFrame) and len(diag_df) > 0:
                diag_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        for name, pred_df in result['predictions'].items():
            pred_df.to_excel(writer, sheet_name=name[:31], index=False)
