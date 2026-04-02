
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..config import BacktestConfig
from ..curves import get_curve
from ..data.aggregators import aggregate_data
from ..data.splitters import split_train_test
from ..distributions import get_distribution
from ..metrics.regression import evaluate_predictions
from ..models.hierarchy import build_hierarchy_inputs
from ..models.model_builder import build_model
from ..models.predictor import extract_posterior_means, predict_hierarchical_keyword
from ..models.trainer import fit_model
from ..segmentation.pipeline import build_segment_table

REQUIRED_AGG_COLUMNS = {'period_start', 'keyword', 'spend', 'click'}
_CLUSTER_DEFAULTS: dict[str, Any] = {
    'cluster_id': -1,
    'cluster_size': 0,
    'is_clustered': False,
}


def _validate_aggregated_frame(df: pd.DataFrame, frame_name: str) -> None:
    missing = REQUIRED_AGG_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f'{frame_name} is missing required columns: {sorted(missing)}')
    if df['keyword'].isna().any():
        raise ValueError(f'{frame_name} contains null keyword values.')
    if (df['spend'] < 0).any() or (df['click'] < 0).any():
        raise ValueError(f'{frame_name} contains negative spend or click values.')


def _as_config_snapshot(config: BacktestConfig) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    return dict(config)


def normalize_cluster_columns(
    df: pd.DataFrame | None,
    *,
    segment_table: pd.DataFrame | None = None,
    keyword_col: str = 'keyword',
    noise_label: int = -1,
) -> pd.DataFrame | None:
    if df is None:
        return None

    out = df.copy()
    out.attrs = {}
    merge_cols = [keyword_col, 'cluster_id', 'cluster_size', 'is_clustered']
    defaults = {
        'cluster_id': int(noise_label),
        'cluster_size': 0,
        'is_clustered': False,
    }

    if segment_table is not None and keyword_col in out.columns and keyword_col in segment_table.columns:
        lookup = segment_table[[c for c in merge_cols if c in segment_table.columns]].drop_duplicates(subset=[keyword_col]).copy()
        lookup.attrs = {}
        if len(lookup) > 0:
            if any(col not in out.columns for col in lookup.columns if col != keyword_col):
                out = out.merge(lookup, on=keyword_col, how='left', suffixes=('', '_seg'))
            else:
                out = out.merge(lookup, on=keyword_col, how='left', suffixes=('', '_seg'))
                for col in ['cluster_id', 'cluster_size', 'is_clustered']:
                    seg_col = f'{col}_seg'
                    if seg_col in out.columns:
                        out[col] = out[col].combine_first(out[seg_col])
                        out = out.drop(columns=[seg_col])

    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
        out[col] = out[col].fillna(default)

    out['cluster_id'] = out['cluster_id'].astype(int)
    out['cluster_size'] = out['cluster_size'].astype(int)
    out['is_clustered'] = out['is_clustered'].astype(bool)
    return out


def _prepare_segment_frames(
    train_aggregated: pd.DataFrame,
    test_aggregated: pd.DataFrame,
    segment_table: pd.DataFrame,
    segment: str,
    *,
    noise_label: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    segment_cols = ['keyword', 'segment', 'cluster_id', 'cluster_size', 'is_clustered']
    mapping = segment_table[segment_cols].drop_duplicates(subset=['keyword']).copy()
    train_df = train_aggregated.merge(mapping, on='keyword', how='left')
    test_df = test_aggregated.merge(mapping, on='keyword', how='left')
    train_df = normalize_cluster_columns(train_df, segment_table=mapping, noise_label=noise_label)
    test_df = normalize_cluster_columns(test_df, segment_table=mapping, noise_label=noise_label)

    if 'segment' not in test_df.columns:
        test_df['segment'] = 'long_tail'
    else:
        test_df['segment'] = test_df['segment'].fillna('long_tail')

    train_df = train_df.loc[train_df['segment'] == segment].copy()
    test_df = test_df.loc[test_df['segment'] == segment].copy()
    segment_mapping = mapping.loc[mapping['segment'] == segment].copy()
    return train_df, test_df, segment_mapping


def _run_single_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    segment_table: pd.DataFrame,
    curve_name: str,
    likelihood_name: str,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, dict, dict]:
    curve = get_curve(curve_name)
    distribution = get_distribution(likelihood_name)

    train_df = normalize_cluster_columns(
        train_df,
        segment_table=segment_table,
        noise_label=int(config.semantic.cluster_noise_label),
    )
    test_df = normalize_cluster_columns(
        test_df,
        segment_table=segment_table,
        noise_label=int(config.semantic.cluster_noise_label),
    )

    train_df, test_df, hierarchy_inputs = build_hierarchy_inputs(
        train_df=train_df,
        test_df=test_df,
        segment_table=segment_table,
        use_semantic_clustering=config.segmentation.use_semantic_clustering,
        noise_label=config.semantic.cluster_noise_label,
        hierarchy_config=config.hierarchy,
    )

    x_train = train_df['spend'].to_numpy(dtype=np.float32)
    y_train = train_df['click'].to_numpy(dtype=np.float32 if likelihood_name == 'gaussian' else np.int32)
    x_test = test_df['spend'].to_numpy(dtype=np.float32)

    spend_scale = float(np.median(x_train)) if len(x_train) else 1.0
    spend_scale = max(spend_scale, 1e-6)
    x_train_scaled = np.maximum(x_train / spend_scale, 1e-6).astype(np.float32)
    x_test_scaled = np.maximum(x_test / spend_scale, 1e-6).astype(np.float32)

    built = build_model(
        x_scaled=x_train_scaled,
        y_obs=y_train,
        keyword_idx=hierarchy_inputs.train_keyword_idx,
        n_keywords=hierarchy_inputs.n_keywords,
        curve=curve,
        distribution=distribution,
        config=config,
        cluster_idx=hierarchy_inputs.train_cluster_idx,
        n_clusters=hierarchy_inputs.n_clusters,
        keyword_idx_to_cluster_idx=hierarchy_inputs.keyword_idx_to_cluster_idx,
        keyword_prior_scale=hierarchy_inputs.keyword_prior_scale,
    )
    trace = fit_model(built.model, training_config=config.training)
    posterior_means = extract_posterior_means(trace)

    pred_df = predict_hierarchical_keyword(
        curve=curve,
        distribution=distribution,
        x_scaled=x_test_scaled,
        posterior_means=posterior_means,
        hierarchy_inputs=hierarchy_inputs,
        test_df=test_df,
        hierarchy_config=config.hierarchy,
    )
    pred_df = normalize_cluster_columns(
        pred_df,
        segment_table=segment_table,
        noise_label=int(config.semantic.cluster_noise_label),
    )
    pred_df['pred_click'] = pred_df['predicted']
    pred_df['actual_click'] = pred_df['click']
    pred_df['spend_scale'] = spend_scale
    pred_df['test_keyword_idx'] = hierarchy_inputs.test_keyword_idx
    if hierarchy_inputs.test_cluster_idx is not None:
        pred_df['test_cluster_idx'] = hierarchy_inputs.test_cluster_idx
    else:
        pred_df['test_cluster_idx'] = np.nan
    pred_df['test_keyword_train_row_count'] = hierarchy_inputs.test_keyword_train_row_count
    pred_df['min_train_rows_for_keyword_prediction'] = int(config.hierarchy.min_train_rows_for_keyword_prediction)

    metrics = evaluate_predictions(test_df['click'].to_numpy(), pred_df['predicted'].to_numpy())
    hierarchy_diag = {
        'n_keywords': int(hierarchy_inputs.n_keywords),
        'n_clusters': int(hierarchy_inputs.n_clusters),
        'seen_keyword_rows': int(np.isfinite(hierarchy_inputs.test_keyword_idx).sum()),
        'unseen_keyword_rows': int((~np.isfinite(hierarchy_inputs.test_keyword_idx)).sum()),
        'cluster_surrogate_rows': int((pred_df['posterior_source'] == 'cluster_surrogate').sum()),
        'global_surrogate_rows': int((pred_df['posterior_source'] == 'global_surrogate').sum()),
        'keyword_rows': int((pred_df['posterior_source'] == 'keyword').sum()),
    }
    return pred_df, metrics, hierarchy_diag


def _build_segmentation_diagnostics(segment_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_keywords = max(int(segment_table['keyword'].nunique()), 1)
    total_clicks = max(float(segment_table['monetary'].sum()), 1.0)
    for segment, segment_df in segment_table.groupby('segment', dropna=False):
        n_keywords = int(segment_df['keyword'].nunique())
        rows.append({
            'segment': segment,
            'n_keywords': n_keywords,
            'keyword_share': n_keywords / total_keywords,
            'click_share': float(segment_df['monetary'].sum()) / total_clicks,
            'mean_recency': float(segment_df['recency'].mean()),
            'mean_frequency': float(segment_df['frequency'].mean()),
            'clustered_keywords': int(segment_df['is_clustered'].sum()),
            'noise_keywords': int((~segment_df['is_clustered']).sum()),
            'noise_share': float((~segment_df['is_clustered']).mean()) if len(segment_df) else 0.0,
        })
    return pd.DataFrame(rows)


def _cluster_representative_keyword_map(segment_table: pd.DataFrame, noise_label: int) -> dict[int, str]:
    if segment_table.empty or 'cluster_id' not in segment_table.columns:
        return {}
    clustered = segment_table.loc[segment_table['cluster_id'].ne(int(noise_label))].copy()
    if clustered.empty:
        return {}
    clustered['keyword'] = clustered['keyword'].astype(str)
    clustered['keyword_len'] = clustered['keyword'].str.len()
    reps = (
        clustered.sort_values(['cluster_id', 'monetary', 'frequency', 'keyword_len', 'keyword'], ascending=[True, False, False, False, True])
        .groupby('cluster_id', as_index=False)
        .first()[['cluster_id', 'keyword']]
    )
    return {int(row.cluster_id): str(row.keyword) for row in reps.itertuples(index=False)}


def _build_cluster_diagnostics(segment_table: pd.DataFrame, noise_label: int) -> pd.DataFrame:
    clustered = segment_table.loc[segment_table['cluster_id'] != noise_label].copy()
    if len(clustered) == 0:
        return pd.DataFrame([{
            'n_clusters': 0,
            'noise_keywords': int((segment_table['cluster_id'] == noise_label).sum()),
            'noise_share': 1.0,
            'mean_cluster_size': 0.0,
            'max_cluster_size': 0,
        }])

    cluster_sizes = clustered.groupby('cluster_id')['keyword'].nunique()
    rep_map = _cluster_representative_keyword_map(segment_table, noise_label)
    rows = [{
        'n_clusters': int(cluster_sizes.shape[0]),
        'noise_keywords': int((segment_table['cluster_id'] == noise_label).sum()),
        'noise_share': float((segment_table['cluster_id'] == noise_label).mean()),
        'mean_cluster_size': float(cluster_sizes.mean()),
        'max_cluster_size': int(cluster_sizes.max()),
    }]
    detail_rows = []
    for cluster_id, cluster_df in clustered.groupby('cluster_id', dropna=False):
        detail_rows.append({
            'cluster_id': int(cluster_id),
            'segment': cluster_df['segment'].iloc[0] if 'segment' in cluster_df.columns and len(cluster_df) else None,
            'n_keywords': int(cluster_df['keyword'].nunique()),
            'cluster_size': int(cluster_df['cluster_size'].max()) if 'cluster_size' in cluster_df.columns else int(cluster_df['keyword'].nunique()),
            'total_monetary': float(cluster_df['monetary'].sum()) if 'monetary' in cluster_df.columns else float('nan'),
            'representative_keyword': rep_map.get(int(cluster_id), ''),
        })
    summary_df = pd.DataFrame(rows)
    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        return summary_df
    return pd.concat([summary_df, detail_df], ignore_index=True, sort=False)


def _build_pooling_diagnostics(predictions_all: pd.DataFrame) -> pd.DataFrame:
    if predictions_all.empty:
        return pd.DataFrame()
    keyword_counts = predictions_all.groupby('keyword').size()
    rows = [{
        'n_keywords': int(predictions_all['keyword'].nunique()),
        'mean_rows_per_keyword': float(keyword_counts.mean()),
        'sparse_keyword_share_lt_3_rows': float((keyword_counts < 3).mean()),
        'keyword_rows': int((predictions_all['posterior_source'] == 'keyword').sum()),
        'cluster_surrogate_rows': int((predictions_all['posterior_source'] == 'cluster_surrogate').sum()),
        'global_surrogate_rows': int((predictions_all['posterior_source'] == 'global_surrogate').sum()),
    }]
    return pd.DataFrame(rows)


def _build_hierarchy_diagnostics(hierarchy_rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not hierarchy_rows:
        return pd.DataFrame()
    return pd.DataFrame(hierarchy_rows)


def _metric_series_from_group(df: pd.DataFrame) -> pd.Series:
    metrics = evaluate_predictions(df['click'].to_numpy(), df['predicted'].to_numpy())
    return pd.Series(metrics)


def _build_keyword_level(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not predictions:
        return pd.DataFrame()
    rows = []
    for _, pred_df in predictions.items():
        if pred_df.empty:
            continue
        grouped = pred_df.groupby(['run_name', 'segment', 'keyword', 'cluster_id'], dropna=False).agg(
            actual_click=('click', 'sum'),
            pred_click=('pred_click', 'sum'),
            spend=('spend', 'sum'),
            n_rows=('keyword', 'size'),
            mean_keyword_train_row_count=('test_keyword_train_row_count', 'first'),
            clustered=('is_clustered', 'max'),
            posterior_source=('posterior_source', lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]),
        ).reset_index()
        grouped['abs_error'] = (grouped['actual_click'] - grouped['pred_click']).abs()
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _build_cluster_level(predictions: dict[str, pd.DataFrame], segment_table: pd.DataFrame, noise_label: int) -> pd.DataFrame:
    if not predictions:
        return pd.DataFrame()
    rep_map = _cluster_representative_keyword_map(segment_table, noise_label)
    rows = []
    for _, pred_df in predictions.items():
        if pred_df.empty:
            continue
        metric_df = (
            pred_df.groupby(['run_name', 'segment', 'cluster_id'], dropna=False)
            .apply(_metric_series_from_group)
            .reset_index()
        )
        grouped = pred_df.groupby(['run_name', 'segment', 'cluster_id'], dropna=False).agg(
            actual_click=('click', 'sum'),
            pred_click=('pred_click', 'sum'),
            spend=('spend', 'sum'),
            n_rows=('keyword', 'size'),
            n_keywords=('keyword', 'nunique'),
            clustered=('is_clustered', 'max'),
        ).reset_index()
        grouped['is_noise_cluster'] = grouped['cluster_id'].eq(int(noise_label))
        grouped['abs_error'] = (grouped['actual_click'] - grouped['pred_click']).abs()
        grouped['representative_keyword'] = grouped['cluster_id'].map(lambda x: rep_map.get(int(x), '') if pd.notna(x) else '')
        grouped = grouped.merge(metric_df, on=['run_name', 'segment', 'cluster_id'], how='left')
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _build_posterior_source_level(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not predictions:
        return pd.DataFrame()
    rows = []
    for _, pred_df in predictions.items():
        if pred_df.empty:
            continue
        metric_df = (
            pred_df.groupby(['run_name', 'segment', 'posterior_source'], dropna=False)
            .apply(_metric_series_from_group)
            .reset_index()
        )
        grouped = pred_df.groupby(['run_name', 'segment', 'posterior_source'], dropna=False).agg(
            n_rows=('keyword', 'size'),
            n_keywords=('keyword', 'nunique'),
            actual_click=('click', 'sum'),
            pred_click=('pred_click', 'sum'),
        ).reset_index()
        grouped['abs_error'] = (grouped['actual_click'] - grouped['pred_click']).abs()
        grouped = grouped.merge(metric_df, on=['run_name', 'segment', 'posterior_source'], how='left')
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _build_overall_run_level(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not predictions:
        return pd.DataFrame()
    rows = []
    for run_name, pred_df in predictions.items():
        if pred_df.empty:
            continue
        metrics = evaluate_predictions(pred_df['click'].to_numpy(), pred_df['predicted'].to_numpy())
        rows.append({
            'run_name': run_name,
            'segment': 'overall',
            'n_rows': int(len(pred_df)),
            'n_keywords': int(pred_df['keyword'].nunique()),
            'actual_click': float(pred_df['click'].sum()),
            'pred_click': float(pred_df['predicted'].sum()),
            **metrics,
        })
    return pd.DataFrame(rows)


def _concat_named_frames(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    ordered = [df for _, df in sorted(frames.items()) if isinstance(df, pd.DataFrame) and len(df) > 0]
    if not ordered:
        return pd.DataFrame()
    return pd.concat(ordered, ignore_index=True)


def run_backtest_suite(raw_df: pd.DataFrame, config: BacktestConfig) -> dict:
    aggregated_df = aggregate_data(raw_df, config.data)
    _validate_aggregated_frame(aggregated_df, 'aggregated_df')
    train_aggregated, test_aggregated = split_train_test(aggregated_df, config.split)
    _validate_aggregated_frame(train_aggregated, 'train_aggregated')
    _validate_aggregated_frame(test_aggregated, 'test_aggregated')

    segment_table = build_segment_table(train_aggregated, config)
    segment_table = normalize_cluster_columns(
        segment_table,
        segment_table=segment_table,
        noise_label=int(config.semantic.cluster_noise_label),
    )
    cluster_rep_map = _cluster_representative_keyword_map(segment_table, int(config.semantic.cluster_noise_label))
    if cluster_rep_map:
        segment_table['cluster_representative_keyword'] = segment_table['cluster_id'].map(lambda x: cluster_rep_map.get(int(x), '') if pd.notna(x) and int(x) != int(config.semantic.cluster_noise_label) else '')
    else:
        segment_table['cluster_representative_keyword'] = ''
    semantic_diags = segment_table.attrs.get('semantic_diagnostics', pd.DataFrame())
    diagnostics: dict[str, pd.DataFrame] = {
        'segmentation': _build_segmentation_diagnostics(segment_table),
        'clusters': _build_cluster_diagnostics(segment_table, int(config.semantic.cluster_noise_label)),
        'semantic_runtime': semantic_diags if isinstance(semantic_diags, pd.DataFrame) else pd.DataFrame(),
    }

    summary_rows: list[dict[str, Any]] = []
    predictions: dict[str, pd.DataFrame] = {}
    metrics_map: dict[str, pd.DataFrame] = {}
    hierarchy_rows: list[dict[str, Any]] = []

    for segment in ['head', 'long_tail']:
        train_df, test_df, segment_mapping = _prepare_segment_frames(
            train_aggregated,
            test_aggregated,
            segment_table,
            segment,
            noise_label=int(config.semantic.cluster_noise_label),
        )
        if len(train_df) == 0 or len(test_df) == 0:
            continue

        for curve_name in config.curves:
            for likelihood_name in config.likelihoods:
                run_name = f'{segment}__semantic_{int(config.segmentation.use_semantic_clustering)}__{config.data.aggregation_level}__{curve_name}__{likelihood_name}'
                pred_df, metric_dict, hierarchy_diag = _run_single_model(
                    train_df=train_df,
                    test_df=test_df,
                    segment_table=segment_mapping,
                    curve_name=curve_name,
                    likelihood_name=likelihood_name,
                    config=config,
                )
                pred_df['segment'] = segment
                pred_df['run_name'] = run_name
                pred_df['cluster_representative_keyword'] = pred_df['cluster_id'].map(lambda x: cluster_rep_map.get(int(x), '') if pd.notna(x) and int(x) != int(config.semantic.cluster_noise_label) else '')
                summary_rows.append({
                    'run_name': run_name,
                    'segment': segment,
                    'use_semantic_clustering': config.segmentation.use_semantic_clustering,
                    'semantic_apply_to_segments': ','.join(config.segmentation.semantic_apply_to_segments),
                    'aggregation_level': config.data.aggregation_level,
                    'curve': curve_name,
                    'likelihood': likelihood_name,
                    **metric_dict,
                })
                hierarchy_rows.append({'run_name': run_name, 'segment': segment, **hierarchy_diag})
                predictions[run_name] = pred_df
                metrics_map[run_name] = pd.DataFrame([metric_dict])

    summary = pd.DataFrame(summary_rows)
    overall_run_level = _build_overall_run_level(predictions)
    if len(summary) > 0:
        summary = pd.concat([summary, overall_run_level], ignore_index=True, sort=False)
        summary = summary.sort_values(['run_name', 'segment']).reset_index(drop=True)

    train_df = normalize_cluster_columns(
        train_aggregated,
        segment_table=segment_table,
        noise_label=int(config.semantic.cluster_noise_label),
    )
    test_df = normalize_cluster_columns(
        test_aggregated,
        segment_table=segment_table,
        noise_label=int(config.semantic.cluster_noise_label),
    )
    predictions_all = _concat_named_frames(predictions)
    diagnostics['pooling'] = _build_pooling_diagnostics(predictions_all)
    diagnostics['hierarchy'] = _build_hierarchy_diagnostics(hierarchy_rows)
    keyword_level = _build_keyword_level(predictions)
    cluster_level = _build_cluster_level(predictions, segment_table, int(config.semantic.cluster_noise_label))
    posterior_source_level = _build_posterior_source_level(predictions)

    return {
        'summary': summary,
        'predictions_all': predictions_all,
        'metrics': metrics_map,
        'segment_table': segment_table,
        'train': train_df,
        'test': test_df,
        'train_aggregated': train_aggregated,
        'test_aggregated': test_aggregated,
        'keyword_level': keyword_level,
        'cluster_level': cluster_level,
        'posterior_source_level': posterior_source_level,
        'diagnostics': diagnostics,
        'config_snapshot': _as_config_snapshot(config),
    }
