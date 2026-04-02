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
    'cluster_representative_keyword': None,
    'topic': 'unknown',
    'intent': 'general',
    'routing_key': None,
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
    merge_cols = [keyword_col, 'cluster_id', 'cluster_size', 'is_clustered', 'cluster_representative_keyword', 'topic', 'intent', 'routing_key']
    defaults = {
        'cluster_id': int(noise_label),
        'cluster_size': 0,
        'is_clustered': False,
        'cluster_representative_keyword': None,
        'topic': 'unknown',
        'intent': 'general',
        'routing_key': None,
    }

    if segment_table is not None and keyword_col in out.columns and keyword_col in segment_table.columns:
        lookup = segment_table[[c for c in merge_cols if c in segment_table.columns]].drop_duplicates(subset=[keyword_col]).copy()
        lookup.attrs = {}
        if len(lookup) > 0:
            out = out.merge(lookup, on=keyword_col, how='left', suffixes=('', '_seg'))
            for col in ['cluster_id', 'cluster_size', 'is_clustered', 'cluster_representative_keyword', 'topic', 'intent', 'routing_key']:
                seg_col = f'{col}_seg'
                if seg_col in out.columns:
                    if col in out.columns:
                        out[col] = out[col].combine_first(out[seg_col])
                    else:
                        out[col] = out[seg_col]
                    out = out.drop(columns=[seg_col])

    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
        if default is None:
            out[col] = out[col].where(out[col].notna(), None)
        else:
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
    segment_cols = ['keyword', 'segment', 'cluster_id', 'cluster_size', 'is_clustered', 'cluster_representative_keyword', 'topic', 'intent', 'routing_key']
    available_cols = [c for c in segment_cols if c in segment_table.columns]
    mapping = segment_table[available_cols].drop_duplicates(subset=['keyword']).copy()
    train_df = train_aggregated.merge(mapping, on='keyword', how='left')
    test_df = test_aggregated.merge(mapping, on='keyword', how='left')
    train_df = normalize_cluster_columns(train_df, segment_table=mapping, noise_label=noise_label)
    test_df = normalize_cluster_columns(test_df, segment_table=mapping, noise_label=noise_label)

    if 'segment' not in test_df.columns:
        test_df['segment'] = 'long_tail'
    else:
        test_df['segment'] = test_df['segment'].fillna('long_tail')
    if 'routing_key' not in test_df.columns:
        test_df['routing_key'] = None
    if 'topic' not in test_df.columns:
        test_df['topic'] = 'unknown'
    if 'intent' not in test_df.columns:
        test_df['intent'] = 'general'
    unseen_mask = test_df['routing_key'].isna() & test_df['segment'].eq('long_tail')
    test_df.loc[unseen_mask, 'topic'] = test_df.loc[unseen_mask, 'topic'].replace('', 'unknown').fillna('unknown')
    test_df.loc[unseen_mask, 'intent'] = test_df.loc[unseen_mask, 'intent'].replace('', 'general').fillna('general')
    test_df.loc[unseen_mask, 'routing_key'] = (
        test_df.loc[unseen_mask, 'segment'].astype(str) + '__' +
        test_df.loc[unseen_mask, 'topic'].astype(str) + '__' +
        test_df.loc[unseen_mask, 'intent'].astype(str)
    )

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

    train_df = normalize_cluster_columns(train_df, segment_table=segment_table, noise_label=int(config.semantic.cluster_noise_label))
    test_df = normalize_cluster_columns(test_df, segment_table=segment_table, noise_label=int(config.semantic.cluster_noise_label))

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
    pred_df = normalize_cluster_columns(pred_df, segment_table=segment_table, noise_label=int(config.semantic.cluster_noise_label))
    pred_df['pred_click'] = pred_df['predicted']
    pred_df['actual_click'] = pred_df['click']
    pred_df['spend_scale'] = spend_scale
    pred_df['test_keyword_idx'] = hierarchy_inputs.test_keyword_idx
    pred_df['test_cluster_idx'] = hierarchy_inputs.test_cluster_idx if hierarchy_inputs.test_cluster_idx is not None else np.nan
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
            'n_routing_keys': int(segment_df['routing_key'].nunique()) if 'routing_key' in segment_df.columns else 0,
        })
    return pd.DataFrame(rows)


def _build_cluster_diagnostics(segment_table: pd.DataFrame, noise_label: int) -> pd.DataFrame:
    clustered = segment_table.loc[segment_table['cluster_id'] != noise_label].copy()
    if len(clustered) == 0:
        return pd.DataFrame([{
            'n_clusters': 0,
            'noise_keywords': int((segment_table['cluster_id'] == noise_label).sum()),
            'noise_share': 1.0,
            'mean_cluster_size': 0.0,
            'max_cluster_size': 0,
            'cluster_status': 'disabled_or_all_noise',
        }])
    cluster_sizes = clustered.groupby('cluster_id')['keyword'].nunique()
    noise_share = float((segment_table['cluster_id'] == noise_label).mean())
    return pd.DataFrame([{
        'n_clusters': int(cluster_sizes.shape[0]),
        'noise_keywords': int((segment_table['cluster_id'] == noise_label).sum()),
        'noise_share': noise_share,
        'mean_cluster_size': float(cluster_sizes.mean()),
        'max_cluster_size': int(cluster_sizes.max()),
        'cluster_status': 'ok' if noise_share < 1.0 else 'disabled_or_all_noise',
    }])


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


def _build_keyword_level(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not predictions:
        return pd.DataFrame()
    rows = []
    for _, pred_df in predictions.items():
        if pred_df.empty:
            continue
        grouped = pred_df.groupby(['run_name', 'run_group_name', 'segment', 'keyword', 'cluster_id'], dropna=False).agg(
            actual_click=('click', 'sum'),
            pred_click=('pred_click', 'sum'),
            spend=('spend', 'sum'),
            n_rows=('keyword', 'size'),
            mean_keyword_train_row_count=('test_keyword_train_row_count', 'first'),
            clustered=('is_clustered', 'max'),
            representative_keyword=('cluster_representative_keyword', 'first'),
            topic=('topic', 'first'),
            intent=('intent', 'first'),
            routing_key=('routing_key', 'first'),
        ).reset_index()
        grouped['abs_error'] = (grouped['actual_click'] - grouped['pred_click']).abs()
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _build_cluster_level(predictions: dict[str, pd.DataFrame], noise_label: int) -> pd.DataFrame:
    if not predictions:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, pred_df in predictions.items():
        if pred_df.empty:
            continue
        for keys, group in pred_df.groupby(['run_name', 'run_group_name', 'segment', 'cluster_id'], dropna=False):
            run_name, run_group_name, segment, cluster_id = keys
            metrics = evaluate_predictions(group['click'].to_numpy(), group['pred_click'].to_numpy())
            rows.append({
                'run_name': run_name,
                'run_group_name': run_group_name,
                'segment': segment,
                'cluster_id': int(cluster_id),
                'representative_keyword': group['cluster_representative_keyword'].dropna().iloc[0] if group['cluster_representative_keyword'].notna().any() else None,
                'topic': group['topic'].dropna().iloc[0] if group['topic'].notna().any() else 'unknown',
                'intent': group['intent'].dropna().iloc[0] if group['intent'].notna().any() else 'general',
                'routing_key': group['routing_key'].dropna().iloc[0] if group['routing_key'].notna().any() else None,
                'actual_click': float(group['click'].sum()),
                'pred_click': float(group['pred_click'].sum()),
                'spend': float(group['spend'].sum()),
                'n_rows': int(len(group)),
                'n_keywords': int(group['keyword'].nunique()),
                'clustered': bool(group['is_clustered'].max()),
                'is_noise_cluster': int(cluster_id) == int(noise_label),
                **metrics,
            })
    return pd.DataFrame(rows)


def _build_posterior_source_level(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not predictions:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, pred_df in predictions.items():
        if pred_df.empty:
            continue
        for keys, group in pred_df.groupby(['run_name', 'run_group_name', 'segment', 'posterior_source'], dropna=False):
            run_name, run_group_name, segment, posterior_source = keys
            metrics = evaluate_predictions(group['click'].to_numpy(), group['pred_click'].to_numpy())
            rows.append({
                'run_name': run_name,
                'run_group_name': run_group_name,
                'segment': segment,
                'posterior_source': posterior_source,
                'actual_click': float(group['click'].sum()),
                'pred_click': float(group['pred_click'].sum()),
                'spend': float(group['spend'].sum()),
                'n_rows': int(len(group)),
                'n_keywords': int(group['keyword'].nunique()),
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


def _build_overall_summary(predictions_all: pd.DataFrame) -> pd.DataFrame:
    if predictions_all.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ['run_group_name', 'use_semantic_clustering', 'semantic_apply_to_segments', 'aggregation_level', 'curve', 'likelihood']
    for keys, group in predictions_all.groupby(group_cols, dropna=False):
        metrics = evaluate_predictions(group['click'].to_numpy(), group['pred_click'].to_numpy())
        run_group_name, use_semantic_clustering, semantic_apply_to_segments, aggregation_level, curve, likelihood = keys
        rows.append({
            'run_name': f'overall__{run_group_name}',
            'run_group_name': run_group_name,
            'segment': 'overall',
            'use_semantic_clustering': use_semantic_clustering,
            'semantic_apply_to_segments': semantic_apply_to_segments,
            'aggregation_level': aggregation_level,
            'curve': curve,
            'likelihood': likelihood,
            **metrics,
        })
    return pd.DataFrame(rows)


def run_backtest_suite(raw_df: pd.DataFrame, config: BacktestConfig) -> dict:
    aggregated_df = aggregate_data(raw_df, config.data)
    _validate_aggregated_frame(aggregated_df, 'aggregated_df')
    train_aggregated, test_aggregated = split_train_test(aggregated_df, config.split)
    _validate_aggregated_frame(train_aggregated, 'train_aggregated')
    _validate_aggregated_frame(test_aggregated, 'test_aggregated')

    segment_table = build_segment_table(train_aggregated, config)
    diagnostics = {
        'segmentation': _build_segmentation_diagnostics(segment_table),
        'clusters': _build_cluster_diagnostics(segment_table, int(config.semantic.cluster_noise_label)),
    }
    semantic_diags = segment_table.attrs.get('semantic_diagnostics')
    diagnostics['semantic_runtime'] = semantic_diags if isinstance(semantic_diags, pd.DataFrame) else pd.DataFrame()

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
                run_group_name = f'semantic_{int(config.segmentation.use_semantic_clustering)}__{config.data.aggregation_level}__{curve_name}__{likelihood_name}'
                run_name = f'{segment}__{run_group_name}'
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
                pred_df['run_group_name'] = run_group_name
                pred_df['use_semantic_clustering'] = bool(config.segmentation.use_semantic_clustering)
                pred_df['semantic_apply_to_segments'] = ','.join(config.segmentation.semantic_apply_to_segments)
                pred_df['aggregation_level'] = config.data.aggregation_level
                pred_df['curve'] = curve_name
                pred_df['likelihood'] = likelihood_name
                summary_rows.append({
                    'run_name': run_name,
                    'run_group_name': run_group_name,
                    'segment': segment,
                    'use_semantic_clustering': config.segmentation.use_semantic_clustering,
                    'semantic_apply_to_segments': ','.join(config.segmentation.semantic_apply_to_segments),
                    'aggregation_level': config.data.aggregation_level,
                    'curve': curve_name,
                    'likelihood': likelihood_name,
                    **metric_dict,
                })
                hierarchy_rows.append({'run_name': run_name, 'run_group_name': run_group_name, 'segment': segment, **hierarchy_diag})
                predictions[run_name] = pred_df
                metrics_map[run_name] = pd.DataFrame([metric_dict])

    summary = pd.DataFrame(summary_rows)
    train_df = normalize_cluster_columns(train_aggregated, segment_table=segment_table, noise_label=int(config.semantic.cluster_noise_label))
    test_df = normalize_cluster_columns(test_aggregated, segment_table=segment_table, noise_label=int(config.semantic.cluster_noise_label))
    predictions_all = _concat_named_frames(predictions)
    overall_summary = _build_overall_summary(predictions_all)
    if len(overall_summary) > 0:
        summary = pd.concat([summary, overall_summary], ignore_index=True)
    if len(summary) > 0:
        summary = summary.sort_values(['segment', 'rmse', 'mae']).reset_index(drop=True)

    diagnostics['pooling'] = _build_pooling_diagnostics(predictions_all)
    diagnostics['hierarchy'] = _build_hierarchy_diagnostics(hierarchy_rows)
    keyword_level = _build_keyword_level(predictions)
    cluster_level = _build_cluster_level(predictions, int(config.semantic.cluster_noise_label))
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
