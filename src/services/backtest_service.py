from __future__ import annotations

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
from ..models.predictor import extract_posterior_means, predict_with_fallback
from ..models.trainer import fit_model
from ..segmentation.pipeline import build_segment_table


REQUIRED_AGG_COLUMNS = {'period_start', 'keyword', 'spend', 'click'}


def _validate_aggregated_frame(df: pd.DataFrame, frame_name: str) -> None:
    missing = REQUIRED_AGG_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f'{frame_name} is missing required columns: {sorted(missing)}')
    if df['keyword'].isna().any():
        raise ValueError(f'{frame_name} contains null keyword values.')
    if (df['spend'] < 0).any() or (df['click'] < 0).any():
        raise ValueError(f'{frame_name} contains negative spend or click values.')


def _prepare_segment_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    segment_table: pd.DataFrame,
    segment_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    segment_cols = ['keyword', 'segment', 'cluster_id', 'cluster_size', 'is_clustered']
    train_seg = train_df.merge(segment_table[segment_cols], on='keyword', how='left')
    test_seg = test_df.merge(segment_table[segment_cols], on='keyword', how='left')
    train_seg = train_seg.loc[train_seg['segment'] == segment_name].copy()
    test_seg = test_seg.loc[test_seg['segment'] == segment_name].copy()
    return train_seg, test_seg


def _run_single_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    segment_table: pd.DataFrame,
    curve_name: str,
    likelihood_name: str,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, dict, dict]:
    curve = get_curve(curve_name)
    dist = get_distribution(likelihood_name)

    train_df, test_df, hierarchy = build_hierarchy_inputs(
        train_df=train_df,
        test_df=test_df,
        segment_table=segment_table,
        use_semantic_clustering=config.segmentation.use_semantic_clustering,
        noise_label=config.semantic.cluster_noise_label,
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
        keyword_idx=hierarchy.keyword_idx_train,
        n_keywords=hierarchy.n_keywords,
        curve=curve,
        distribution=dist,
        config=config,
        cluster_idx=hierarchy.cluster_idx_train,
        n_clusters=hierarchy.n_clusters,
        keyword_to_cluster_idx=hierarchy.keyword_to_cluster_idx,
    )
    trace = fit_model(built.model, training_config=config.training)
    posterior_means = extract_posterior_means(trace)

    y_pred, prediction_levels = predict_with_fallback(
        curve=curve,
        distribution=dist,
        x_scaled=x_test_scaled,
        keyword_idx=hierarchy.keyword_idx_test,
        cluster_idx=hierarchy.cluster_idx_test,
        keyword_train_counts=hierarchy.keyword_train_counts_test,
        posterior_means=posterior_means,
        use_semantic_clustering=config.segmentation.use_semantic_clustering,
        hierarchy_config=config.hierarchy,
    )

    pred_df = test_df.copy()
    pred_df['pred_click'] = y_pred
    pred_df['spend_scale'] = spend_scale
    pred_df['prediction_level'] = prediction_levels
    pred_df['keyword_train_count'] = hierarchy.keyword_train_counts_test
    metrics = evaluate_predictions(test_df['click'].to_numpy(), y_pred)
    total = max(len(prediction_levels), 1)
    fallback = {
        'prediction_level_keyword': int((prediction_levels == 'keyword').sum()),
        'prediction_level_cluster': int((prediction_levels == 'cluster').sum()),
        'prediction_level_global': int((prediction_levels == 'global').sum()),
        'prediction_level_keyword_share': float((prediction_levels == 'keyword').sum()) / total,
        'prediction_level_cluster_share': float((prediction_levels == 'cluster').sum()) / total,
        'prediction_level_global_share': float((prediction_levels == 'global').sum()) / total,
        'n_test_rows': int(len(prediction_levels)),
    }
    return pred_df, metrics, fallback


def _build_segmentation_diagnostics(segment_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_keywords = max(int(segment_table['keyword'].nunique()), 1)
    total_clicks = max(float(segment_table['monetary'].sum()), 1.0)
    for seg_name, seg_df in segment_table.groupby('segment', dropna=False):
        n_keywords = int(seg_df['keyword'].nunique())
        rows.append({
            'segment': seg_name,
            'n_keywords': n_keywords,
            'keyword_share': n_keywords / total_keywords,
            'click_share': float(seg_df['monetary'].sum()) / total_clicks,
            'mean_recency': float(seg_df['recency'].mean()),
            'mean_frequency': float(seg_df['frequency'].mean()),
            'clustered_keywords': int(seg_df['is_clustered'].sum()),
            'noise_keywords': int((~seg_df['is_clustered']).sum()),
            'noise_share': float((~seg_df['is_clustered']).mean()) if len(seg_df) else 0.0,
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


def run_backtest_suite(raw_df: pd.DataFrame, config: BacktestConfig) -> dict:
    aggregated_df = aggregate_data(raw_df, config.data)
    _validate_aggregated_frame(aggregated_df, 'aggregated_df')
    train_df, test_df = split_train_test(aggregated_df, config.split)
    _validate_aggregated_frame(train_df, 'train_df')
    _validate_aggregated_frame(test_df, 'test_df')

    segment_table = build_segment_table(train_df, config)
    semantic_diags = segment_table.attrs.get('semantic_diagnostics', pd.DataFrame())
    diagnostics = {
        'segmentation': _build_segmentation_diagnostics(segment_table),
        'clusters': _build_cluster_diagnostics(segment_table, int(config.semantic.cluster_noise_label)),
        'semantic_runtime': semantic_diags if isinstance(semantic_diags, pd.DataFrame) else pd.DataFrame(),
        'fallback': pd.DataFrame(),
    }

    summary_rows: list[dict] = []
    predictions: dict[str, pd.DataFrame] = {}
    metrics_map: dict[str, pd.DataFrame] = {}
    fallback_rows: list[dict] = []

    for segment_name in ['head', 'long_tail']:
        train_seg, test_seg = _prepare_segment_frames(train_df, test_df, segment_table, segment_name)
        if len(train_seg) == 0 or len(test_seg) == 0:
            continue

        segment_mapping = segment_table.loc[segment_table['segment'] == segment_name].copy()

        for curve_name in config.curves:
            for likelihood_name in config.likelihoods:
                run_name = f'{segment_name}__semantic_{int(config.segmentation.use_semantic_clustering)}__{config.data.aggregation_level}__{curve_name}__{likelihood_name}'
                pred_df, metric_dict, fallback_dict = _run_single_model(
                    train_df=train_seg,
                    test_df=test_seg,
                    segment_table=segment_mapping,
                    curve_name=curve_name,
                    likelihood_name=likelihood_name,
                    config=config,
                )
                pred_df['segment'] = segment_name
                pred_df['run_name'] = run_name
                summary_rows.append({
                    'run_name': run_name,
                    'segment': segment_name,
                    'use_semantic_clustering': config.segmentation.use_semantic_clustering,
                    'semantic_apply_to_segments': ','.join(config.segmentation.semantic_apply_to_segments),
                    'aggregation_level': config.data.aggregation_level,
                    'curve': curve_name,
                    'likelihood': likelihood_name,
                    **metric_dict,
                })
                fallback_rows.append({'run_name': run_name, 'segment': segment_name, **fallback_dict})
                predictions[run_name] = pred_df
                metrics_map[run_name] = pd.DataFrame([metric_dict])

    summary = pd.DataFrame(summary_rows)
    if len(summary) > 0:
        summary = summary.sort_values(['rmse', 'mae']).reset_index(drop=True)
    diagnostics['fallback'] = pd.DataFrame(fallback_rows)

    return {
        'summary': summary,
        'predictions': predictions,
        'metrics': metrics_map,
        'segments': segment_table,
        'train_aggregated': train_df,
        'test_aggregated': test_df,
        'diagnostics': diagnostics,
        'config_snapshot': config,
    }
