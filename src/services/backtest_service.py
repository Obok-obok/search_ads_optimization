
def _ensure_cluster_id(df, segment_table, keyword_col="keyword", noise_label=-1):
    out = df.copy()
    if "cluster_id" not in out.columns:
        if keyword_col in out.columns and keyword_col in segment_table.columns:
            out = out.merge(
                segment_table[[keyword_col, "cluster_id"]].drop_duplicates(),
                on=keyword_col,
                how="left",
            )
        else:
            out["cluster_id"] = noise_label
    out["cluster_id"] = out["cluster_id"].fillna(noise_label).astype(int)
    return out


def run_backtest_suite(raw_df, config):
    from src.data.aggregators import aggregate_data
    from src.data.splitters import split_train_test
    from src.segmentation.pipeline import build_segment_table
    from src.models.hierarchy import build_hierarchy_inputs

    agg_df = aggregate_data(raw_df, config.data)
    train_df, test_df = split_train_test(agg_df, config.split)

    segment_table = build_segment_table(train_df, config)

    train_df = _ensure_cluster_id(train_df, segment_table)
    test_df = _ensure_cluster_id(test_df, segment_table)

    hierarchy_inputs = build_hierarchy_inputs(
        train_df=train_df,
        test_df=test_df,
        segment_table=segment_table,
        use_semantic_clustering=config.segmentation.use_semantic_clustering,
        noise_label=-1,
    )

    return {
        "train": train_df,
        "test": test_df,
        "segment_table": segment_table,
        "hierarchy_inputs": hierarchy_inputs,
    }
