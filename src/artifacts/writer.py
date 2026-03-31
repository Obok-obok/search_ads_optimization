
import os

def _ensure_cluster_id_for_export(df, segment_table=None, keyword_col="keyword", noise_label=-1):
    if df is None:
        return df
    out = df.copy()
    if "cluster_id" not in out.columns:
        if segment_table is not None and keyword_col in out.columns:
            out = out.merge(
                segment_table[[keyword_col, "cluster_id"]].drop_duplicates(),
                on=keyword_col,
                how="left",
            )
        else:
            out["cluster_id"] = noise_label
    out["cluster_id"] = out["cluster_id"].fillna(noise_label).astype(int)
    return out


def save_backtest_suite(result, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    segment_table = result.get("segment_table")

    for key in ["train", "test"]:
        df = _ensure_cluster_id_for_export(result.get(key), segment_table)
        if df is not None:
            df.to_csv(os.path.join(output_dir, f"{key}.csv"), index=False)
