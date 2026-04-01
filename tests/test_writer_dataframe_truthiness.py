from pathlib import Path

import pandas as pd

from src.artifacts.writer import save_backtest_suite


def test_save_backtest_suite_handles_dataframe_frames_without_boolean_eval(tmp_path: Path):
    result = {
        "summary": pd.DataFrame([{"curve": "log", "likelihood": "gaussian", "rmse": 1.0, "mae": 0.5}]),
        "segment_table": pd.DataFrame([{"keyword": "a", "segment": "head", "cluster_id": -1, "cluster_size": 0, "is_clustered": False}]),
        "train": pd.DataFrame([{"keyword": "a", "click": 1}]),
        "test": pd.DataFrame([{"keyword": "a", "click": 2}]),
        "train_aggregated": pd.DataFrame([{"keyword": "a", "click": 1}]),
        "test_aggregated": pd.DataFrame([{"keyword": "a", "click": 2}]),
        "predictions_all": pd.DataFrame([{"keyword": "a", "actual": 2, "predicted": 1.5, "error": 0.5, "abs_error": 0.5}]),
        "keyword_level": pd.DataFrame([{"keyword": "a", "cluster_id": -1, "actual": 2, "predicted": 1.5}]),
        "cluster_level": pd.DataFrame([{"cluster_id": -1, "n_keywords": 1, "actual": 2, "predicted": 1.5}]),
        "diagnostics": {
            "segmentation": pd.DataFrame([{"segment": "head", "n_keywords": 1}]),
            "clusters": pd.DataFrame([{"n_clusters": 0, "noise_share": 1.0}]),
        },
        "config_snapshot": {"test": True},
    }

    save_backtest_suite(result, str(tmp_path))

    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "segment_table.csv").exists()
    assert (tmp_path / "diagnostics" / "segmentation.csv").exists()
    assert (tmp_path / "backtest_outputs.xlsx").exists()
