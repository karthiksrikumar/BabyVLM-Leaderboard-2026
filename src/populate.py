import json
import os

import pandas as pd

from src.display.formatting import make_clickable_model
from src.display.utils import AutoEvalColumn, EvalQueueColumn
from src.leaderboard.read_evals import get_raw_eval_results


def get_leaderboard_df(results_path: str, requests_path: str, cols: list, benchmark_cols: list) -> pd.DataFrame:
    """Creates a dataframe from all the individual experiment results"""
    raw_data = get_raw_eval_results(results_path, requests_path)
    all_data_json = [v.to_dict() for v in raw_data]
    for item in all_data_json:
        item["Track"] = item["eval_name"].split("_")[-1]

    df = pd.DataFrame.from_records(all_data_json)
    if not df.empty:
        df = df.sort_values(by=[AutoEvalColumn.overall_average.name], ascending=False)
        # round numeric benchmark columns for display
        present_bench_cols = [c for c in benchmark_cols if c in df.columns]
        df[present_bench_cols] = df[present_bench_cols].round(decimals=2)
        if AutoEvalColumn.overall_average.name in df.columns:
            df[AutoEvalColumn.overall_average.name] = df[AutoEvalColumn.overall_average.name].round(decimals=2)
        # Keep rows with partial results — missing benchmarks are shown as NaN.
    return df


def get_evaluation_queue_df(save_path: str, cols: list) -> list[pd.DataFrame]:
    """Creates the different dataframes for the evaluation queue requests"""
    if not os.path.isdir(save_path):
        empty = pd.DataFrame(columns=cols)
        return empty, empty, empty

    entries = [entry for entry in os.listdir(save_path) if not entry.startswith(".")]
    all_evals = []

    for entry in entries:
        if entry.endswith(".json"):
            file_path = os.path.join(save_path, entry)
            with open(file_path) as fp:
                data = json.load(fp)
            _fill_queue_row(data)
            all_evals.append(data)
        elif not entry.endswith(".md"):
            sub_path = os.path.join(save_path, entry)
            if not os.path.isdir(sub_path):
                continue
            sub_entries = [e for e in os.listdir(sub_path) if e.endswith(".json") and not e.startswith(".")]
            for sub_entry in sub_entries:
                file_path = os.path.join(sub_path, sub_entry)
                with open(file_path) as fp:
                    data = json.load(fp)
                _fill_queue_row(data)
                all_evals.append(data)

    pending_list = [e for e in all_evals if e["status"] in ["PENDING", "RERUN"]]
    running_list = [e for e in all_evals if e["status"] == "RUNNING"]
    finished_list = [e for e in all_evals if e["status"].startswith("FINISHED") or e["status"] == "PENDING_NEW_EVAL"]
    df_pending = pd.DataFrame.from_records(pending_list, columns=cols)
    df_running = pd.DataFrame.from_records(running_list, columns=cols)
    df_finished = pd.DataFrame.from_records(finished_list, columns=cols)
    return df_finished[cols], df_running[cols], df_pending[cols]


def _fill_queue_row(data: dict) -> None:
    hf_repo = data.get("hf_repo")
    if hf_repo and hf_repo != "Unknown":
        data[EvalQueueColumn.model.name] = make_clickable_model(hf_repo, data["model"])
    else:
        data[EvalQueueColumn.model.name] = data["model"]
    data[EvalQueueColumn.revision.name] = data.get("revision", "main")
