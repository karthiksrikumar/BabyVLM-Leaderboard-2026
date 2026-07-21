from __future__ import annotations
import json
import os
from collections import defaultdict

from src.about import BENCHMARK_TASK_KEYS, VALID_TASK_METRICS


def already_submitted_models(requested_models_dir: str) -> tuple[set, dict]:
    """Gather the list of already-submitted models to avoid duplicates."""
    depth = 1
    file_names = []
    users_to_submission_dates = defaultdict(list)

    if not os.path.isdir(requested_models_dir):
        return set(file_names), users_to_submission_dates

    for root, _, files in os.walk(requested_models_dir):
        current_depth = root.count(os.sep) - requested_models_dir.count(os.sep)
        if current_depth == depth:
            for file in files:
                if not file.endswith(".json"):
                    continue
                with open(os.path.join(root, file), "r") as f:
                    info = json.load(f)
                    try:
                        file_names.append(f"{info['model']}_{info['revision']}_{info['track']}")
                    except Exception:
                        continue
                    if "submitted_time" in info:
                        users_to_submission_dates[info.get("user", "Unknown")].append(info["submitted_time"])

    return set(file_names), users_to_submission_dates


def is_valid_scores_file(path: str) -> tuple[bool, str]:
    """Validate a BabyVLM scores file produced by collate_results.py.

    Expected format:
        {"results": {benchmark: {metric: score_in_[0,1], ...}, ...}}
      or, for convenience, the bare mapping:
        {benchmark: {metric: score_in_[0,1], ...}, ...}

    Rules:
      - At least one recognised (benchmark, metric) pair with a numeric score.
      - Unknown benchmarks/metrics are ignored (not an error) but reported.
      - Every score present must be a number in [0, 1].
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return False, f"Error: scores file could not be read as JSON ({e})."

    if not isinstance(data, dict):
        return False, "Error: scores file must be a JSON object."

    results = data.get("results", data)
    if not isinstance(results, dict):
        return False, "Error: 'results' must be a JSON object mapping benchmark -> {metric: score}."

    n_recognised = 0
    unknown = []
    for bench, metrics in results.items():
        if bench not in BENCHMARK_TASK_KEYS:
            unknown.append(bench)
            continue
        if not isinstance(metrics, dict):
            return False, f"Error: '{bench}' must map to a dict of {{metric: score}}, got {type(metrics).__name__}."
        for metric, value in metrics.items():
            if metric not in VALID_TASK_METRICS.get(bench, set()):
                unknown.append(f"{bench}.{metric}")
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return False, f"Error: score for '{bench}.{metric}' must be a number, got {type(value).__name__}."
            if not (0.0 <= float(value) <= 1.0):
                return False, f"Error: score for '{bench}.{metric}' = {value} is outside [0, 1] (report accuracy as a fraction)."
            n_recognised += 1

    if n_recognised == 0:
        return False, (
            "Error: no recognised benchmark scores found. Expected keys like "
            f"{', '.join(BENCHMARK_TASK_KEYS[:4])}, ... See the submission instructions."
        )

    msg = "Upload successful."
    if unknown:
        msg += f" (Ignored unrecognised keys: {', '.join(sorted(set(unknown))[:10])})"
    return True, msg
