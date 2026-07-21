import json

from src.about import BENCHMARK_TASK_KEYS, VALID_TASK_METRICS


def evaluate_submission(path_to_submission: str) -> dict[str, dict[str, float]]:
    """Turn an uploaded BabyVLM scores file into the canonical results dict.

    BabyVLM does NOT re-score server-side: the DevCV test data is IRB-restricted
    (SAYCam), so gold answers cannot be hosted here. Participants run the DevCV
    Toolbox locally against the held-out sets and upload the resulting scores.
    This function validates and passes those pre-computed scores through, keeping
    only recognised (benchmark, metric) pairs with numeric values in [0, 1].

    Input format (either wrapped or bare):
        {"results": {benchmark: {metric: score, ...}, ...}}
        {benchmark: {metric: score, ...}, ...}

    Output:
        {benchmark: {metric: score_in_[0,1], ...}, ...}
    """
    with open(path_to_submission, "r") as f:
        data = json.load(f)

    results = data.get("results", data) if isinstance(data, dict) else {}

    processed: dict[str, dict[str, float]] = {}
    for bench, metrics in results.items():
        if bench not in BENCHMARK_TASK_KEYS or not isinstance(metrics, dict):
            continue
        kept = {}
        for metric, value in metrics.items():
            if metric not in VALID_TASK_METRICS.get(bench, set()):
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= float(value) <= 1.0:
                kept[metric] = float(value)
        if kept:
            processed[bench] = kept
    return processed
