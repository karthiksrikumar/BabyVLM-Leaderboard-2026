"""End-to-end tests for the BabyVLM leaderboard backend.

Run from the repo root:  python -m pytest tests/ -q   (or: python tests/test_pipeline.py)
No HF token or GPU required — the whole submission pipeline is exercised on local files.
"""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.about import Tasks
from src.submission.check_validity import is_valid_scores_file
from src.submission.eval_submission import evaluate_submission
from src.display.utils import COLS, BENCHMARK_COLS
from src.populate import get_leaderboard_df


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def test_11_benchmarks_defined():
    primary = [t for t in Tasks if t.value.in_overall]
    assert len(primary) == 11, f"expected 11 primary benchmarks, got {len(primary)}"


def test_full_submission_overall_average():
    with tempfile.TemporaryDirectory() as tmp:
        scores = {"results": {
            "baby_pv": {"acc": 0.4213}, "baby_winoground": {"group_score": 0.1124},
            "baby_localize": {"acc": 0.5087}, "baby_leftright": {"acc": 0.9991},
            "baby_spatialdetails": {"acc": 0.362}, "baby_vdr_binary": {"acc_exact": 0.6215},
            "baby_vdr_open": {"acc_exact": 0.2989}, "baby_compare_real": {"acc": 0.9969},
            "baby_compare_synthetic": {"acc": 0.6031}, "baby_count": {"acc": 0.1599},
            "baby_memory": {"acc_testing_adjusted": 0.407},
        }}
        sp = os.path.join(tmp, "scores.json")
        _write(sp, scores)
        ok, _ = is_valid_scores_file(sp)
        assert ok
        processed = evaluate_submission(sp)
        _write(os.path.join(tmp, "eval-results", "u", "results_2026-01-01T00:00:00Z.json"),
               {"track": "babyvlm", "config": {"model_name": "m", "hf_repo": "u/m", "model_sha": "main"}, "results": processed})
        df = get_leaderboard_df(os.path.join(tmp, "eval-results"), os.path.join(tmp, "eval-queue"), COLS, BENCHMARK_COLS)
        prims = [0.4213, 0.1124, 0.5087, 0.9991, 0.362, 0.6215, 0.2989, 0.9969, 0.6031, 0.1599, 0.407]
        assert abs(float(df["Overall Average"].iloc[0]) - round(sum(prims) / 11 * 100, 2)) < 0.01


def test_partial_submission_scores_missing_as_zero():
    with tempfile.TemporaryDirectory() as tmp:
        _write(os.path.join(tmp, "eval-results", "u", "results_2026-01-01T00:00:00Z.json"),
               {"track": "babyvlm", "config": {"model_name": "p", "hf_repo": "u/p", "model_sha": "main"},
                "results": {"baby_pv": {"acc": 1.0}, "baby_count": {"acc": 1.0}}})
        df = get_leaderboard_df(os.path.join(tmp, "eval-results"), os.path.join(tmp, "eval-queue"), COLS, BENCHMARK_COLS)
        assert abs(float(df["Overall Average"].iloc[0]) - round(200 / 11, 2)) < 0.01
        assert np.isnan(df["Localization"].iloc[0])


def test_resubmit_merges_latest_nonempty_per_benchmark():
    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, "eval-results", "u")
        _write(os.path.join(base, "results_2026-01-01T00:00:00Z.json"),
               {"track": "babyvlm", "config": {"model_name": "m", "hf_repo": "u/m", "model_sha": "main"},
                "results": {"baby_pv": {"acc": 0.5}, "baby_count": {"acc": 0.1}}})
        _write(os.path.join(base, "results_2026-02-02T00:00:00Z.json"),
               {"track": "babyvlm", "config": {"model_name": "m", "hf_repo": "u/m", "model_sha": "main"},
                "results": {"baby_count": {"acc": 0.9}, "baby_localize": {"acc": 0.7}}})
        df = get_leaderboard_df(os.path.join(tmp, "eval-results"), os.path.join(tmp, "eval-queue"), COLS, BENCHMARK_COLS)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["Picture Vocabulary"] == 50.0  # kept from first
        assert row["Object Counting"] == 90.0  # updated by second
        assert row["Localization"] == 70.0  # added by second


def test_invalid_files_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        for name, payload in [
            ("oor", {"results": {"baby_pv": {"acc": 1.5}}}),
            ("empty", {"results": {}}),
            ("unknown", {"results": {"foo": {"bar": 0.5}}}),
        ]:
            p = os.path.join(tmp, name + ".json")
            _write(p, payload)
            ok, _ = is_valid_scores_file(p)
            assert not ok, f"{name} should be invalid"


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print(f"PASS {fn.__name__}")
    print("All tests passed.")
