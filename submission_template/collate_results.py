#!/usr/bin/env python3
"""Collate DevCV-Toolbox (lmms-eval) output into a BabyVLM leaderboard scores file.

lmms-eval writes one `<date>_results.json` per run under `--output_path`, containing:

    {"results": {"baby_pv": {"acc,none": 0.42, "acc_stderr,none": ...,
                             "alias": "baby_pv"}, ...}, ...}

This script scans a logs directory for every such file, extracts the recognised
(benchmark, metric) scores for the 11 BabyVLM benchmarks, and writes a single
scores file ready to upload to the BabyVLM Leaderboard:

    {"results": {"baby_pv": {"acc": 0.42}, "baby_vdr_open": {"acc_exact": ...}, ...}}

Usage:
    python collate_results.py --logs_dir ./logs --out my_scores.json
"""
import argparse
import glob
import json
import os

# The 11 BabyVLM benchmarks and the metric keys the leaderboard understands.
# (Mirror of src/about.py Tasks — the leaderboard ignores anything not listed here.)
VALID_TASK_METRICS = {
    "baby_pv": ["acc"],
    "baby_winoground": ["group_score", "image_score", "text_score"],
    "baby_localize": ["acc"],
    "baby_leftright": ["acc"],
    "baby_spatialdetails": ["acc"],
    "baby_vdr_binary": ["acc_exact", "acc_adjacent"],
    "baby_vdr_open": ["acc_exact", "acc_adjacent"],
    "baby_compare_real": ["acc"],
    "baby_compare_synthetic": ["acc"],
    "baby_count": ["acc"],
    "baby_memory": ["acc_learning", "acc_testing_raw", "acc_testing_adjusted"],
}


def collate(logs_dir: str) -> dict:
    result_files = sorted(glob.glob(os.path.join(logs_dir, "**", "*_results.json"), recursive=True))
    if not result_files:
        # also accept a single results file passed directly as the dir
        if os.path.isfile(logs_dir) and logs_dir.endswith(".json"):
            result_files = [logs_dir]
    if not result_files:
        raise SystemExit(f"No *_results.json files found under {logs_dir!r}")

    collated: dict[str, dict[str, float]] = {}
    for path in result_files:  # oldest -> newest; later files overwrite earlier scores
        with open(path) as f:
            data = json.load(f)
        results = data.get("results", {})
        for task, metrics in results.items():
            if task not in VALID_TASK_METRICS or not isinstance(metrics, dict):
                continue
            for wanted in VALID_TASK_METRICS[task]:
                # lmms-eval keys metrics as "<metric>,<filter>" (filter is usually "none")
                val = None
                for key, v in metrics.items():
                    name = key.split(",")[0]
                    if name == wanted and isinstance(v, (int, float)):
                        val = float(v)
                        break
                if val is not None:
                    collated.setdefault(task, {})[wanted] = val
    return collated


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs_dir", required=True, help="lmms-eval --output_path directory (or a single *_results.json)")
    ap.add_argument("--out", default="my_scores.json", help="output scores file")
    args = ap.parse_args()

    collated = collate(args.logs_dir)
    if not collated:
        raise SystemExit("No recognised BabyVLM benchmark scores found in the logs.")

    with open(args.out, "w") as f:
        json.dump({"results": collated}, f, indent=2)

    found = sorted(collated.keys())
    missing = [t for t in VALID_TASK_METRICS if t not in collated]
    print(f"Wrote {args.out} with {len(found)}/11 benchmarks: {', '.join(found)}")
    if missing:
        print(f"Missing (will score as 0 on the leaderboard): {', '.join(missing)}")


if __name__ == "__main__":
    main()
