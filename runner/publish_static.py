#!/usr/bin/env python3
"""Regenerate the static leaderboard's ``leaderboard.json`` and push it to the Space.

The BabyVLM leaderboard is hosted as a **free static** HF Space (no PRO needed). The
page (``index.html``) renders a JSON file. This script rebuilds that JSON from every
published result in the results dataset and uploads it (plus ``index.html`` /
``README.md`` if present) to the Space repo.

Called automatically by ``runner/approve.py yes``; can also be run standalone:

    python -m runner.publish_static
"""
import datetime
import json
import os
import sys
import tempfile

from huggingface_hub import HfApi, snapshot_download

from runner import config

# Plain (non-HTML) column order the static page expects.
PRIMARY_COLS = [
    "Overall Average", "Picture Vocabulary", "Looking While Listening", "Localization",
    "Left/Right", "Spatial Details", "VDR (Binary)", "VDR (Open)", "Who Has More (Real)",
    "Who Has More (Synth)", "Object Counting", "Memory",
]

SPACE_REPO = f"{config.OWNER}/BabyVLM-Leaderboard-2026"


def build_leaderboard_json(results_dir: str, requests_dir: str) -> dict:
    """Build the leaderboard payload from local result/request files."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.leaderboard.read_evals import get_raw_eval_results, _result_key
    from src.about import Tasks

    raw = get_raw_eval_results(results_dir, requests_dir)
    rows = []
    for ev in raw:
        row = {"Model": ev.full_model, "repo": (ev.repo_id if ev.repo_id and ev.repo_id != "Unknown" else "")}
        # per-benchmark cells
        primary_vals = []
        for t in Tasks:
            v = ev.results.get(_result_key(t.value.benchmark, t.value.metric))
            val = None if (v is None or (isinstance(v, float) and v != v)) else round(float(v), 2)
            row[t.value.col_name] = val
            if t.value.in_overall:
                primary_vals.append(val if val is not None else 0.0)
        row["Overall Average"] = round(sum(primary_vals) / len(primary_vals), 2) if primary_vals else 0.0
        # challenge flag comes from the request metadata if available
        rows.append(row)
    rows.sort(key=lambda r: r["Overall Average"], reverse=True)
    return {
        "updated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "benchmarks": PRIMARY_COLS,
        "rows": rows,
    }


def publish(token: str | None = None) -> str:
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not set; cannot publish the static leaderboard.")
    api = HfApi(token=token)

    with tempfile.TemporaryDirectory() as tmp:
        results_dir = os.path.join(tmp, "results")
        requests_dir = os.path.join(tmp, "requests")
        try:
            snapshot_download(repo_id=config.RESULTS_REPO, local_dir=results_dir, repo_type="dataset", token=token, tqdm_class=None)
        except Exception as e:
            print(f"[publish_static] could not fetch results ({e}); building empty board")
        try:
            snapshot_download(repo_id=config.QUEUE_REPO, local_dir=requests_dir, repo_type="dataset", token=token, tqdm_class=None)
        except Exception:
            pass

        payload = build_leaderboard_json(results_dir, requests_dir)
        lb_path = os.path.join(tmp, "leaderboard.json")
        with open(lb_path, "w") as f:
            json.dump(payload, f, indent=2)

        api.create_repo(repo_id=SPACE_REPO, repo_type="space", space_sdk="static", exist_ok=True)
        api.upload_file(path_or_fileobj=lb_path, path_in_repo="leaderboard.json", repo_id=SPACE_REPO, repo_type="space",
                        commit_message=f"Update leaderboard ({len(payload['rows'])} models)")

        # also (re)upload the page + README if bundled alongside this repo
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for fname in ("index.html", "README.md"):
            local = os.path.join(here, "static_space", fname)
            if os.path.exists(local):
                api.upload_file(path_or_fileobj=local, path_in_repo=fname, repo_id=SPACE_REPO, repo_type="space",
                                commit_message=f"Update {fname}")
    print(f"[publish_static] pushed leaderboard.json ({len(payload['rows'])} rows) to {SPACE_REPO}")
    return SPACE_REPO


if __name__ == "__main__":
    publish()
