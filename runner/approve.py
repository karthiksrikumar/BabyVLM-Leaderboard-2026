#!/usr/bin/env python3
"""Approve (``yes``) or reject (``no``) a staged BabyVLM submission.

``yes`` uploads the staged result + request files to the leaderboard's HuggingFace
dataset repos — the Space picks them up and the row appears on the leaderboard.
``no`` marks the submission rejected and publishes nothing.

Usage:
    python -m runner.approve yes <submission_id>
    python -m runner.approve no  <submission_id>

Requires an ``HF_TOKEN`` (write access to the leaderboard org) for ``yes``.
The per-submission ``pending/<id>/yes`` and ``pending/<id>/no`` scripts call this.
"""
import argparse
import json
import os
import sys

from huggingface_hub import HfApi

from runner import config
from runner.emailer import send_email


def _load(stage_dir, name):
    with open(os.path.join(stage_dir, name)) as f:
        return json.load(f)


def _save(stage_dir, name, obj):
    with open(os.path.join(stage_dir, name), "w") as f:
        json.dump(obj, f, indent=2)


def approve_yes(submission_id: str) -> None:
    stage_dir = os.path.join(config.PENDING_DIR, submission_id)
    if not os.path.isdir(stage_dir):
        sys.exit(f"No staged submission '{submission_id}' under {config.PENDING_DIR}")

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not set. Export a write token for the leaderboard org before approving.")

    meta = _load(stage_dir, "meta.json")
    results_obj = _load(stage_dir, "results.json")
    request_obj = _load(stage_dir, "request.json")
    user = meta.get("user", "Unknown")
    submitted_time = meta.get("submitted_time")

    api = HfApi(token=token)

    # Ensure the dataset repos exist.
    for repo in (config.QUEUE_REPO, config.RESULTS_REPO):
        api.create_repo(repo_id=repo, repo_type="dataset", private=False, exist_ok=True)

    # Upload the results file -> results dataset (leaderboard reads results_*.json).
    results_path_in_repo = f"{user}/results_{submitted_time}.json"
    api.upload_file(
        path_or_fileobj=os.path.join(stage_dir, "results.json"),
        path_in_repo=results_path_in_repo,
        repo_id=config.RESULTS_REPO,
        repo_type="dataset",
        commit_message=f"Publish {request_obj.get('model')} to BabyVLM leaderboard",
    )

    # Upload the request file -> requests dataset (drives the queue / date).
    model_path = str(request_obj.get("model", "model")).split("/")[-1]
    request_path_in_repo = f"{user}/{model_path}_eval_request_{submitted_time}.json"
    api.upload_file(
        path_or_fileobj=os.path.join(stage_dir, "request.json"),
        path_in_repo=request_path_in_repo,
        repo_id=config.QUEUE_REPO,
        repo_type="dataset",
        commit_message=f"Add {request_obj.get('model')} request (approved)",
    )

    meta["status"] = "APPROVED"
    _save(stage_dir, "meta.json", meta)

    print(f"[approve] PUBLISHED '{request_obj.get('model')}' to {config.RESULTS_REPO}")
    send_email(
        subject=f"[BabyVLM] PUBLISHED to leaderboard: {request_obj.get('model')}",
        body_text=(
            f"Submission {submission_id} ({request_obj.get('model')}) was APPROVED and pushed "
            f"to the BabyVLM leaderboard.\n\n"
            f"Results repo : {config.RESULTS_REPO}\n"
            f"File         : {results_path_in_repo}\n"
            f"The leaderboard will show it on its next refresh."
        ),
    )


def approve_no(submission_id: str, reason: str = "") -> None:
    stage_dir = os.path.join(config.PENDING_DIR, submission_id)
    if not os.path.isdir(stage_dir):
        sys.exit(f"No staged submission '{submission_id}' under {config.PENDING_DIR}")
    meta = _load(stage_dir, "meta.json")
    meta["status"] = "REJECTED"
    if reason:
        meta["reject_reason"] = reason
    _save(stage_dir, "meta.json", meta)
    request_obj = _load(stage_dir, "request.json")
    print(f"[approve] REJECTED '{request_obj.get('model')}' (submission {submission_id}). Nothing published.")
    send_email(
        subject=f"[BabyVLM] REJECTED (not published): {request_obj.get('model')}",
        body_text=(
            f"Submission {submission_id} ({request_obj.get('model')}) was REJECTED. "
            f"Nothing was pushed to the leaderboard.\n" + (f"\nReason: {reason}\n" if reason else "")
        ),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("decision", choices=["yes", "no"], help="yes = publish, no = reject")
    ap.add_argument("submission_id", help="Staged submission id (from the runner / results email)")
    ap.add_argument("--reason", default="", help="Optional reason (for 'no')")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if args.decision == "yes":
        approve_yes(args.submission_id)
    else:
        approve_no(args.submission_id, args.reason)


if __name__ == "__main__":
    main()
