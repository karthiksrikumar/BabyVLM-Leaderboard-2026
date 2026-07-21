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
from __future__ import annotations
import argparse
import datetime
import json
import os
import shutil
import sys

from huggingface_hub import HfApi

from runner import config
from runner.emailer import send_email


def _load(stage_dir, name):
    with open(os.path.join(stage_dir, name)) as f:
        return json.load(f)


def _dir_size_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1e6


def _cleanup_staged(stage_dir, submission_id, request_obj, results_obj):
    """Delete the local staged bundle after publishing, to save disk in the org's quota.

    A one-line audit record (no bulky logs) is appended to PENDING_DIR/published_log.jsonl
    so there's a lasting record of what went to the leaderboard. Set BABYVLM_KEEP_STAGED=1
    to keep the full staged directory instead.
    """
    if os.environ.get("BABYVLM_KEEP_STAGED") == "1":
        print(f"[approve] BABYVLM_KEEP_STAGED=1 — leaving staged files at {stage_dir}")
        return
    freed = _dir_size_mb(stage_dir)
    os.makedirs(config.PENDING_DIR, exist_ok=True)
    record = {
        "submission_id": submission_id,
        "model": request_obj.get("model"),
        "hf_repo": request_obj.get("hf_repo"),
        "submitted_time": request_obj.get("submitted_time"),
        "published_time": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results": results_obj.get("results", {}),
    }
    with open(os.path.join(config.PENDING_DIR, "published_log.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")
    shutil.rmtree(stage_dir, ignore_errors=True)
    print(f"[approve] cleaned up staged files ({freed:.1f} MB freed). Audit record kept in "
          f"{os.path.join(config.PENDING_DIR, 'published_log.jsonl')}.")


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

    # Regenerate + push the static leaderboard page so the new row shows immediately.
    try:
        from runner.publish_static import publish
        publish(token=token)
    except Exception as e:
        print(f"[approve] warning: could not refresh static leaderboard ({e}). "
              f"Run `python -m runner.publish_static` manually.")

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

    # Free local disk now that everything is safely on the Hub.
    _cleanup_staged(stage_dir, submission_id, request_obj, results_obj)


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
    # Trim the heavy lmms-eval logs but keep the small records so the submission can be re-reviewed.
    logs_dir = os.path.join(stage_dir, "logs")
    if os.path.isdir(logs_dir) and os.environ.get("BABYVLM_KEEP_STAGED") != "1":
        freed = _dir_size_mb(logs_dir)
        shutil.rmtree(logs_dir, ignore_errors=True)
        print(f"[approve] removed rejected submission's logs ({freed:.1f} MB freed); kept JSON records.")
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
