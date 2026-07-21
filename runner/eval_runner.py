#!/usr/bin/env python3
"""BabyVLM organizer-side evaluation runner.

A participant uploads their ``model.py`` (an lmms-eval wrapper) + checkpoint. The
organizer runs THIS on their own infrastructure — where the IRB-restricted test
data lives — so participants never see the test set. The runner:

  1. emails the organizers that an evaluation has STARTED;
  2. symlinks the held-out test sets into the DevCV-Toolbox ``dataset/`` dir;
  3. (optionally) installs + registers the submitted model wrapper;
  4. runs all 11 benchmarks with lmms-eval;
  5. collates the scores;
  6. stages the result under ``pending/<submission_id>/`` together with ready-to-run
     ``yes`` and ``no`` approval scripts;
  7. emails the organizers the RESULTS plus the exact approve commands.

Nothing is pushed to the public leaderboard here — that only happens when an
organizer runs ``yes`` (see runner/approve.py).

Example
-------
    python -m runner.eval_runner \
        --model_name babyllava-vit-tinyllama \
        --registered_model babyllava \
        --checkpoint /path/to/checkpoint-3536 \
        --conv_template baby_v1 \
        --devcv_root /path/to/DevCV-Toolbox \
        --metadata submission_meta.json
"""
import argparse
import datetime
import glob
import json
import os
import shutil
import subprocess
import sys

from runner import config
from runner.emailer import send_email


# ----------------------------------------------------------------------------- helpers
def _now_id(model_name: str) -> str:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in model_name)
    return f"{safe}__{stamp}"


def _iso_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def materialize_datasets(devcv_root: str) -> list[str]:
    """Symlink the held-out test sets into <devcv_root>/dataset/. Returns tasks whose data is missing."""
    dataset_dir = os.path.join(devcv_root, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    missing = []
    for fname, src in config.DATA_MAP.items():
        dst = os.path.join(dataset_dir, fname)
        if not os.path.exists(src):
            missing.append(f"{fname} (source not found: {src})")
            continue
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)
    return missing


def install_wrapper(devcv_root: str, wrapper_path: str, registered_model: str) -> None:
    """Copy a submitted model.py into lmms_eval/models/ and register it in __init__.py."""
    models_dir = os.path.join(devcv_root, "lmms_eval", "models")
    dst = os.path.join(models_dir, f"{registered_model}.py")
    shutil.copyfile(wrapper_path, dst)
    init_path = os.path.join(models_dir, "__init__.py")
    with open(init_path) as f:
        init_src = f.read()
    # Best-effort registration: add "<name>": "<ClassName>" to AVAILABLE_MODELS if absent.
    if f'"{registered_model}"' not in init_src:
        print(f"[warn] '{registered_model}' not found in {init_path}. "
              f"Add it to AVAILABLE_MODELS manually if lmms-eval can't resolve it.")


def collate_logs(logs_dir: str) -> dict:
    """Extract the 11-benchmark scores from lmms-eval's *_results.json output."""
    from src.about import VALID_TASK_METRICS  # {benchmark: {metric, ...}}

    files = sorted(glob.glob(os.path.join(logs_dir, "**", "*_results.json"), recursive=True))
    collated: dict[str, dict[str, float]] = {}
    for path in files:
        with open(path) as f:
            data = json.load(f)
        for task, metrics in data.get("results", {}).items():
            if task not in VALID_TASK_METRICS or not isinstance(metrics, dict):
                continue
            for key, v in metrics.items():
                name = key.split(",")[0]
                if name in VALID_TASK_METRICS[task] and isinstance(v, (int, float)) and not isinstance(v, bool):
                    collated.setdefault(task, {})[name] = float(v)
    return collated


def _results_table(scores: dict) -> str:
    from src.about import Tasks

    lines = [f"{'Benchmark':<26} {'Metric':<22} {'Score':>7}"]
    lines.append("-" * 58)
    primary = []
    for t in Tasks:
        b, m = t.value.benchmark, t.value.metric
        v = scores.get(b, {}).get(m)
        cell = f"{v * 100:6.2f}" if isinstance(v, (int, float)) else "   -- "
        lines.append(f"{t.value.col_name:<26} {m:<22} {cell:>7}")
        if t.value.in_overall:
            primary.append(v * 100 if isinstance(v, (int, float)) else 0.0)
    overall = sum(primary) / len(primary) if primary else 0.0
    lines.append("-" * 58)
    lines.append(f"{'OVERALL AVERAGE':<26} {'(11 primary metrics)':<22} {overall:6.2f}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- main
def run(args) -> str:
    devcv_root = args.devcv_root or config.DEVCV_ROOT
    if not devcv_root or not os.path.isdir(devcv_root):
        sys.exit(f"DevCV-Toolbox checkout not found. Pass --devcv_root or set BABYVLM_DEVCV_ROOT (got: {devcv_root!r})")

    metadata = {}
    if args.metadata and os.path.isfile(args.metadata):
        with open(args.metadata) as f:
            metadata = json.load(f)

    submission_id = _now_id(args.model_name)
    stage_dir = os.path.join(config.PENDING_DIR, submission_id)
    os.makedirs(stage_dir, exist_ok=True)
    logs_dir = os.path.join(stage_dir, "logs")
    tasks = args.tasks.split(",") if args.tasks else config.TASKS

    # --- (1) email: STARTED ---
    send_email(
        subject=f"[BabyVLM] Evaluation STARTED: {args.model_name}",
        body_text=(
            f"An evaluation has started on the BabyVLM held-out test set.\n\n"
            f"  Submission id : {submission_id}\n"
            f"  Model name    : {args.model_name}\n"
            f"  Checkpoint    : {args.checkpoint}\n"
            f"  Model repo    : {metadata.get('hf_repo', 'n/a')}\n"
            f"  Tasks         : {', '.join(tasks)}\n"
            f"  Host          : {os.uname().nodename}\n"
            f"  Started (UTC) : {_iso_now()}\n\n"
            f"You'll get a second email with the results and approval commands when it finishes."
        ),
    )

    # --- (2) held-out data ---
    missing = materialize_datasets(devcv_root)
    if missing:
        print(f"[warn] missing test data for: {missing}")

    # --- (3) install wrapper (optional) ---
    registered_model = args.registered_model
    if args.wrapper:
        registered_model = registered_model or os.path.splitext(os.path.basename(args.wrapper))[0]
        install_wrapper(devcv_root, args.wrapper, registered_model)
    if not registered_model:
        sys.exit("Provide --registered_model (name already in lmms_eval/models) or --wrapper (path to model.py).")

    # --- (4) run lmms-eval ---
    model_args = f"pretrained={args.checkpoint},conv_template={args.conv_template}"
    cmd = [
        "accelerate", "launch", "--num_processes=1", "-m", "lmms_eval",
        "--model", registered_model,
        "--model_args", model_args,
        "--task", ",".join(tasks),
        "--batch_size", str(args.batch_size),
        "--output_path", logs_dir,
        "--trust_remote_code",
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    # If precomputed scores are supplied, skip lmms-eval entirely (e.g. the eval was
    # run on another machine). Otherwise run the harness.
    if getattr(args, "scores_json", None):
        with open(args.scores_json) as f:
            _sj = json.load(f)
        scores = _sj.get("results", _sj)
        print(f"[runner] using precomputed scores from {args.scores_json} (skipping lmms-eval)")
    else:
        print("[runner] $", " ".join(cmd))
        if not args.dry_run_eval:
            env = dict(os.environ)
            proc = subprocess.run(cmd, cwd=devcv_root, env=env)
            if proc.returncode != 0:
                send_email(
                    subject=f"[BabyVLM] Evaluation FAILED: {args.model_name}",
                    body_text=f"lmms-eval exited with code {proc.returncode} for submission {submission_id}.\n"
                              f"Check logs at {logs_dir} on {os.uname().nodename}.",
                )
                sys.exit(f"lmms-eval failed (exit {proc.returncode})")
        # --- (5) collate ---
        scores = collate_logs(logs_dir)

    # --- (6) stage result + request + yes/no scripts ---
    results_obj = {
        "track": "babyvlm",
        "config": {
            "model_name": args.model_name,
            "model_sha": metadata.get("revision", "main"),
            **{k: v for k, v in metadata.items() if k != "revision"},
        },
        "results": scores,
    }
    submitted_time = _iso_now()
    user_name = (metadata.get("hf_repo", "Unknown").split("/")[0]) if metadata.get("hf_repo") else "Unknown"
    request_obj = {
        "model": args.model_name,
        "hf_repo": metadata.get("hf_repo", "Unknown"),
        "track": "babyvlm",
        "user": user_name,
        "revision": metadata.get("revision", "main"),
        "status": "FINISHED",
        "submitted_time": submitted_time,
        **{k: v for k, v in metadata.items() if k not in ("hf_repo", "revision")},
    }
    with open(os.path.join(stage_dir, "results.json"), "w") as f:
        json.dump(results_obj, f, indent=2)
    with open(os.path.join(stage_dir, "request.json"), "w") as f:
        json.dump(request_obj, f, indent=2)
    with open(os.path.join(stage_dir, "meta.json"), "w") as f:
        json.dump({"submission_id": submission_id, "user": user_name, "submitted_time": submitted_time, "status": "PENDING_APPROVAL"}, f, indent=2)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for decision in ("yes", "no"):
        script = os.path.join(stage_dir, decision)
        with open(script, "w") as f:
            f.write(
                "#!/usr/bin/env bash\n"
                f"# Run this to {'PUSH' if decision == 'yes' else 'REJECT'} submission {submission_id}\n"
                f'cd "{repo_root}"\n'
                f'exec python -m runner.approve {decision} "{submission_id}" "$@"\n'
            )
        os.chmod(script, 0o755)

    # --- (7) email: RESULTS + approval commands ---
    table = _results_table(scores)
    yes_cmd = f"cd {repo_root} && python -m runner.approve yes {submission_id}"
    no_cmd = f"cd {repo_root} && python -m runner.approve no {submission_id}"
    send_email(
        subject=f"[BabyVLM] Evaluation RESULTS — approve? {args.model_name}",
        body_text=(
            f"Evaluation finished for '{args.model_name}' (submission {submission_id}).\n\n"
            f"{table}\n\n"
            f"Review the scores above. To PUBLISH to the BabyVLM leaderboard, run:\n\n"
            f"    {yes_cmd}\n"
            f"      (or:  {stage_dir}/yes )\n\n"
            f"To REJECT (do not publish), run:\n\n"
            f"    {no_cmd}\n"
            f"      (or:  {stage_dir}/no )\n\n"
            f"Staged files: {stage_dir}\n"
        ),
    )

    print(f"\n[runner] Done. Staged at {stage_dir}")
    print(f"[runner] Approve:  python -m runner.approve yes {submission_id}")
    print(f"[runner] Reject :  python -m runner.approve no  {submission_id}")
    return submission_id


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", required=True, help="Unique leaderboard name for this submission")
    ap.add_argument("--checkpoint", required=True, help="Path (or HF repo) to the participant's checkpoint")
    ap.add_argument("--registered_model", default=None, help="Wrapper name already registered in lmms_eval/models")
    ap.add_argument("--wrapper", default=None, help="Path to a submitted model.py to install + register")
    ap.add_argument("--conv_template", default="baby_v1")
    ap.add_argument("--devcv_root", default=None, help="Path to a DevCV-Toolbox checkout (or set BABYVLM_DEVCV_ROOT)")
    ap.add_argument("--metadata", default=None, help="JSON file of submission metadata (hf_repo, model_type, ...)")
    ap.add_argument("--tasks", default=None, help="Comma-separated task subset (default: all 11)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="Limit examples per task (smoke test)")
    ap.add_argument("--scores_json", default=None, help="Use precomputed scores (from another machine); skip lmms-eval")
    ap.add_argument("--dry_run_eval", action="store_true", help="Skip the lmms-eval call (test staging/email/approve only)")
    args = ap.parse_args()

    # make repo root importable (for `import src.about`)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run(args)


if __name__ == "__main__":
    main()
