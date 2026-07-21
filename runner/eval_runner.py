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
from __future__ import annotations
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


def _find_wrapper_class(src: str) -> str | None:
    """Return the wrapper's class name: the class decorated @register_model, else the first class."""
    import ast
    try:
        tree = ast.parse(src)
    except Exception:
        return None
    first = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            first = first or node.name
            for dec in node.decorator_list:
                fn = dec.func if isinstance(dec, ast.Call) else dec
                if getattr(fn, "id", None) == "register_model" or getattr(fn, "attr", None) == "register_model":
                    return node.name
    return first


def install_wrapper(devcv_root: str, wrapper_path: str, registered_model: str) -> None:
    """Copy a submitted model.py into lmms_eval/models/<name>.py AND register it so
    `--model <name>` resolves.

    lmms-eval resolves a model via AVAILABLE_MODELS in lmms_eval/models/__init__.py
    (get_model() raises if the name is absent), so copying the file is not enough — we
    insert `"<name>": "<ClassName>"` into that dict. Idempotent; safe to call per submission
    on our own DevCV-Toolbox checkout.
    """
    models_dir = os.path.join(devcv_root, "lmms_eval", "models")
    dst = os.path.join(models_dir, f"{registered_model}.py")
    shutil.copyfile(wrapper_path, dst)

    init_path = os.path.join(models_dir, "__init__.py")
    with open(init_path) as f:
        init_src = f.read()
    if f'"{registered_model}"' in init_src:
        print(f"[runner] '{registered_model}' already registered in lmms_eval/models/__init__.py")
        return

    class_name = _find_wrapper_class(open(wrapper_path).read())
    if not class_name:
        print(f"[warn] no class found in {wrapper_path}; could not auto-register '{registered_model}'.")
        return

    marker = "AVAILABLE_MODELS = {"
    if marker not in init_src:
        print(f"[warn] AVAILABLE_MODELS not found in {init_path}; register '{registered_model}' manually.")
        return
    new_src = init_src.replace(marker, f'{marker}\n    "{registered_model}": "{class_name}",', 1)
    with open(init_path, "w") as f:
        f.write(new_src)
    print(f"[runner] registered '{registered_model}' -> {class_name} in lmms_eval/models/__init__.py")


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

    # Submitter is part of the row identity so two people can reuse a model name safely.
    submitter = (metadata.get("submitter") or getattr(args, "submitter", "") or "").strip()

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
    # If the participant supplied a model.py, copy it in under a name UNIQUE to this
    # (submitter, model) so two people's wrappers never overwrite each other in the toolbox.
    registered_model = args.registered_model
    if args.wrapper:
        internal = "sub_" + "".join(c if (c.isalnum() or c == "_") else "_"
                                    for c in f"{submitter}_{args.model_name}".strip("_")).lower()
        install_wrapper(devcv_root, args.wrapper, internal)
        registered_model = internal
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
            "submitter": submitter,  # part of the leaderboard row identity
            **{k: v for k, v in metadata.items() if k != "revision"},
        },
        "results": scores,
    }
    submitted_time = _iso_now()
    # Group result/request files by submitter so different people never share a folder.
    user_name = submitter or (metadata.get("hf_repo", "Unknown").split("/")[0] if metadata.get("hf_repo") else "Unknown")
    request_obj = {
        "model": args.model_name,
        "hf_repo": metadata.get("hf_repo", "Unknown"),
        "track": "babyvlm",
        "user": user_name,
        "revision": metadata.get("revision", "main"),
        "status": "FINISHED",
        "submitted_time": submitted_time,
        "submitter": submitter,
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


def _load_submission_dir(args):
    """Populate args from a submissions/<name>/ folder (submission.json + model.py).

    Writes the metadata block to a temp file under PENDING_DIR (ivc-ml, not home) and points
    args.metadata at it. Returns that temp path so the caller can delete it afterwards.
    """
    sub_dir = args.submission
    with open(os.path.join(sub_dir, "submission.json")) as f:
        sub = json.load(f)
    args.model_name = args.model_name or sub.get("model_name")
    args.checkpoint = args.checkpoint or sub.get("checkpoint")
    args.registered_model = args.registered_model or sub.get("registered_model")
    args.conv_template = sub.get("conv_template", args.conv_template)
    # a model.py in the submission folder (skip the EXAMPLE placeholder)
    cand = os.path.join(sub_dir, "model.py")
    if not args.wrapper and os.path.isfile(cand) and "EXAMPLE" not in os.path.basename(os.path.dirname(cand)):
        args.wrapper = cand
    meta = dict(sub.get("metadata", {}))
    if sub.get("submitter"):
        meta["submitter"] = sub["submitter"]
    if sub.get("hf_model_url"):
        meta.setdefault("hf_repo", sub["hf_model_url"])
    os.makedirs(config.PENDING_DIR, exist_ok=True)
    meta_path = os.path.join(config.PENDING_DIR, f".meta_{_now_id(args.model_name or 'sub')}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    args.metadata = meta_path
    return meta_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission", default=None, help="Path to a submissions/<name>/ folder (reads submission.json + model.py)")
    ap.add_argument("--model_name", default=None, help="Unique leaderboard name (or taken from --submission)")
    ap.add_argument("--checkpoint", default=None, help="Path/HF repo to the checkpoint (or taken from --submission)")
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

    meta_tmp = None
    if args.submission:
        meta_tmp = _load_submission_dir(args)
    if not args.model_name or not args.checkpoint:
        sys.exit("Provide --model_name and --checkpoint (directly or via --submission).")
    try:
        run(args)
    finally:
        if meta_tmp and os.path.exists(meta_tmp):
            os.remove(meta_tmp)  # temp metadata file lives under ivc-ml; remove it


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
