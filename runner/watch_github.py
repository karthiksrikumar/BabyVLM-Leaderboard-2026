#!/usr/bin/env python3
"""Watch GitHub for new BabyVLM submissions and email the organizers.

A submission arrives as a GitHub issue (from the leaderboard form or the issue form,
labeled ``submission``). This watcher — run on the SCC where email + the L40S live —
polls the repo's issues, and for each NEW one it:

  1. parses the submission fields out of the issue body,
  2. materializes ``submissions/<model_name>/submission.json`` (+ ``model.py`` if the
     body contains a python code block),
  3. emails the organizers "SUBMISSION RECEIVED (via GitHub)" **with the exact command
     to run the evaluation on the L40S**.

The organizer just runs the command from the email to start the eval.

Run once:      python -m runner.watch_github
Poll forever:  python -m runner.watch_github --loop 600      # every 10 min
Seen-issue state is kept in <PENDING_DIR>/.github_seen.json so each issue emails once.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
import urllib.request

from runner import config
from runner.emailer import send_email

REPO = f"{config.OWNER}/BabyVLM-Leaderboard-2026"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEN_PATH = os.path.join(config.PENDING_DIR, ".github_seen.json")

# issue-form header keyword -> submission field. Order matters: more specific
# keywords are checked FIRST and "model name" is LAST, so headers like
# "Registered wrapper name (the @register_model name ...)" don't false-match it.
FIELD_MAP = [
    ("registered wrapper", "registered_model"),
    ("conversation template", "conv_template"),
    ("link your huggingface", "hf_model_url"),
    ("model.py", "model_py"),
    ("vision encoder", "vision_encoder"),
    ("language model", "language_model"),
    ("training data", "training_data"),
    ("number of parameters", "num_parameters"),
    ("description", "model_description"),
    ("model type", "model_type"),
    ("checkpoint", "checkpoint"),
    ("model name", "model_name"),
]


def _gh_get(url: str, token: str):
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
        "User-Agent": "babyvlm-watch",
    })
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def parse_issue_body(body: str) -> dict:
    """Parse a GitHub issue-form body ('### Header\\n\\nvalue') into submission fields."""
    fields: dict[str, str] = {}
    # split into (header, value) chunks on '### ' markers
    parts = re.split(r"\n#{2,3}\s+", "\n" + (body or ""))
    for chunk in parts:
        if "\n" not in chunk:
            continue
        header, _, value = chunk.partition("\n")
        header = header.strip().lower()
        value = value.strip()
        if value in ("_No response_", "_No response", "*No response*", ""):
            value = ""
        # strip a ```python ... ``` fence for the model.py field
        m = re.search(r"```(?:python)?\s*(.*?)```", value, re.S)
        code = m.group(1).strip() if m else None
        for kw, key in FIELD_MAP:
            if kw in header:
                fields[key] = code if (key == "model_py" and code is not None) else value
                break
    return fields


def materialize_submission(fields: dict) -> str | None:
    """Write submissions/<submitter>__<model_name>/{submission.json, model.py}. Returns the folder.

    Namespacing by submitter means two people who pick the same model name never overwrite
    each other's folder (and, downstream, never merge into one leaderboard row)."""
    name = fields.get("model_name")
    if not name:
        return None
    submitter = fields.get("submitter", "") or ""
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    safe_sub = re.sub(r"[^A-Za-z0-9._-]", "_", submitter)
    folder_name = f"{safe_sub}__{safe_name}" if safe_sub else safe_name
    folder = os.path.join(REPO_ROOT, "submissions", folder_name)
    os.makedirs(folder, exist_ok=True)
    metadata = {k: fields[k] for k in (
        "model_type", "vision_encoder", "language_model", "training_data",
        "num_parameters", "model_description") if fields.get(k)}
    if fields.get("hf_model_url"):
        metadata["hf_repo"] = fields["hf_model_url"]
    sub = {
        "model_name": name,
        "submitter": submitter,
        "checkpoint": fields.get("checkpoint", ""),
        "registered_model": fields.get("registered_model", ""),
        "conv_template": fields.get("conv_template") or "baby_v1",
        "hf_model_url": fields.get("hf_model_url", ""),
        "metadata": metadata,
    }
    with open(os.path.join(folder, "submission.json"), "w") as f:
        json.dump(sub, f, indent=2)
    py = fields.get("model_py", "")
    if py and "http" not in py[:12] and len(py) > 40:  # looks like code, not a bare link
        with open(os.path.join(folder, "model.py"), "w") as f:
            f.write(py + "\n")
    return folder


def _load_seen() -> set:
    try:
        with open(SEEN_PATH) as f:
            return set(json.load(f))
    except Exception:
        return set()


def _save_seen(seen: set):
    os.makedirs(config.PENDING_DIR, exist_ok=True)
    with open(SEEN_PATH, "w") as f:
        json.dump(sorted(seen), f)


def check_once(token: str) -> int:
    issues = _gh_get(f"https://api.github.com/repos/{REPO}/issues?state=open&labels=submission&per_page=50", token)
    # also accept issues whose title marks a submission, in case the label wasn't applied
    try:
        extra = _gh_get(f"https://api.github.com/repos/{REPO}/issues?state=open&per_page=50", token)
        seen_nums = {i["number"] for i in issues}
        issues += [i for i in extra if i.get("title", "").startswith("[Submission]") and i["number"] not in seen_nums]
    except Exception:
        pass

    seen = _load_seen()
    new_count = 0
    for issue in issues:
        num = issue["number"]
        if num in seen or "pull_request" in issue:
            continue
        fields = parse_issue_body(issue.get("body", ""))
        fields["submitter"] = issue.get("user", {}).get("login", "") or ""
        folder = materialize_submission(fields)
        name = fields.get("model_name") or issue.get("title", f"issue-{num}")
        rel = os.path.relpath(folder, REPO_ROOT) if folder else "submissions/<name>"
        run_cmd = f"cd {REPO_ROOT} && qsub runner/submit_l40s.qsub --submission {rel}"
        send_email(
            subject=f"[BabyVLM] SUBMISSION RECEIVED (via GitHub): {name}",
            body_text=(
                f"A new BabyVLM submission arrived via GitHub.\n\n"
                f"  Model       : {name}\n"
                f"  Checkpoint  : {fields.get('checkpoint', '(see issue)')}\n"
                f"  HF model    : {fields.get('hf_model_url') or '(not linked)'}\n"
                f"  GitHub issue: {issue.get('html_url')}\n"
                f"  Submitted by: {issue.get('user', {}).get('login', '?')}\n\n"
                f"It has been staged at: {rel}\n\n"
                f"👉 To RUN the evaluation on the L40S, run this command:\n\n"
                f"    {run_cmd}\n\n"
                f"You'll then get a STARTED email, a RESULTS email, and can approve with yes/no.\n"
                f"If you do NOT want to run it, just ignore this email."
            ),
        )
        print(f"[watch_github] emailed run command for issue #{num} ({name})")
        seen.add(num)
        new_count += 1
    _save_seen(seen)
    return new_count


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loop", type=int, default=0, help="Poll every N seconds (0 = run once)")
    args = ap.parse_args()
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        sec = "/projectnb/ivc-ml/srikumar/.secrets/.babyvlm_gh_pat"
        if os.path.exists(sec):
            token = open(sec).read().strip()
    if not token:
        sys.exit("No GitHub token (set GITHUB_TOKEN or place it in .secrets/.babyvlm_gh_pat).")

    if args.loop:
        print(f"[watch_github] polling {REPO} every {args.loop}s ...")
        while True:
            try:
                n = check_once(token)
                if n:
                    print(f"[watch_github] {n} new submission(s) emailed")
            except Exception as e:
                print(f"[watch_github] poll error: {e}")
            time.sleep(args.loop)
    else:
        n = check_once(token)
        print(f"[watch_github] done — {n} new submission(s) emailed")


if __name__ == "__main__":
    main()
