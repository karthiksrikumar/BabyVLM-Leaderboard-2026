# BabyVLM organizer evaluation runner

This directory holds the **organizer-side** tooling. Participants never run the test
set — they hand you a `model.py` wrapper + checkpoint, and you run it here, on the
machine that has the held-out DevCV/SAYCam test data.

```
runner/
├── config.py       # recipients, HF repos, held-out-data map, paths
├── emailer.py      # notification emails (sendmail by default; SMTP if configured)
├── eval_runner.py  # run the 11 benchmarks, collate, stage, email
└── approve.py      # yes = publish to leaderboard, no = reject
```

## One-time setup

```bash
# 1. Check out the toolbox the runner drives
git clone https://github.com/ShawnKing98/DevCV-Toolbox
cd DevCV-Toolbox && pip install -e .          # + the model's deps (e.g. the LLaVA package)
export BABYVLM_DEVCV_ROOT=$PWD

# 2. Point the runner at this leaderboard repo and an HF write token
export HF_TOKEN=hf_xxx                          # write access to the karthiksrikumar org
# (optional) external SMTP instead of the local MTA:
#   export BABYVLM_SMTP_HOST=smtp.gmail.com BABYVLM_SMTP_USER=you@gmail.com BABYVLM_SMTP_PASS=app_password
```

The held-out test sets are wired up in `config.py` (`DATA_MAP`). On BU's SCC they resolve
automatically; elsewhere set `BABYVLM_DATA_ROOT` / `BABYVLM_PV_ROOT`.

## Evaluating a submission

A submission (from the Space's Submit tab) is a PENDING request in the queue dataset giving
a model name, a repo/checkpoint pointer, the wrapper name, and metadata. To evaluate it:

```bash
# from the leaderboard repo root
python -m runner.eval_runner \
    --model_name babyllava-vit-tinyllama \
    --checkpoint karthiksrikumar/babyllava   # or a local checkpoint dir \
    --registered_model babyllava             # wrapper name in lmms_eval/models \
    --conv_template baby_v1 \
    --metadata submission_meta.json          # optional: hf_repo, model_type, ... \
    --devcv_root $BABYVLM_DEVCV_ROOT
```

If the participant sent a standalone `model.py`, install it automatically with
`--wrapper /path/to/model.py` (it is copied into `lmms_eval/models/`).

Use `--limit 4` for a quick smoke test, or `--dry_run_eval` to exercise the
staging/email/approval plumbing without running the model.

The runner then:
1. emails the organizers **"Evaluation STARTED"**;
2. runs all 11 benchmarks and collates the scores;
3. stages everything under `pending/<submission_id>/`;
4. emails the organizers **"Evaluation RESULTS — approve?"** with the score table and the
   exact `yes` / `no` commands.

## Approving (publishing) or rejecting

The results email contains ready-to-run commands. From the repo root:

```bash
python -m runner.approve yes <submission_id>   # push scores to the leaderboard
python -m runner.approve no  <submission_id>   # reject; publish nothing
```

Each staged submission also has convenience scripts you can run directly:

```bash
pending/<submission_id>/yes     # publish
pending/<submission_id>/no      # reject
```

`yes` uploads the results + request files to the leaderboard's HF dataset repos
(`karthiksrikumar/babyvlm-leaderboard-2026-results` / `-requests`); the Space shows the row
on its next refresh. `no` marks the submission rejected and pushes nothing. Both send a
final confirmation email.

## Emails

By default emails go out through the local `sendmail` MTA (works on SCC `@bu.edu` hosts).
To use an external SMTP server (e.g. a Gmail app password), set `BABYVLM_SMTP_HOST` and
friends (see `emailer.py`). Set `BABYVLM_EMAIL_DRYRUN=1` to print emails instead of sending.
Recipients live in `config.RECIPIENTS`.
