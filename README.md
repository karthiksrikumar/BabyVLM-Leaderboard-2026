---
title: BabyVLM Leaderboard 2026
emoji: 👶🧠
colorFrom: pink
colorTo: purple
sdk: gradio
app_file: app.py
pinned: true
license: apache-2.0
short_description: The BabyVLM 2026 (DevCV Toolbox) leaderboard
sdk_version: 5.19.0
---

# BabyVLM Leaderboard 2026

A HuggingFace-Space leaderboard for the **DevCV Toolbox** — the 11-benchmark evaluation suite
from *BabyVLM-V2: Toward Developmentally Grounded Pretraining and Benchmarking of Vision
Foundation Models*. Modelled on the [BabyLM 2026 leaderboard](https://huggingface.co/spaces/BabyLM-community/BabyLM-Leaderboard-2026).

## How submission works (organizer-run)

Because the DevCV test sets derive from the IRB-restricted
[SAYCam](https://nyu.databrary.org/volume/564) corpus, **participants never see the test data.**
A submission is, at its core, **two files: a `model.py` wrapper and a checkpoint**, which the
participant hands to the organizers; **the organizers run the held-out test set** on their own
infrastructure and, after review, publish the scores.

1. **Participant** trains/validates locally and produces:
   - a `model.py` — an `lmms-eval` model wrapper (subclass of `lmms_eval.api.model.lmms`,
     decorated `@register_model(...)`, implementing `generate_until` + `generate_until_multi_round`
     + `loglikelihood`). Reference: [`submission_template/model.py`](submission_template/model.py)
     (the `babyllava` wrapper).
   - a checkpoint.
2. **Participant** submits via the **form on the leaderboard Space** (the *Submit a model* section),
   which opens a **pre-filled GitHub issue**, or by opening a **PR** adding `submissions/<name>/`
   (`submission.json` + `model.py`). Linking a HuggingFace model page is **optional**. All submitting
   happens on GitHub; see [`submissions/README.md`](submissions/README.md).
3. **Organizer** runs the held-out test set on an **L40S GPU** (SCC batch job):
   ```bash
   qsub runner/submit_l40s.qsub --submission submissions/<name>
   # or directly:
   qsub runner/submit_l40s.qsub --model_name <name> --checkpoint <repo_or_path> --registered_model <wrapper>
   ```
   The runner emails the organizers (incl. `babyvlm-challenge@googlegroups.com`) when the eval
   **starts** and again when it **finishes** with the full score table, and stages the result.
4. **Organizer** approves or rejects:
   ```bash
   python -m runner.approve yes <submission_id>   # publish to the leaderboard
   python -m runner.approve no  <submission_id>   # reject; publish nothing
   ```
   `yes` pushes the scores to the results dataset and the row appears on the next refresh.

See [`runner/README.md`](runner/README.md) for the full organizer workflow (email config, the
held-out-data map, `yes`/`no` scripts).

## The 11 benchmarks

Picture Vocabulary · Looking While Listening · Localization · Left/Right · Spatial Details ·
Visual Delayed Response (Binary & Open) · Who Has More (Real & Synthetic) · Object Counting · Memory.

All metrics are accuracy (0–1 in the scores file, shown as 0–100). The **Overall Average** is the
mean of the 11 primary metrics; missing benchmarks count as 0.

## Configuration

- **`src/envs.py`** — HF org/user (`OWNER`) and the request/results dataset repo names.
- **`src/about.py`** — the 11 benchmark tasks (`Tasks` enum), their primary metric, and all UI text.
- **`src/display/utils.py`** — leaderboard columns.
- **`src/leaderboard/read_evals.py`** + **`src/populate.py`** — read result/request files → dataframe.
- **`src/submission/`** — `check_validity.py` (validate scores file), `eval_submission.py`
  (pass-through scoring), `submit.py` (form handler, writes request + results to the HF datasets).

## Setup on a fresh org

1. Set `OWNER` in `src/envs.py` (or the `BABYVLM_OWNER` env var) to your HF org/user.
2. Provide an `HF_TOKEN` secret with write access in the Space settings.
3. The request/results datasets (`<OWNER>/babyvlm-leaderboard-2026-requests` and
   `...-results`) are created automatically on the first submission.

## Scores file format

```json
{
  "results": {
    "baby_pv": {"acc": 0.42},
    "baby_winoground": {"group_score": 0.11, "image_score": 0.3, "text_score": 0.2},
    "baby_localize": {"acc": 0.51},
    "baby_leftright": {"acc": 0.40},
    "baby_spatialdetails": {"acc": 0.33},
    "baby_vdr_binary": {"acc_exact": 0.62, "acc_adjacent": 0.71},
    "baby_vdr_open": {"acc_exact": 0.30, "acc_adjacent": 0.55},
    "baby_compare_real": {"acc": 0.54},
    "baby_compare_synthetic": {"acc": 0.60},
    "baby_count": {"acc": 0.16},
    "baby_memory": {"acc_learning": 0.90, "acc_testing_raw": 0.50, "acc_testing_adjusted": 0.41}
  }
}
```
Scores are fractions in `[0, 1]`. Incomplete submissions are allowed.
