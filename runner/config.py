"""Configuration for the BabyVLM organizer-side evaluation runner.

The organizer holds the IRB-restricted test data and runs it on submitted models,
so participants never see it. This file centralises everything the runner needs:
who gets notified, which repos results are pushed to, and where the held-out
test sets live.
"""
from __future__ import annotations
import os

# --- People notified when an evaluation starts and finishes ---
RECIPIENTS = [
    "babyvlm-challenge@googlegroups.com",
    "karthiksrikumar83@gmail.com",
    "ac25@bu.edu",
    "maxwh@bu.edu",
    "bgong@bu.edu",
]

# From address for notification emails (override with BABYVLM_EMAIL_FROM).
EMAIL_FROM = os.environ.get("BABYVLM_EMAIL_FROM", "babyvlm-leaderboard@scc.bu.edu")

# --- GPU the evaluation requests on the SCC scheduler ---
# Submissions run on an L40S. The qsub batch script (runner/submit_l40s.qsub) requests
# this GPU type; kept here so it's configurable in one place.
GPU_TYPE = os.environ.get("BABYVLM_GPU_TYPE", "L40S")
CONDA_ENV = os.environ.get("BABYVLM_CONDA_ENV", "/projectnb/ivc-ml/wsashawn/miniconda3/envs/llava2")

# HuggingFace org/user that owns the leaderboard + its datasets.
OWNER = os.environ.get("BABYVLM_OWNER", "karthiksrikumar")
QUEUE_REPO = f"{OWNER}/babyvlm-leaderboard-2026-requests"
RESULTS_REPO = f"{OWNER}/babyvlm-leaderboard-2026-results"

# The 11 registered DevCV-Toolbox benchmark tasks.
TASKS = [
    "baby_pv",
    "baby_winoground",
    "baby_localize",
    "baby_leftright",
    "baby_spatialdetails",
    "baby_vdr_binary",
    "baby_vdr_open",
    "baby_compare_real",
    "baby_compare_synthetic",
    "baby_count",
    "baby_memory",
]

# Map each task's expected `dataset/<name>` file (from the task YAML `data_files`)
# to the real held-out test JSON on disk. The runner symlinks these into
# <DevCV-Toolbox>/dataset/ before evaluating so the harness finds the test sets.
# Override the data root with BABYVLM_DATA_ROOT if the org's copy lives elsewhere.
_SCC = os.environ.get("BABYVLM_DATA_ROOT", "/projectnb/ivc-ml/wqwang/Codelab/babydata/scc_json")
_PV = os.environ.get("BABYVLM_PV_ROOT", "/projectnb/ivc-ml/vwang")

DATA_MAP = {
    # dataset/<expected filename>  ->  real held-out test set
    "pv_test.json": f"{_PV}/picture_vocabulary_v2/dataset_seen.json",
    "localize_test.json": f"{_SCC}/localize_task_cropped/localize_4_crop_instructions_abc_test.json",
    "leftright_test.json": f"{_SCC}/leftright/interleaved_leftright_instructions_abc_test.json",
    "spatialdetails_test.json": f"{_SCC}/spatialdetails/abc_interleaved_spatialdetails_instructions_test.json",
    "vdr_binary_test.json": f"{_SCC}/visual_delay_response/binary_dataset_test.json",
    "vdr_open_test.json": f"{_SCC}/visual_delay_response/open_dataset_test.json",
    "compare_real_test.json": f"{_SCC}/compare_count_real/whohasmore_naturalistic.json",
    "compare_synthetic_test.json": f"{_SCC}/compare_count_synthetic/compare_synthetic_instructions_test_abc.json",
    "count_test.json": f"{_SCC}/object_counting/count_v2_test.json",
    "memory_test.json": f"{_SCC}/memory_task/memory_task_test_base5_p5_250.json",
    # baby_winoground reads dataset/baby_winoground/baby_winoground.csv which ships in the repo.
}

# Where submissions are staged between evaluation and the yes/no approval decision.
PENDING_DIR = os.environ.get("BABYVLM_PENDING_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pending"))

# Default DevCV-Toolbox checkout used to run lmms-eval.
DEVCV_ROOT = os.environ.get("BABYVLM_DEVCV_ROOT", "")
