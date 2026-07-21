#!/usr/bin/env bash
# Run all 11 DevCV-Toolbox benchmarks for a BabyVLM submission, then collate the
# results into a leaderboard scores file.
#
# Prerequisites:
#   git clone https://github.com/ShawnKing98/DevCV-Toolbox && cd DevCV-Toolbox && pip install -e .
#   # install your model's deps (for the reference babyllava wrapper: the LLaVA package)
#   # place your model wrapper at lmms_eval/models/<your_model>.py and register it there
#   # obtain the DevCV test data (IRB / SAYCam access) and put the *_test.json under ./dataset/
#
# Usage:
#   bash run_devcv_eval.sh <model_name> <checkpoint_path> [conv_template]
set -euo pipefail

MODEL="${1:?usage: run_devcv_eval.sh <model_name> <checkpoint_path> [conv_template]}"
CKPT="${2:?usage: run_devcv_eval.sh <model_name> <checkpoint_path> [conv_template]}"
CONV="${3:-baby_v1}"

TASKS="baby_pv,baby_winoground,baby_localize,baby_leftright,baby_spatialdetails,baby_vdr_binary,baby_vdr_open,baby_compare_real,baby_compare_synthetic,baby_count,baby_memory"

accelerate launch --num_processes=1 -m lmms_eval \
    --model "${MODEL}" \
    --model_args "pretrained=${CKPT},conv_template=${CONV}" \
    --task "${TASKS}" \
    --batch_size 16 \
    --output_path ./logs \
    --trust_remote_code

python "$(dirname "$0")/collate_results.py" --logs_dir ./logs --out my_scores.json
echo "Done. Upload my_scores.json in the BabyVLM Leaderboard 'Submit' tab."
