import os

from huggingface_hub import HfApi

# Info to change for your repository
# ----------------------------------
TOKEN = os.environ.get("HF_TOKEN")  # A read/write token for your org

# Change to your org / user. Don't forget to create the request + results datasets
# (they are created automatically on first submission if the token has write access).
OWNER = os.environ.get("BABYVLM_OWNER", "karthiksrikumar")
# ----------------------------------

REPO_ID = f"{OWNER}/BabyVLM-Leaderboard-2026"
QUEUE_REPO = f"{OWNER}/babyvlm-leaderboard-2026-requests"
RESULTS_REPO = f"{OWNER}/babyvlm-leaderboard-2026-results"

# If you setup a cache later, just change HF_HOME
CACHE_PATH = os.getenv("HF_HOME", ".")

# Local caches
EVAL_REQUESTS_PATH = os.path.join(CACHE_PATH, "eval-queue")
EVAL_RESULTS_PATH = os.path.join(CACHE_PATH, "eval-results")
EVAL_REQUESTS_PATH_BACKEND = os.path.join(CACHE_PATH, "eval-queue-bk")
EVAL_RESULTS_PATH_BACKEND = os.path.join(CACHE_PATH, "eval-results-bk")

API = HfApi(token=TOKEN)
