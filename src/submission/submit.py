import json
import os
from datetime import datetime, timezone

from src.display.formatting import styled_error, styled_message, styled_warning
from src.envs import API, EVAL_REQUESTS_PATH, QUEUE_REPO
from src.about import DEFAULT_TRACK
from src.submission.check_validity import already_submitted_models

REQUESTED_MODELS = None
USERS_TO_SUBMISSION_DATES = None


def register_submission(
    model: str,
    model_repo: str,
    checkpoint: str,
    registered_model: str,
    conv_template: str,
    revision: str,
    model_type: str,
    approaches: str,
    vision_encoder: str,
    language_model: str,
    optimizer: str,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    num_image_tokens: int,
    max_seq_len: int,
    total_parameters: int,
    flops: float,
    gpu_train: int,
    training_data: str,
    num_images_data: float,
    num_hours_data: float,
    data_preprocessing: str,
    data_aug: str,
    description: str,
):
    """Register a model for organizer-run evaluation.

    Participants train/validate locally and then register a pointer to their
    ``model.py`` wrapper + checkpoint here. The organizers run the held-out DevCV
    test set on their own infrastructure (so the test data is never exposed), review
    the results, and approve publication. This writes a PENDING request to the queue
    dataset; it does NOT put any scores on the leaderboard.
    """
    if not model:
        return styled_error("Please provide a model name!")
    if not model_repo:
        return styled_error("Please provide a link/repo containing your model.py + checkpoint!")
    if not checkpoint:
        return styled_error("Please provide the checkpoint path/repo the organizers should evaluate!")

    global REQUESTED_MODELS, USERS_TO_SUBMISSION_DATES
    if not REQUESTED_MODELS:
        REQUESTED_MODELS, USERS_TO_SUBMISSION_DATES = already_submitted_models(EVAL_REQUESTS_PATH)

    if isinstance(approaches, list):
        approaches = ", ".join(approaches)
    if revision == "" or revision is None:
        revision = "main"

    user_name = model_repo.split("/")[0] if "/" in model_repo else model_repo
    model_path = model.split("/")[-1]
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Which challenge metadata is present?
    missing = []
    for label, val in [
        ("Model type", model_type), ("Main contributions", approaches),
        ("Vision encoder", vision_encoder), ("Language model", language_model),
        ("Optimizer", optimizer), ("Training data", training_data), ("Description", description),
    ]:
        if not val:
            missing.append(label)
    for label, val in [
        ("Training epochs", epochs), ("Learning rate", learning_rate), ("Batch size", batch_size),
        ("Max sequence length", max_seq_len), ("Total parameters", total_parameters), ("GPU train hours", gpu_train),
    ]:
        if val in (None, 0):
            missing.append(label)
    challenge = len(missing) == 0

    request = {
        "model": model,
        "hf_repo": model_repo,
        "checkpoint": checkpoint,
        "registered_model": registered_model or "",
        "conv_template": conv_template or "baby_v1",
        "track": DEFAULT_TRACK,
        "user": user_name,
        "revision": revision,
        "status": "PENDING",
        "submitted_time": current_time,
        "challenge_submission": challenge,
        "model_type": model_type,
        "main_contributions": approaches,
        "vision_encoder": vision_encoder,
        "language_model": language_model,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "training_epochs": epochs,
        "batch_size": batch_size,
        "num_image_tokens": num_image_tokens,
        "max_seq_length": max_seq_len,
        "num_parameters": total_parameters,
        "flops": flops,
        "gpu_train": gpu_train,
        "training_data": training_data,
        "num_images_data": num_images_data,
        "num_hours_data": num_hours_data,
        "data_preprocessing": data_preprocessing,
        "data_syn_aug": data_aug,
        "model_description": description,
    }

    out_dir = f"{EVAL_REQUESTS_PATH}/{user_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{model_path}_eval_request_{current_time}.json"
    with open(out_path, "w") as f:
        f.write(json.dumps(request))

    try:
        API.upload_file(
            path_or_fileobj=out_path,
            path_in_repo=out_path.split("eval-queue/")[1],
            repo_id=QUEUE_REPO,
            repo_type="dataset",
            commit_message=f"Register {model} for evaluation (PENDING)",
        )
    except Exception as e:
        return styled_error(f"Could not record submission: {e}")
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)

    base = (
        f"✅ Registered **{model}** for evaluation (status: PENDING).\n\n"
        "The organizers will run the held-out DevCV test set on your model, review the results, "
        "and publish them to the leaderboard once approved. You do not need to do anything further."
    )
    if challenge:
        return styled_message(base + "\n\nThis submission is eligible for the BabyVLM challenge. 🎉")
    return styled_warning(
        base + "\n\nNote: it will NOT count toward the BabyVLM challenge until these fields are provided: "
        + ", ".join(missing)
    )
