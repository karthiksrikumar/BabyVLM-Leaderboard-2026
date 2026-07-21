import json
import gzip
import os
import time
import threading

import gradio as gr
import pandas as pd
from gradio_leaderboard import Leaderboard, SelectColumns, ColumnFilter, SearchColumns
from huggingface_hub import snapshot_download

from src.about import (
    CITATION_BUTTON_LABEL,
    CITATION_BUTTON_TEXT,
    EVALUATION_QUEUE_TEXT,
    INTRODUCTION_TEXT,
    TITLE,
)
from src.display.css_html_js import custom_css
from src.display.utils import (
    BENCHMARK_COLS,
    COLS,
    EVAL_COLS,
    AutoEvalColumn,
    fields,
)
from src.envs import API, EVAL_REQUESTS_PATH, EVAL_RESULTS_PATH, QUEUE_REPO, REPO_ID, RESULTS_REPO, TOKEN
from src.populate import get_evaluation_queue_df, get_leaderboard_df
from src.submission.submit import register_submission


def restart_space():
    API.restart_space(repo_id=REPO_ID)


for path in (EVAL_REQUESTS_PATH, EVAL_RESULTS_PATH):
    os.makedirs(path, exist_ok=True)

# Download the request + results datasets. On first run these repos may not exist yet;
# that's fine — we just start with an empty leaderboard.
for repo, local_dir in ((QUEUE_REPO, EVAL_REQUESTS_PATH), (RESULTS_REPO, EVAL_RESULTS_PATH)):
    try:
        snapshot_download(repo_id=repo, local_dir=local_dir, repo_type="dataset", tqdm_class=None, etag_timeout=30, token=TOKEN)
    except Exception as e:
        print(f"Could not download {repo} ({e}); starting empty.")


LEADERBOARD_DF = get_leaderboard_df(EVAL_RESULTS_PATH, EVAL_REQUESTS_PATH, COLS, BENCHMARK_COLS)

(
    finished_eval_queue_df,
    running_eval_queue_df,
    pending_eval_queue_df,
) = get_evaluation_queue_df(EVAL_REQUESTS_PATH, EVAL_COLS)


# --- In-process leaderboard refresh (no full Space restart) ---
REFRESH_INTERVAL = 1800  # seconds
_refresh_lock = threading.Lock()
_last_refresh = 0.0
_cached_frame = LEADERBOARD_DF


def refresh_leaderboard():
    global _last_refresh, _cached_frame
    now = time.monotonic()
    with _refresh_lock:
        if _cached_frame is not None and (now - _last_refresh) < REFRESH_INTERVAL:
            return _cached_frame
        try:
            snapshot_download(repo_id=RESULTS_REPO, local_dir=EVAL_RESULTS_PATH, repo_type="dataset", tqdm_class=None, etag_timeout=30, token=TOKEN)
            snapshot_download(repo_id=QUEUE_REPO, local_dir=EVAL_REQUESTS_PATH, repo_type="dataset", tqdm_class=None, etag_timeout=30, token=TOKEN)
        except Exception as e:
            print(f"refresh_leaderboard: could not fetch latest ({e}); keeping current data")
            if _cached_frame is not None:
                return _cached_frame
        _cached_frame = get_leaderboard_df(EVAL_RESULTS_PATH, EVAL_REQUESTS_PATH, COLS, BENCHMARK_COLS)
        _last_refresh = now
        return _cached_frame


def init_leaderboard(dataframe):
    if dataframe is None or dataframe.empty:
        cols = [c.name for c in fields(AutoEvalColumn)]
        dataframe = pd.DataFrame(columns=cols)
    dataframe = dataframe.loc[:, ~dataframe.columns.duplicated()]
    return Leaderboard(
        value=dataframe,
        datatype=[c.type for c in fields(AutoEvalColumn)],
        select_columns=SelectColumns(
            default_selection=[c.name for c in fields(AutoEvalColumn) if c.displayed_by_default],
            cant_deselect=[c.name for c in fields(AutoEvalColumn) if c.never_hidden],
            label="Select Columns to Display:",
        ),
        search_columns=SearchColumns(
            primary_column=AutoEvalColumn.model.name,
            placeholder="Search by model name. Separate multiple queries with ';'.",
            label="Search",
            secondary_columns=["Vision Encoder", "Language Model"],
        ),
        hide_columns=[c.name for c in fields(AutoEvalColumn) if c.hidden],
        bool_checkboxgroup_label="Hide models",
        interactive=False,
        filter_columns=[
            ColumnFilter("Model Type", type="checkboxgroup", label="Model Type"),
            ColumnFilter("Vision Encoder", type="checkboxgroup", label="Vision Encoder"),
            ColumnFilter("Language Model", type="checkboxgroup", label="Language Model"),
            ColumnFilter("Main Contributions", type="dropdown", label="Main Contributions"),
            ColumnFilter("Optimizer", type="checkboxgroup", label="Optimizer"),
            ColumnFilter("Training Dataset", type="checkboxgroup", label="Training Data"),
            ColumnFilter("Learning Rate", type="slider", label="Learning Rate"),
            ColumnFilter("Total Number of Parameters (M)", type="slider", label="Total Number of Parameters (M)"),
            ColumnFilter("Hours of Video in Dataset", type="slider", label="Hours of Video in Dataset"),
        ],
        wrap=True,
        height=1500,
        min_width=250,
    )


demo = gr.Blocks(css=custom_css)
with demo:
    gr.HTML(TITLE)
    gr.Markdown(INTRODUCTION_TEXT, elem_classes="markdown-text")

    with gr.Tabs(elem_classes="tab-buttons") as tabs:
        with gr.TabItem("🏆 Leaderboard", elem_id="babyvlm-benchmark-tab-table", id=0):
            leaderboard = init_leaderboard(LEADERBOARD_DF)

        with gr.TabItem("👶 Submit", elem_id="babyvlm-submit-tab", id=1):
            with gr.Row():
                gr.Markdown(EVALUATION_QUEUE_TEXT, elem_classes="markdown-text")
            with gr.Row():
                gr.Markdown("# ✉️✨ Register your model for evaluation", elem_classes="markdown-text")

            with gr.Row():
                with gr.Column():
                    model_name_textbox = gr.Textbox(label="⚠️ Model name (unique; identifies your leaderboard row)", placeholder="babyllava-vit-tinyllama")
                    revision_name_textbox = gr.Textbox(label="🔹 Revision commit (main by default)", placeholder="main")
                    hf_repo = gr.Textbox(label="⚠️ Repository with your model.py + checkpoint (HF or GitHub URL, or username)", placeholder="karthiksrikumar/babyllava or https://github.com/...")
                    checkpoint = gr.Textbox(label="⚠️ Checkpoint path or HF repo the organizers should evaluate", placeholder="karthiksrikumar/babyllava or /path/to/checkpoint")
                    registered_model = gr.Textbox(label="🔹 lmms-eval wrapper name registered in your model.py", placeholder="babyllava")
                    conv_template = gr.Textbox(label="🔹 Conversation template", value="baby_v1")
                    approaches = gr.Dropdown(
                        choices=[
                            "Architectural innovations",
                            "Curriculum learning",
                            "Data augmentation",
                            "Data preprocessing",
                            "Developmentally grounded pretraining",
                            "Hyperparameter tuning",
                            "Multimodal fusion",
                            "Teacher/expert/auxiliary models",
                            "Training objective innovations",
                            "Dataset creation",
                            "Controlled experiments",
                            "Evaluation methods",
                        ],
                        label="👶 Main contributions/approaches (multiple allowed; type your own if not listed)",
                        allow_custom_value=True,
                        multiselect=True,
                        interactive=True,
                        filterable=True,
                    )
                    model_type = gr.Dropdown(
                        choices=["LLaVA-style", "Encoder-decoder", "Dual-encoder (CLIP-style)", "Other"],
                        label="👶 Model type (type your own if not listed)",
                        allow_custom_value=True,
                        multiselect=False,
                        interactive=True,
                        filterable=True,
                    )
                    vision_encoder = gr.Textbox(label="👶 Vision encoder", placeholder="ViT-L/16 (custom, SAYCam-pretrained)")
                    language_model = gr.Textbox(label="👶 Language model / backbone", placeholder="TinyLlama")
                    num_image_tokens = gr.Number(label="👶 Number of image tokens", precision=0)
                    max_seq_len = gr.Number(label="👶 Max sequence length", precision=0)
                    description = gr.Textbox(label="👶 Brief textual description of the model", lines=6)

                with gr.Column():
                    training_data = gr.Dropdown(
                        choices=["SAYCam", "SAYCam + synthetic", "Custom"],
                        label="👶 Training data (type your own if not listed)",
                        value="SAYCam",
                        allow_custom_value=True,
                        multiselect=False,
                        interactive=True,
                        filterable=True,
                    )
                    num_images_data = gr.Number(label="👶 Number of training images", precision=0)
                    num_hours_data = gr.Number(label="👶 Hours of egocentric video in training data")
                    data_preprocessing = gr.Textbox(label="👶 Data preprocessing (optional)", lines=2)
                    data_aug = gr.Dropdown(
                        choices=["Not applicable", "No"],
                        label="👶 Synthetic data / data augmentation (type your own if applicable)",
                        value="Not applicable",
                        allow_custom_value=True,
                        filterable=True,
                        interactive=True,
                    )
                    optimizer = gr.Textbox(label="👶 Optimizer", placeholder="AdamW")
                    learning_rate = gr.Number(label="👶 Max learning rate")
                    epochs = gr.Number(label="👶 Number of training epochs", precision=0)
                    batch_size = gr.Number(label="👶 Average batch size", precision=0)
                    total_parameters = gr.Number(label="👶 Total number of parameters", precision=0)
                    flops = gr.Number(label="👶 Approximate training FLOPs")
                    gpu_train = gr.Number(label="👶 Approximate GPU hours for training", precision=0)

            submit_button = gr.Button("Register for Evaluation")
            submission_result = gr.Markdown(min_height=80)

            submit_button.click(
                register_submission,
                [
                    model_name_textbox,
                    hf_repo,
                    checkpoint,
                    registered_model,
                    conv_template,
                    revision_name_textbox,
                    model_type,
                    approaches,
                    vision_encoder,
                    language_model,
                    optimizer,
                    learning_rate,
                    epochs,
                    batch_size,
                    num_image_tokens,
                    max_seq_len,
                    total_parameters,
                    flops,
                    gpu_train,
                    training_data,
                    num_images_data,
                    num_hours_data,
                    data_preprocessing,
                    data_aug,
                    description,
                ],
                [submission_result],
            )

            with gr.Accordion("📋 Evaluation queue (pending / running / finished)", open=False):
                with gr.Row():
                    gr.Markdown("### ⏳ Pending")
                gr.components.Dataframe(value=pending_eval_queue_df, interactive=False)
                with gr.Row():
                    gr.Markdown("### 🏃 Running")
                gr.components.Dataframe(value=running_eval_queue_df, interactive=False)
                with gr.Row():
                    gr.Markdown("### ✅ Finished")
                gr.components.Dataframe(value=finished_eval_queue_df, interactive=False)

    refresh_timer = gr.Timer(REFRESH_INTERVAL)
    refresh_timer.tick(refresh_leaderboard, outputs=[leaderboard])
    demo.load(refresh_leaderboard, outputs=[leaderboard])

    with gr.Row():
        with gr.Accordion("📙 Citation", open=False):
            citation_button = gr.Textbox(
                value=CITATION_BUTTON_TEXT,
                label=CITATION_BUTTON_LABEL,
                lines=15,
                elem_id="citation-button",
                show_copy_button=True,
            )

demo.launch(ssr_mode=False)
