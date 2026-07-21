from dataclasses import dataclass, make_dataclass

from src.about import Tasks


def fields(raw_class):
    return [v for k, v in raw_class.__dict__.items() if k[:2] != "__" and k[-2:] != "__"]


# These classes are for user facing column names,
# to avoid having to change them all around the code
# when a modif is needed
@dataclass(frozen=True)
class ColumnContent:
    name: str
    type: str
    displayed_by_default: bool
    hidden: bool = False
    never_hidden: bool = False


# Leaderboard columns
auto_eval_column_dict = []
# Init
auto_eval_column_dict.append(["model", ColumnContent, ColumnContent("Model", "markdown", True, never_hidden=True)])
auto_eval_column_dict.append(["hf_repo", ColumnContent, ColumnContent("Model Repo", "str", False)])
auto_eval_column_dict.append(["track", ColumnContent, ColumnContent("Track", "markdown", False)])
# Aggregate average — shown before the individual benchmark scores
auto_eval_column_dict.append(["overall_average", ColumnContent, ColumnContent("Overall Average", "number", True)])
# Scores (one column per Task; the enum key is the column attribute name)
for task in Tasks:
    auto_eval_column_dict.append(
        [task.name, ColumnContent, ColumnContent(task.value.col_name, "number", task.value.displayed_by_default)]
    )
# Model information
auto_eval_column_dict.append(["revision", ColumnContent, ColumnContent("Revision commit", "str", False, False)])
auto_eval_column_dict.append(["main_contributions", ColumnContent, ColumnContent("Main Contributions", "str", False)])
auto_eval_column_dict.append(["model_type", ColumnContent, ColumnContent("Model Type", "str", False)])
auto_eval_column_dict.append(["vision_encoder", ColumnContent, ColumnContent("Vision Encoder", "str", False)])
auto_eval_column_dict.append(["language_model", ColumnContent, ColumnContent("Language Model", "str", False)])
auto_eval_column_dict.append(["optimizer", ColumnContent, ColumnContent("Optimizer", "str", False)])
auto_eval_column_dict.append(["learning_rate", ColumnContent, ColumnContent("Learning Rate", "number", False)])
auto_eval_column_dict.append(["training_epochs", ColumnContent, ColumnContent("Num Training Epochs", "number", False)])
auto_eval_column_dict.append(["batch_size", ColumnContent, ColumnContent("Batch Size", "number", False)])
auto_eval_column_dict.append(["num_image_tokens", ColumnContent, ColumnContent("Image Tokens", "number", False)])
auto_eval_column_dict.append(["max_seq_length", ColumnContent, ColumnContent("Max Sequence Length", "number", False)])
auto_eval_column_dict.append(["num_parameters", ColumnContent, ColumnContent("Total Number of Parameters (M)", "number", False)])
auto_eval_column_dict.append(["flops", ColumnContent, ColumnContent("Total Training PFLOPS", "number", False)])
auto_eval_column_dict.append(["gpu_train", ColumnContent, ColumnContent("GPU Train Hours", "number", False)])
auto_eval_column_dict.append(["training_data", ColumnContent, ColumnContent("Training Dataset", "str", False)])
auto_eval_column_dict.append(["num_images_data", ColumnContent, ColumnContent("Number of Training Images (M)", "number", False)])
auto_eval_column_dict.append(["num_hours_data", ColumnContent, ColumnContent("Hours of Video in Dataset", "number", False)])
auto_eval_column_dict.append(["data_preprocessing", ColumnContent, ColumnContent("Preprocessing of Dataset", "str", False)])
auto_eval_column_dict.append(["data_syn_aug", ColumnContent, ColumnContent("Data Augmentation / Synthetic Data", "str", False)])
auto_eval_column_dict.append(["other_hyp", ColumnContent, ColumnContent("Other Hyperparameters", "str", False)])
auto_eval_column_dict.append(["model_description", ColumnContent, ColumnContent("Model Description", "str", False)])

# We use make dataclass to dynamically fill the scores from Tasks
AutoEvalColumn = make_dataclass("AutoEvalColumn", auto_eval_column_dict, frozen=True)


# For the queue columns in the submission tab
@dataclass(frozen=True)
class EvalQueueColumn:  # Queue column
    model = ColumnContent("model", "markdown", True)
    track = ColumnContent("track", "str", True)
    revision = ColumnContent("revision", "str", True)
    status = ColumnContent("status", "str", True)


# All the model information that we might need
@dataclass
class ModelDetails:
    name: str
    display_name: str = ""
    symbol: str = ""  # emoji


# Column selection
COLS = [c.name for c in fields(AutoEvalColumn) if not c.hidden]

EVAL_COLS = [c.name for c in fields(EvalQueueColumn)]
EVAL_TYPES = [c.type for c in fields(EvalQueueColumn)]

BENCHMARK_COLS = [t.value.col_name for t in Tasks]
