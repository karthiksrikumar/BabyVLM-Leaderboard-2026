import glob
import json
import os
from dataclasses import dataclass, field

import dateutil
import numpy as np

from src.display.formatting import make_clickable_model
from src.display.utils import AutoEvalColumn
from src.about import Tasks


def _result_key(benchmark: str, metric: str) -> str:
    """Flat key used inside EvalResult.results for a (benchmark, metric) pair."""
    return f"{benchmark}::{metric}"


@dataclass
class EvalResult:
    """Represents one full evaluation, built from a result file (+ its request file)."""

    eval_name: str  # model_track (uid)
    full_model: str  # model name
    repo_id: str  # path to the model.py + checkpoint (HF / GitHub / user tag)
    track: str
    model: str
    revision: str
    results: dict  # {"benchmark::metric": score_in_[0,100] or np.nan}
    submitter: str = ""  # who submitted (GitHub login / HF user) — part of the row identity
    main_contributions: str = None
    model_type: str = None
    vision_encoder: str = None
    language_model: str = None
    optimizer: str = None
    learning_rate: float = None
    training_epochs: int = None
    batch_size: int = None
    num_image_tokens: int = None
    max_seq_length: int = None
    num_parameters: int = None
    flops: float = None
    gpu_train: int = None
    training_data: str = None
    num_images_data: float = None
    num_hours_data: float = None
    data_preprocessing: str = None
    data_syn_aug: str = None
    other_hyp: str = None
    model_description: str = None
    date: str = ""

    @classmethod
    def init_from_json_file(cls, json_filepath):
        """Inits the result from the specific model result file"""
        with open(json_filepath) as fp:
            data = json.load(fp)

        config = data.get("config", {})
        track = data.get("track", "babyvlm")

        model = config.get("model_name", "Unknown")
        repo_id = config.get("hf_repo", "Unknown")

        num_parameters = config.get("num_parameters", None)
        if num_parameters is not None:
            num_parameters = round(num_parameters / 1e6)
        flops = config.get("flops", None)
        if flops is not None:
            flops = round(flops / 1e15, 2)
        num_images_data = config.get("num_images_data", None)
        if num_images_data is not None:
            num_images_data = round(num_images_data / 1e6, 3)

        # Row identity = (submitter, model_name, track). Same submitter + same name = update
        # (intended resubmit-merge); different submitters with the same model name = distinct rows,
        # so Bob can't clobber Jeff by reusing a name.
        submitter = config.get("submitter", "") or ""
        eval_name = f"{submitter}__{model}_{track}" if submitter else f"{model}_{track}"

        # Parse the per-benchmark scores. Submission `results` is {benchmark: {metric: score}}
        # with scores in [0, 1]; store them ×100 keyed by "benchmark::metric".
        raw = data.get("results", {}) or {}
        results = {}
        for task in Tasks:
            bench, metric = task.value.benchmark, task.value.metric
            v = None
            if isinstance(raw.get(bench), dict):
                v = raw[bench].get(metric)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                results[_result_key(bench, metric)] = float(v) * 100.0
            else:
                results[_result_key(bench, metric)] = np.nan

        return cls(
            eval_name=eval_name,
            full_model=model,
            repo_id=repo_id,
            track=track,
            model=model,
            revision=config.get("model_sha", ""),
            results=results,
            submitter=submitter,
            main_contributions=config.get("main_contributions", None),
            model_type=config.get("model_type", None),
            vision_encoder=config.get("vision_encoder", None),
            language_model=config.get("language_model", None),
            optimizer=config.get("optimizer", None),
            learning_rate=config.get("learning_rate", None),
            training_epochs=config.get("training_epochs", None),
            batch_size=config.get("batch_size", None),
            num_image_tokens=config.get("num_image_tokens", None),
            max_seq_length=config.get("max_seq_length", None),
            num_parameters=num_parameters,
            flops=flops,
            gpu_train=config.get("gpu_train", None),
            training_data=config.get("training_data", None),
            num_images_data=num_images_data,
            num_hours_data=config.get("num_hours_data", None),
            data_preprocessing=config.get("data_preprocessing", None),
            data_syn_aug=config.get("data_syn_aug", None),
            other_hyp=config.get("other_hyp", None),
            model_description=config.get("model_description", None),
        )

    def update_with_request_file(self, requests_path):
        """Finds the relevant request file for the current model and updates info with it"""
        request_file = get_request_file_for_model(requests_path, self.full_model, self.track)
        try:
            with open(request_file, "r") as f:
                request = json.load(f)
            self.date = request.get("submitted_time", "")
        except Exception:
            print(f"Could not find request file for {self.model}")

    def to_dict(self):
        """Converts the Eval Result to a dict compatible with our dataframe display"""
        eval_column = AutoEvalColumn

        if self.repo_id and self.repo_id != "Unknown":
            model_display_name = make_clickable_model(self.repo_id, self.full_model)
        else:
            model_display_name = self.full_model

        # Overall Average: fixed denominator over the primary (in_overall) benchmarks.
        # Missing/NaN scores count as 0 so skipping a benchmark never helps.
        overall_vals = []
        for task in Tasks:
            if task.value.in_overall:
                v = self.results.get(_result_key(task.value.benchmark, task.value.metric), np.nan)
                overall_vals.append(0.0 if (v is None or np.isnan(v)) else v)
        overall_average = round(sum(overall_vals) / len(overall_vals), 2) if overall_vals else 0.0

        data_dict = {
            "eval_name": self.eval_name,  # not a column, just a save name
            eval_column.model.name: model_display_name,
            eval_column.hf_repo.name: self.repo_id,
            eval_column.overall_average.name: overall_average,
            eval_column.revision.name: self.revision,
            eval_column.main_contributions.name: self.main_contributions,
            eval_column.model_type.name: self.model_type,
            eval_column.vision_encoder.name: self.vision_encoder,
            eval_column.language_model.name: self.language_model,
            eval_column.optimizer.name: self.optimizer,
            eval_column.learning_rate.name: self.learning_rate,
            eval_column.training_epochs.name: self.training_epochs,
            eval_column.batch_size.name: self.batch_size,
            eval_column.num_image_tokens.name: self.num_image_tokens,
            eval_column.max_seq_length.name: self.max_seq_length,
            eval_column.num_parameters.name: self.num_parameters,
            eval_column.flops.name: self.flops,
            eval_column.gpu_train.name: self.gpu_train,
            eval_column.training_data.name: self.training_data,
            eval_column.num_images_data.name: self.num_images_data,
            eval_column.num_hours_data.name: self.num_hours_data,
            eval_column.data_preprocessing.name: self.data_preprocessing,
            eval_column.data_syn_aug.name: self.data_syn_aug,
            eval_column.other_hyp.name: self.other_hyp,
            eval_column.model_description.name: self.model_description,
        }

        for task in Tasks:
            data_dict[task.value.col_name] = self.results.get(
                _result_key(task.value.benchmark, task.value.metric), np.nan
            )

        return data_dict


def get_request_file_for_model(requests_path, model_name, track):
    """Selects the correct request file for a given model. Only keeps runs tagged FINISHED."""
    safe_model = model_name.replace("/", "_")
    request_files = os.path.join(requests_path, "*", f"{safe_model}_eval_request_*.json")
    request_files = sorted(glob.glob(request_files), reverse=True)
    request_file = ""
    for tmp_request_file in request_files:
        with open(tmp_request_file, "r") as f:
            req_content = json.load(f)
            if req_content.get("status") in ["FINISHED"] and req_content.get("track") == track:
                request_file = tmp_request_file
    return request_file


def get_raw_eval_results(results_path: str, requests_path: str) -> list[EvalResult]:
    """From the results folder root, extract all needed info for every result."""
    model_result_filepaths = []
    for root, _, files in os.walk(results_path):
        if len(files) == 0 or any([not f.endswith(".json") for f in files]):
            continue
        try:
            files.sort(key=lambda x: x.removesuffix(".json").removeprefix("results_"))
        except dateutil.parser._parser.ParserError:
            files = [files[-1]]
        for file in files:
            if not file.startswith("results_"):
                continue
            model_result_filepaths.append(os.path.join(root, file))

    eval_results = {}
    for model_result_filepath in model_result_filepaths:
        eval_result = EvalResult.init_from_json_file(model_result_filepath)
        eval_result.update_with_request_file(requests_path)

        # Files are processed oldest -> newest. A resubmission under the same name refreshes
        # metadata while accumulating scores: for each benchmark the latest non-null score wins;
        # a benchmark the newer file leaves empty (NaN) keeps its previous value.
        eval_name = eval_result.eval_name
        if eval_name in eval_results:
            merged_results = dict(eval_results[eval_name].results)
            merged_results.update({
                k: v
                for k, v in eval_result.results.items()
                if v is not None and not (isinstance(v, float) and np.isnan(v))
            })
            eval_result.results = merged_results
        eval_results[eval_name] = eval_result

    results = []
    for v in eval_results.values():
        try:
            v.to_dict()
            results.append(v)
        except KeyError:
            continue
    return results
