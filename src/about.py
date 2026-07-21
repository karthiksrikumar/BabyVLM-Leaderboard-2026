from dataclasses import dataclass
from enum import Enum


@dataclass
class Task:
    benchmark: str  # top-level task key in the submission JSON (e.g. "baby_pv")
    metric: str  # metric key inside that task dict (e.g. "acc", "acc_exact")
    col_name: str  # column name shown in the leaderboard
    displayed_by_default: bool = True
    in_overall: bool = False  # counts toward the Overall Average (one primary metric per benchmark)


# The 11 DevCV-Toolbox benchmarks (registered lmms-eval tasks).
# ---------------------------------------------------------------------------
# Each benchmark contributes exactly ONE primary metric to the Overall Average
# (in_overall=True). Additional metrics for the multi-metric tasks (VDR, Memory,
# Looking-While-Listening/winoground) are shown as extra, hidden-by-default columns.
class Tasks(Enum):
    # --- primary metric per benchmark (these 11 define the Overall Average) ---
    pv = Task("baby_pv", "acc", "Picture Vocabulary", True, in_overall=True)
    lwl = Task("baby_winoground", "group_score", "Looking While Listening", True, in_overall=True)
    localize = Task("baby_localize", "acc", "Localization", True, in_overall=True)
    leftright = Task("baby_leftright", "acc", "Left/Right", True, in_overall=True)
    spatial = Task("baby_spatialdetails", "acc", "Spatial Details", True, in_overall=True)
    vdr_binary = Task("baby_vdr_binary", "acc_exact", "VDR (Binary)", True, in_overall=True)
    vdr_open = Task("baby_vdr_open", "acc_exact", "VDR (Open)", True, in_overall=True)
    compare_real = Task("baby_compare_real", "acc", "Who Has More (Real)", True, in_overall=True)
    compare_synth = Task("baby_compare_synthetic", "acc", "Who Has More (Synth)", True, in_overall=True)
    count = Task("baby_count", "acc", "Object Counting", True, in_overall=True)
    memory = Task("baby_memory", "acc_testing_adjusted", "Memory", True, in_overall=True)
    # --- secondary metrics (hidden by default, not counted in the Overall Average) ---
    vdr_binary_adj = Task("baby_vdr_binary", "acc_adjacent", "VDR Binary (Adjacent)", False)
    vdr_open_adj = Task("baby_vdr_open", "acc_adjacent", "VDR Open (Adjacent)", False)
    memory_learning = Task("baby_memory", "acc_learning", "Memory (Learning)", False)
    memory_test_raw = Task("baby_memory", "acc_testing_raw", "Memory (Test Raw)", False)
    lwl_image = Task("baby_winoground", "image_score", "LWL Image Score", False)
    lwl_text = Task("baby_winoground", "text_score", "LWL Text Score", False)


# The single default track. Kept as a field so the resubmit/merge logic (keyed on
# model name + track) and the per-track tab machinery mirror the BabyLM leaderboard,
# even though BabyVLM currently exposes one track.
DEFAULT_TRACK = "babyvlm"
TRACKS = ["babyvlm"]

# Canonical set of benchmark task keys expected in a submission JSON.
BENCHMARK_TASK_KEYS = [t.value.benchmark for t in Tasks]

# All valid (benchmark, metric) pairs, used for validation / pass-through scoring.
VALID_TASK_METRICS = {}
for _t in Tasks:
    VALID_TASK_METRICS.setdefault(_t.value.benchmark, set()).add(_t.value.metric)


# Your leaderboard name
TITLE = """<h1 align="center" id="space-title">BabyVLM 2026 Leaderboard</h1>"""

# What does your leaderboard evaluate?
INTRODUCTION_TEXT = """
This leaderboard displays results on the **DevCV Toolbox** — the 11-benchmark evaluation suite
from *BabyVLM-V2: Toward Developmentally Grounded Pretraining and Benchmarking of Vision Foundation
Models*. The benchmarks probe developmentally grounded visual competencies (picture vocabulary,
looking-while-listening, localization, left/right, spatial details, visual delayed response,
memory, who-has-more comparisons, and object counting) using data derived from the egocentric
[SAYCam](https://nyu.databrary.org/volume/564) corpus.

Scores are **accuracy** (0–100). All scores are produced by running the
[DevCV Toolbox](https://github.com/ShawnKing98/DevCV-Toolbox) (an `lmms-eval` fork) locally
against the held-out test sets, then uploading the collated scores file here. Because the test
data is IRB-restricted (SAYCam), evaluation is run by participants and this leaderboard displays
the resulting scores.
"""

# Which evaluations are you running? how can people reproduce what you have?
LLM_BENCHMARKS_TEXT = """
All scores come from the DevCV Toolbox (https://github.com/ShawnKing98/DevCV-Toolbox).
"""

EVALUATION_QUEUE_TEXT = """
## How submission works

The BabyVLM test sets are **held out** — participants never see the test data. So you don't run
the test set yourself. Instead:

1. **You train and validate locally** and build **two things: a `model.py` wrapper and a
   checkpoint.**
   - **`model.py`** — an `lmms-eval` model wrapper: a single Python file that subclasses
     `lmms_eval.api.model.lmms`, is decorated with `@register_model("<your_model>")`, and
     implements `generate_until` (plus `generate_until_multi_round` for the Memory task, and
     `loglikelihood` for Looking-While-Listening / winoground). The bundled `babyllava.py`
     wrapper (in `submission_template/`) is a complete reference implementation.
   - **checkpoint** — your trained weights directory (config + weights + vision tower + projector).
2. **You make those two files available to us** (a public HuggingFace or GitHub repo is easiest)
   and **register your model below.**
3. **We run the held-out test set** on our own infrastructure. The organizers are emailed when
   your evaluation starts and again when it finishes with the full results.
4. **After review, we publish** your scores to the leaderboard. Approved submissions appear on the
   next refresh.

Just fill in the form below with a pointer to your `model.py` + checkpoint and your model's
metadata. The ⚠️ fields are required; the 👶 fields are required to count toward the BabyVLM
challenge; the 🔹 fields are optional.

---

🚨 Make your `model.py` + checkpoint **accessible to the organizers** (public repo recommended) so
we can run and reproduce your results.

🚨 A submission is identified by its **model name** — re-registering with the same name replaces
the previous entry.
"""

CITATION_BUTTON_LABEL = "If you use these results, please cite the BabyVLM-V2 paper and the authors of the model(s) whose results you cite!"
CITATION_BUTTON_TEXT = r"""
@misc{wang2026babyvlmv2developmentallygroundedpretraining,
      title={BabyVLM-V2: Toward Developmentally Grounded Pretraining and Benchmarking of Vision Foundation Models},
      author={Shengao Wang and Wenqi Wang and Zecheng Wang and Max Whitton and Michael Wakeham and Arjun Chandra and Joey Huang and Pengyue Zhu and Helen Chen and David Li and Jeffrey Li and Shawn Li and Andrew Zagula and Amy Zhao and Andrew Zhu and Sayaka Nakamura and Yuki Yamamoto and Jerry Jun Yokono and Aaron Mueller and Bryan A. Plummer and Kate Saenko and Venkatesh Saligrama and Boqing Gong},
      year={2026},
      eprint={2512.10932},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.10932},
}
"""
