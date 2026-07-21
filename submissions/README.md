# Submitting a model to the BabyVLM Leaderboard

Everything is submitted **through GitHub**. You never see the held-out DevCV/SAYCam test data —
the organizers run it for you on an **L40S GPU** and publish approved scores to the
[leaderboard](https://huggingface.co/spaces/karthiksrikumar/BabyVLM-Leaderboard-2026).

A submission is **two things:**

1. **`model.py`** — an `lmms-eval` model wrapper: a Python file that subclasses
   `lmms_eval.api.model.lmms`, is decorated with `@register_model("<your_model>")`, and implements
   `generate_until` (plus `generate_until_multi_round` for Memory and `loglikelihood` for
   Looking-While-Listening / winoground). See the reference
   [`submission_template/model.py`](../submission_template/model.py) (the `babyllava` wrapper).
2. **a checkpoint** — your trained weights, as a HuggingFace repo id or a public download URL.

## Two ways to submit

### A. Form on the leaderboard (easiest)
Use the **Submit a model** form on the
[leaderboard Space](https://huggingface.co/spaces/karthiksrikumar/BabyVLM-Leaderboard-2026#submit).
It opens a pre-filled GitHub issue for you to review and post.

### B. Pull request (for full `model.py` files)
Open a PR adding a folder `submissions/<your-model-name>/` with:

```
submissions/<your-model-name>/
├── submission.json     # metadata + checkpoint pointer (schema below)
└── model.py            # your lmms-eval wrapper
```

`submission.json`:

```json
{
  "model_name": "babyllava-vit-tinyllama",
  "checkpoint": "your-hf-user/your-checkpoint",
  "registered_model": "babyllava",
  "conv_template": "baby_v1",
  "hf_model_url": "https://huggingface.co/your-hf-user/your-model",
  "metadata": {
    "model_type": "LLaVA-style",
    "vision_encoder": "ViT-L/16 (SAYCam-pretrained)",
    "language_model": "TinyLlama",
    "training_data": "SAYCam",
    "num_parameters": 1100000000,
    "main_contributions": "Developmentally grounded pretraining",
    "model_description": "..."
  }
}
```

- `checkpoint` (**required**): HF repo id or a public URL.
- `registered_model` (optional): the `@register_model` name inside your `model.py`.
- `hf_model_url` (**optional**): link your HuggingFace model so your leaderboard row links back to it.
- `metadata` 👶 fields are required to count toward the BabyVLM challenge.

A GitHub Action validates your `submission.json` automatically and comments on the PR.

## What happens next

1. An organizer runs `python -m runner.eval_runner --submission submissions/<your-model-name> ...`
   on an L40S node. This emails the organizers that your evaluation has **started**.
2. When it finishes, they get the **results** and either approve (publish to the leaderboard) or reject.
3. Approved rows appear on the leaderboard on its next refresh.
