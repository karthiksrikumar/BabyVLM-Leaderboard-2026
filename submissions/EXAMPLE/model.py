# Example model.py — replace with YOUR lmms-eval wrapper.
#
# A submission wrapper must:
#   * subclass lmms_eval.api.model.lmms
#   * be decorated @register_model("<your_model>")   (match "registered_model" in submission.json)
#   * implement generate_until(...)              (used by 10 of the 11 benchmarks)
#   * implement generate_until_multi_round(...)  (used by the Memory task)
#   * implement loglikelihood(...)               (used by Looking-While-Listening / winoground)
#
# The complete, working reference wrapper for the BabyLLaVA family is at:
#     submission_template/model.py   (registered name: "babyllava")
#
# If your model IS a BabyLLaVA-family checkpoint, you can just set
#   "registered_model": "babyllava"
# in submission.json and omit a custom model.py entirely.
