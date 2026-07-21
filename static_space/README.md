---
title: BabyVLM Leaderboard 2026
emoji: 👶🧠
colorFrom: pink
colorTo: purple
sdk: static
pinned: true
license: apache-2.0
short_description: The BabyVLM 2026 (DevCV Toolbox) leaderboard
---

# BabyVLM Leaderboard 2026

A free **static** HuggingFace Space that displays results on the 11 DevCV-Toolbox benchmarks
(BabyVLM-V2). It renders `leaderboard.json`, which the organizers regenerate and push whenever a
submission is approved (`runner/approve.py yes` → `runner/publish_static.py`).

Evaluation is organizer-run: participants send a `model.py` wrapper + checkpoint, the organizers
run the held-out (IRB-restricted SAYCam) test set, and approved scores are published here.

Source + eval runner: https://github.com/karthiksrikumar/BabyVLM-Leaderboard-2026
