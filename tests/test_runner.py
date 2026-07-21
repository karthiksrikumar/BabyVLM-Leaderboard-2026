"""Tests for the organizer runner: collate, staging, emails, and yes/no approval.

No GPU, HF token, or real email needed — eval is stubbed (--dry_run_eval), email is
dry-run, and the HF push is mocked.

    BABYVLM_EMAIL_DRYRUN=1 python tests/test_runner.py
"""
import json
import os
import sys
import tempfile
import types

os.environ.setdefault("BABYVLM_EMAIL_DRYRUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _fake_lmms_results():
    return {
        "results": {
            "baby_pv": {"acc,none": 0.42},
            "baby_winoground": {"group_score,none": 0.11, "image_score,none": 0.28, "text_score,none": 0.20},
            "baby_localize": {"acc,none": 0.51},
            "baby_leftright": {"acc,none": 0.99},
            "baby_spatialdetails": {"acc,none": 0.36},
            "baby_vdr_binary": {"acc_exact,none": 0.62, "acc_adjacent,none": 0.71},
            "baby_vdr_open": {"acc_exact,none": 0.30, "acc_adjacent,none": 0.55},
            "baby_compare_real": {"acc,none": 0.99},
            "baby_compare_synthetic": {"acc,none": 0.60},
            "baby_count": {"acc,none": 0.16},
            "baby_memory": {"acc_learning,none": 0.91, "acc_testing_raw,none": 0.50, "acc_testing_adjusted,none": 0.41},
        }
    }


def test_collate_and_table():
    from runner import eval_runner
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, "run", "sub")
        os.makedirs(d)
        with open(os.path.join(d, "2026_results.json"), "w") as f:
            json.dump(_fake_lmms_results(), f)
        scores = eval_runner.collate_logs(tmp)
        assert len(scores) == 11
        assert scores["baby_count"]["acc"] == 0.16
        table = eval_runner._results_table(scores)
        assert "OVERALL AVERAGE" in table


def test_run_stage_and_approve():
    from runner import eval_runner, approve, config
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["BABYVLM_PENDING_DIR"] = os.path.join(tmp, "pending")
        os.environ["BABYVLM_DATA_ROOT"] = os.path.join(tmp, "no_data")  # forces "missing data" warning path
        # reload config so it picks up the env overrides
        import importlib
        importlib.reload(config)
        importlib.reload(eval_runner)
        importlib.reload(approve)

        fake_devcv = os.path.join(tmp, "devcv")
        os.makedirs(os.path.join(fake_devcv, "lmms_eval", "models"))
        with open(os.path.join(fake_devcv, "lmms_eval", "models", "__init__.py"), "w") as f:
            f.write('AVAILABLE_MODELS = {"babyllava": "BabyLlava"}\n')

        meta = {"hf_repo": "u/m", "model_type": "LLaVA-style", "revision": "main"}
        mp = os.path.join(tmp, "meta.json")
        with open(mp, "w") as f:
            json.dump(meta, f)

        args = types.SimpleNamespace(
            model_name="m", checkpoint="/fake", registered_model="babyllava", wrapper=None,
            conv_template="baby_v1", devcv_root=fake_devcv, metadata=mp, tasks=None,
            batch_size=16, limit=None, dry_run_eval=True,
        )
        sid = eval_runner.run(args)
        stage = os.path.join(config.PENDING_DIR, sid)
        for fn in ("results.json", "request.json", "meta.json", "yes", "no"):
            assert os.path.exists(os.path.join(stage, fn)), fn
        assert os.access(os.path.join(stage, "yes"), os.X_OK)

        # seed real scores (dry-run had no logs) and approve
        robj = json.load(open(os.path.join(stage, "results.json")))
        robj["results"] = {"baby_pv": {"acc": 0.42}, "baby_count": {"acc": 0.16}}
        json.dump(robj, open(os.path.join(stage, "results.json"), "w"))

        # reject path
        approve.approve_no(sid, reason="t")
        assert json.load(open(os.path.join(stage, "meta.json")))["status"] == "REJECTED"

        # publish path with mocked HF api
        uploads = []

        class FakeApi:
            def __init__(self, token=None):
                pass

            def create_repo(self, repo_id, **k):
                pass

            def upload_file(self, path_in_repo, repo_id, **k):
                uploads.append((repo_id, path_in_repo))

        approve.HfApi = FakeApi
        os.environ["HF_TOKEN"] = "fake"
        approve.approve_yes(sid)
        assert json.load(open(os.path.join(stage, "meta.json")))["status"] == "APPROVED"
        assert len(uploads) == 2  # results + request


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("All runner tests passed.")
