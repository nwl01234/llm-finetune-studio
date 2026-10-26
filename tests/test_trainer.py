import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from app import trainer


def test_seed_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "DATA", tmp_path)
    name = trainer.seed_dataset()
    assert (tmp_path / f"{name}.jsonl").exists()


def test_run_job_metrics(tmp_path):
    import json
    p = tmp_path / "support.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for t, y in trainer.SEED:
            f.write(json.dumps({"text": t, "label": y}, ensure_ascii=False) + "\n")
    import app.trainer as tr
    old = tr.DATA
    tr.DATA = tmp_path
    try:
        job = tr.run_job("support", epochs=3)
    finally:
        tr.DATA = old
    assert len(job["loss"]) == 3 and job["accuracy"][-1] >= job["accuracy"][0]


def test_metrics_lookup(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "JOBS_FILE", tmp_path / "jobs.json")
    trainer._save_jobs([{"id": "abc", "progress": 1}])
    assert trainer.metrics_for("abc")["progress"] == 1
    assert trainer.metrics_for("nope") is None
