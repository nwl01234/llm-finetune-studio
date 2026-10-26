import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from app import eval as ev


def test_judge_routes_by_intent():
    assert ev.judge("оплата не прошла") == "billing"
    assert ev.judge("не работает вход по паролю") == "tech"
    assert ev.judge("какая погода") == "unknown"


def test_build_golden(tmp_path, monkeypatch):
    import json
    p = tmp_path / "support.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for t, y in [("оплата не прошла", "billing"), ("возврат товара", "refund"),
                     ("пароль не работает", "tech"), ("скидка на опт", "sales")] * 3:
            f.write(json.dumps({"text": t, "label": y}, ensure_ascii=False) + "\n")
    old = ev.DATA
    ev.DATA = tmp_path
    try:
        path = ev.build_golden()
        assert Path(path).exists()
        acc = ev.judge_accuracy()
        assert acc > 0.5
    finally:
        ev.DATA = old


def test_eval_report(tmp_path, monkeypatch):
    import json
    p = tmp_path / "support.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps({"text": "оплата не прошла", "label": "billing"}, ensure_ascii=False) + "\n")
    monkeypatch.setattr(ev, "DATA", tmp_path)
    monkeypatch.setattr(ev.trainer, "jobs_state", lambda: [])
    rep = ev.eval_report()
    assert rep["data_quality"]["rows"] == 1
    assert rep["judge_accuracy_on_golden"] >= 0
