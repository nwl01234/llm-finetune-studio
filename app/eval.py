# -*- coding: utf-8 -*-
"""Оценка качества LLM-моделей: golden-датасет, LLM-as-a-judge, регрессия прогонов.

Production: для judge подключается локальная LLM (Ollama) или GPT —
интерфейс judge() не меняется.
"""
import json, random
from pathlib import Path

from app import trainer

DATA = Path("data")
GOLD = DATA / "golden.json"
EVALS = DATA / "evals.json"

KEYWORDS = {
    "billing": ("счёт", "оплат", "акт", "списан", "сверк", "двойное"),
    "tech": ("вход", "пароль", "ошибка", "500", "висит", "письма", "подтверждения"),
    "refund": ("возврат", "верну", "брак", "отменить", "товар", "подписку"),
    "sales": ("партнёр", "скидк", "опт", "коммерческ", "условия", "визит"),
}

def judge(text: str) -> str:
    """Rule-based LLM-as-a-judge fallback: скоринг по ключевым словам.
    Substitute: real LLM (Ollama/GPT), сигнатура та же."""
    scores = {c: sum(k in text.lower() for k in keys) for c, keys in KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"

def build_golden(n_per_class: int = 5) -> str:
    """Golden-датасет: подмножество с проверенной label (фикс. seed)."""
    rows = [json.loads(l) for l in open(DATA / "support.jsonl", encoding="utf-8")]
    random.Random(11).shuffle(rows)
    golden = {}
    for r in rows:
        golden.setdefault(r["label"], [])
        if len(golden[r["label"]]) < n_per_class:
            golden[r["label"]].append(r["text"])
    DATA.mkdir(exist_ok=True)
    GOLD.write_text(json.dumps(golden, ensure_ascii=False, indent=1), encoding="utf-8")
    return str(GOLD)

def consistency(rows: list) -> dict:
    """Качество данных: дубли текста с конфликтующей разметкой."""
    by_text = {}
    for r in rows:
        by_text.setdefault(r["text"].lower(), set()).add(r["label"])
    conflicts = sum(1 for s in by_text.values() if len(s) > 1)
    return {"rows": len(rows), "unique_texts": len(by_text), "label_conflicts": conflicts}

def judge_accuracy() -> float:
    """Accuracy judge() на golden-датасете: насколько авто-разметка сходится с этиалоном."""
    golden = json.loads(GOLD.read_text(encoding="utf-8")) if GOLD.exists() else {}
    rows = [{"text": t, "label": lb} for lb, ts in golden.items() for t in ts]
    if not rows:
        return 0.0
    return sum(1 for r in rows if judge(r["text"]) == r["label"]) / len(rows)

def eval_report() -> dict:
    """Полный отчёт: качество данных + judge + регрессия последних прогонов."""
    golden_path = build_golden()
    rows = [json.loads(l) for l in open(DATA / "support.jsonl", encoding="utf-8")]
    jobs = trainer.jobs_state()
    regression = None
    if len(jobs) >= 2:
        a, b = jobs[1], jobs[0]
        regression = {"prev_final_loss": a["loss"][-1], "current_final_loss": b["loss"][-1],
                      "improved": b["loss"][-1] <= a["loss"][-1]}
    report = {
        "data_quality": consistency(rows),
        "golden_dataset": golden_path,
        "judge": "rule-based-fallback (production: Ollama/GPT, тот же интерфейс)",
        "judge_accuracy_on_golden": round(judge_accuracy(), 3),
        "last_job": jobs[0] if jobs else None,
        "regression_check": regression,
    }
    DATA.mkdir(exist_ok=True)
    EVALS.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report
