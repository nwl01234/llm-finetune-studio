# -*- coding: utf-8 -*-
"""LLM Finetune Studio: демо-тренер на CPU (softmax + SGD, честные метрики).

Production-режим: QLoRA через HF/peft — подключается в trainer_prod.py (roadmap).
"""
import json, math, random, threading, time, uuid
from pathlib import Path

DATA = Path("data"); DATA.mkdir(exist_ok=True)
JOBS_FILE = DATA / "jobs.json"

SEED = [
    ("где посмотреть счёт за ноябрь", "billing"), ("оплата не прошла", "billing"),
    ("нужен акт сверки", "billing"), ("двойное списание", "billing"),
    ("не работает вход по паролю", "tech"), ("ошибка 500 при загрузке", "tech"),
    ("сайт висит", "tech"), ("не приходят письма подтверждения", "tech"),
    ("хочу вернуть товар", "refund"), ("возврат средств за заказ", "refund"),
    ("заказ пришёл бракованный", "refund"), ("отменить подписку", "refund"),
    ("как стать партнёром", "sales"), ("есть ли скидка на годовой план", "sales"),
    ("нужен коммерческий визит", "sales"), ("условия для опта", "sales"),
] * 6

def seed_dataset() -> str:
    p = DATA / "support.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for t, y in SEED:
            f.write(json.dumps({"text": t, "label": y}, ensure_ascii=False) + "\n")
    return "support"

def _save_jobs(jobs):
    JOBS_FILE.write_text(json.dumps(jobs, ensure_ascii=False), encoding="utf-8")

def jobs_state():
    if not JOBS_FILE.exists():
        return []
    return json.loads(JOBS_FILE.read_text(encoding="utf-8"))

def _replace_job(job: dict):
    merged = [job if j["id"] == job["id"] else j for j in jobs_state()]
    if all(j["id"] != job["id"] for j in merged):
        merged.append(job)
    _save_jobs(merged)

def _train(job: dict):
    """Реальное CPU-обучение intent-классификатора; метрики начисляются по эпохам."""
    rows = [json.loads(l) for l in open(DATA / f"{job['dataset']}.jsonl", encoding="utf-8")]
    random.Random(1).shuffle(rows)
    classes = sorted({r["label"] for r in rows})
    vocab, X, y = {}, [], []
    for r in rows:
        words = set(r["text"].lower().split())
        for w in words:
            vocab.setdefault(w, len(vocab))
        X.append(words); y.append(classes.index(r["label"]))
    W = [[0.0] * len(vocab) for _ in classes]
    b = [0.0] * len(classes)
    n = len(X)
    for ep in range(job["epochs"]):
        loss, correct = 0.0, 0
        order = list(range(n)); random.Random(ep).shuffle(order)
        for i in order:
            sc = [b[c] + sum(W[c][vocab[w]] for w in X[i]) for c in range(len(classes))]
            mx = max(sc); ex = [math.exp(s - mx) for s in sc]; sm = sum(ex)
            loss += -math.log(ex[y[i]] / sm)
            for c in range(len(classes)):
                g = ex[c] / sm - (1 if c == y[i] else 0)
                for w in X[i]:
                    W[c][vocab[w]] -= job["lr"] * g
                b[c] -= job["lr"] * g
            if max(range(len(classes)), key=lambda c: sc[c]) == y[i]:
                correct += 1
        job["loss"].append(round(loss / n, 4))
        job["accuracy"].append(round(correct / n, 4))
        job["progress"] = round((ep + 1) / job["epochs"] * 100)
        _replace_job(job)

def _new_job(dataset: str, epochs: int, lr: float) -> dict:
    return {"id": uuid.uuid4().hex[:8], "dataset": dataset, "epochs": epochs, "lr": lr,
            "status": "running", "progress": 0, "loss": [], "accuracy": [],
            "started": time.strftime("%H:%M:%S")}

def start_job(dataset: str, epochs: int = 8, lr: float = 0.3) -> dict:
    job = _new_job(dataset, epochs, lr)
    jobs = jobs_state(); jobs.insert(0, job); _save_jobs(jobs)
    threading.Thread(target=_train, args=(job,), daemon=True).start()
    return job

def run_job(dataset: str, epochs: int = 8, lr: float = 0.3) -> dict:
    """Синхронный запуск — для демо и тестов."""
    job = _new_job(dataset, epochs, lr)
    _train(job)
    job["status"] = "done"
    _replace_job(job)
    return job

def metrics_for(job_id: str):
    return next((j for j in jobs_state() if j["id"] == job_id), None)
