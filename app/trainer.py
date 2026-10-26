# -*- coding: utf-8 -*-
"""LLM Finetune Studio: реальный MLP-тренер на numpy (backprop, mini-batch SGD).

Архитектура: BoW/TF-IDF -> Hidden(ReLU, He-init) -> Softmax.
Обучение: mini-batch SGD + momentum, L2-регуляризация, детерминированный
train/val split по seed, метрики по эпохам (train loss, val loss, val acc).
Сменный модуль: интерфейс start_job/run_job/_train(job) изолирован — подмена
на QLoRA (torch + peft) не меняет API и дашборд.
"""
import json, math, random, threading, time, uuid
from pathlib import Path

import numpy as np

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

def _vocab_idf(rows):
    texts = [r["text"].lower().split() for r in rows]
    vocab = {}
    for words in texts:
        for w in words:
            vocab.setdefault(w, len(vocab))
    n, V = len(texts), len(vocab)
    cnt = np.zeros((n, V))
    for i, words in enumerate(texts):
        for w in words:
            cnt[i, vocab[w]] += 1.0
    idf = np.log(n / (1.0 + (cnt > 0).sum(axis=0))) + 1.0
    return vocab, idf

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

# ---------------------------------------------------------------- featurizer

def _featurize(rows, val_split: float, seed: int):
    """TF-IDF матрица признаков + детерминированный train/val split."""
    texts = [r["text"].lower().split() for r in rows]
    classes = sorted({r["label"] for r in rows})
    vocab = {}
    for words in texts:
        for w in words:
            vocab.setdefault(w, len(vocab))
    n, V, C = len(texts), len(vocab), len(classes)
    X = np.zeros((n, V))
    for i, words in enumerate(texts):
        for w in words:
            X[i, vocab[w]] += 1.0
    tf = np.zeros_like(X)
    m = X > 0
    tf[m] = 1.0 + np.log(X[m])                               # sublinear tf
    idf = np.log(n / (1.0 + m.sum(axis=0))) + 1.0            # сглаженный idf
    X = tf * idf
    y = np.array([classes.index(r["label"]) for r in rows])

    rng = random.Random(seed)
    idx = list(range(n)); rng.shuffle(idx)
    n_val = max(1, int(n * val_split)) if n > 5 else 0
    val_idx, tr_idx = np.array(idx[:n_val]), np.array(idx[n_val:])
    return X, y, tr_idx, val_idx, classes

# ------------------------------------------------------------------- network

class MLP:
    """Двухслойный персептрон: ReLU -> Softmax, backprop вручную."""

    def __init__(self, n_in: int, n_hidden: int, n_out: int, seed: int):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, math.sqrt(2.0 / n_in), (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, math.sqrt(2.0 / n_hidden), (n_hidden, n_out))
        self.b2 = np.zeros(n_out)
        self.vW1 = self.vb1 = self.vW2 = self.vb2 = None  # momentum

    def forward(self, X):
        h = np.maximum(X @ self.W1 + self.b1, 0.0)
        z = h @ self.W2 + self.b2
        z -= z.max(axis=1, keepdims=True)                    # численно стабильный softmax
        e = np.exp(z)
        return h, e / e.sum(axis=1, keepdims=True)

    def step(self, X, y, lr, momentum, wd, batch):
        perm = np.random.permutation(len(X))
        total_loss, correct = 0.0, 0
        for s in range(0, len(X), batch):
            b = perm[s:s + batch]
            xb, yb = X[b], y[b]
            h, p = self.forward(xb)
            total_loss += -np.log(p[np.arange(len(b)), yb] + 1e-12).sum()
            correct += int((p.argmax(axis=1) == yb).sum())
            dz = p.copy(); dz[np.arange(len(b)), yb] -= 1.0  # softmax + CE градиент
            dz /= len(b)
            gW2, gb2 = h.T @ dz + wd * self.W2, dz.sum(axis=0)
            dh = dz @ self.W2.T; dh[h <= 0] = 0.0            # ReLU backward
            gW1, gb1 = xb.T @ dh + wd * self.W1, dh.sum(axis=0)
            if self.vW1 is None:                             # momentum-буферы
                self.vW1, self.vb1 = np.zeros_like(gW1), np.zeros_like(gb1)
                self.vW2, self.vb2 = np.zeros_like(gW2), np.zeros_like(gb2)
            self.vW1 = momentum * self.vW1 - lr * gW1
            self.vb1 = momentum * self.vb1 - lr * gb1
            self.vW2 = momentum * self.vW2 - lr * gW2
            self.vb2 = momentum * self.vb2 - lr * gb2
            self.W1 += self.vW1; self.b1 += self.vb1
            self.W2 += self.vW2; self.b2 += self.vb2
        return total_loss / len(X), correct / len(X)

# -------------------------------------------------------------------- runner

def _train(job: dict):
    """Реальное обучение MLP на CPU; метрики пишутся в job после каждой эпохи."""
    rows = [json.loads(l) for l in open(DATA / f"{job['dataset']}.jsonl", encoding="utf-8")]
    X, y, tr_idx, val_idx, classes = _featurize(
        rows, job.get("val_split", 0.2), job.get("seed", 42))
    vocab, idf = _vocab_idf(rows)
    net = MLP(X.shape[1], job.get("hidden", 64), len(classes), job.get("seed", 42))
    Xtr, ytr = X[tr_idx], y[tr_idx]
    job["classes"] = classes
    job["params"] = int(net.W1.size + net.W2.size + net.b1.size + net.b2.size)
    job["vocab_size"] = int(X.shape[1])
    job["train_rows"], job["val_rows"] = int(len(tr_idx)), int(len(val_idx))
    _replace_job(job)

    for ep in range(job["epochs"]):
        loss, acc = net.step(Xtr, ytr, job["lr"], momentum=0.9, wd=1e-4,
                             batch=job.get("batch", 8))
        job["loss"].append(round(loss, 4))
        job["accuracy"].append(round(acc, 4))
        if len(val_idx):
            _, pv = net.forward(X[val_idx])
            vloss = -np.log(pv[np.arange(len(val_idx)), y[val_idx]] + 1e-12).mean()
            job["val_loss"].append(round(float(vloss), 4))
            job["val_accuracy"].append(round(float((pv.argmax(1) == y[val_idx]).mean()), 4))
        job["progress"] = round((ep + 1) / job["epochs"] * 100)
        _replace_job(job)
    job["model"] = {"W1": net.W1.tolist(), "b1": net.b1.tolist(),
                    "W2": net.W2.tolist(), "b2": net.b2.tolist(),
                    "classes": classes, "vocab": list(vocab), "idf": idf.tolist()}
    job["status"] = "done"

def predict(text: str, job_id: str = None):
    """Инференс: вероятности классов по обученной модели последнего джоба."""
    jobs = jobs_state()
    candidates = [j for j in jobs if j.get("model")]
    if job_id:
        candidates = [j for j in candidates if j["id"] == job_id]
    if not candidates:
        return None
    job = candidates[0]
    m = job["model"]
    vocab = {w: i for i, w in enumerate(m["vocab"])}
    x = np.zeros(len(vocab))
    for w in text.lower().split():
        if w in vocab:
            x[vocab[w]] += 1.0
    mask = x > 0
    tf = np.zeros_like(x)
    tf[mask] = 1.0 + np.log(x[mask])
    X = (tf * np.array(m["idf"]))[None, :]
    W1, b1 = np.array(m["W1"]), np.array(m["b1"])
    h = np.maximum(X @ W1 + b1, 0.0)
    z = h @ np.array(m["W2"]) + np.array(m["b2"])
    z -= z.max()
    e = np.exp(z); p = (e / e.sum())[0]
    return {"job": job["id"],
            "probs": {c: round(float(pi), 4)
                      for c, pi in sorted(zip(m["classes"], p), key=lambda t: -t[1])}}

def _new_job(dataset: str, epochs: int, lr: float, **cfg) -> dict:
    return {"id": uuid.uuid4().hex[:8], "dataset": dataset, "epochs": epochs, "lr": lr,
            **cfg, "status": "running", "progress": 0,
            "loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [],
            "started": time.strftime("%H:%M:%S")}

def start_job(dataset: str, epochs: int = 8, lr: float = 0.3, **cfg) -> dict:
    job = _new_job(dataset, epochs, lr, **cfg)
    jobs = jobs_state(); jobs.insert(0, job); _save_jobs(jobs)
    threading.Thread(target=_train, args=(job,), daemon=True).start()
    return job

def run_job(dataset: str, epochs: int = 8, lr: float = 0.3, **cfg) -> dict:
    """Синхронный запуск — для демо и тестов."""
    job = _new_job(dataset, epochs, lr, **cfg)
    _train(job)
    _replace_job(job)
    return job

def metrics_for(job_id: str):
    return next((j for j in jobs_state() if j["id"] == job_id), None)
