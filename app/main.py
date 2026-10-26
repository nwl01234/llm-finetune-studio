# -*- coding: utf-8 -*-
"""LLM Finetune Studio API."""
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import trainer

app = FastAPI(title="LLM Finetune Studio", version="2.0.0")

class JobReq(BaseModel):
    dataset: str = "support"
    epochs: int = 12
    lr: float = 0.08
    hidden: int = 64
    seed: int = 42
    val_split: float = 0.2
    batch: int = 8

@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.get("/api/seed")
def seed():
    return {"dataset": trainer.seed_dataset(), "samples": len(trainer.SEED)}

@app.get("/api/dataset/{name}")
def dataset_stats(name: str):
    """Статистика датасета: размер, распределение классов, превью."""
    p = trainer.DATA / f"{name}.jsonl"
    if not p.exists():
        raise HTTPException(404, "dataset not found")
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    dist = {}
    for r in rows:
        dist[r["label"]] = dist.get(r["label"], 0) + 1
    return {"name": name, "rows": len(rows), "labels": dist,
            "preview": rows[:8]}

@app.post("/api/jobs")
def create_job(req: JobReq):
    return trainer.start_job(
        req.dataset, req.epochs, req.lr, hidden=req.hidden, seed=req.seed,
        val_split=req.val_split, batch=req.batch)

@app.get("/api/jobs")
def list_jobs():
    return trainer.jobs_state()

@app.get("/api/jobs/{job_id}")
def job_metrics(job_id: str):
    job = trainer.metrics_for(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


class PredictReq(BaseModel):
    text: str
    job_id: str | None = None

@app.post("/api/predict")
def predict(req: PredictReq):
    """Живой инференс: вероятности классов по обученной модели."""
    if not req.text.strip():
        raise HTTPException(422, "text is empty")
    res = trainer.predict(req.text, req.job_id)
    if res is None:
        raise HTTPException(404, "no trained model yet — запустите обучение")
    return res


@app.post("/api/eval/run")
def eval_run():
    """Отчёт качества: golden set, consistency данных, judge accuracy, регрессия."""
    from app import eval as ev
    return ev.eval_report()
