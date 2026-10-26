# -*- coding: utf-8 -*-
"""LLM Finetune Studio API."""
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import trainer

app = FastAPI(title="LLM Finetune Studio", version="1.0.0")

class JobReq(BaseModel):
    dataset: str = "support"
    epochs: int = 8
    lr: float = 0.3

@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.get("/api/seed")
def seed():
    return {"dataset": trainer.seed_dataset(), "samples": len(trainer.SEED)}

@app.post("/api/jobs")
def create_job(req: JobReq):
    return trainer.start_job(req.dataset, req.epochs, req.lr)

@app.get("/api/jobs")
def list_jobs():
    return trainer.jobs_state()

@app.get("/api/jobs/{job_id}")
def job_metrics(job_id: str):
    job = trainer.metrics_for(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


@app.post("/api/eval/run")
def eval_run():
    """Отчёт качества: golden set, consistency данных, judge accuracy, регрессия."""
    from app import eval as ev
    return ev.eval_report()
