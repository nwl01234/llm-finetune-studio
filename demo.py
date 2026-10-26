# -*- coding: utf-8 -*-
"""Демо обучения без сервера: python demo.py"""
from app import trainer

if __name__ == "__main__":
    print("датасет:", trainer.seed_dataset(), "-", len(trainer.SEED), "примеров")
    job = trainer.run_job("support", epochs=8, lr=0.3)
    for i, (l, a) in enumerate(zip(job["loss"], job["accuracy"]), 1):
        print(f"epoch {i:>2}/8  loss={l:.4f}  accuracy={a:.2%}")
    print(f"итог: loss {job['loss'][0]} -> {job['loss'][-1]}, accuracy {job['accuracy'][-1]:.0%}")
