# -*- coding: utf-8 -*-
"""Демо обучения без сервера: python demo.py"""
from app import trainer

if __name__ == "__main__":
    n = len(trainer.SEED)
    print(f"датасет: support — {n} примеров, MLP numpy (backprop, mini-batch SGD)")
    job = trainer.run_job("support", epochs=15, lr=0.05, hidden=64,
                          val_split=0.2, batch=8, seed=42)
    print(f"параметров: {job['params']}, словарь: {job['vocab_size']}, "
          f"train/val: {job['train_rows']}/{job['val_rows']}")
    for i, l in enumerate(job["loss"], 1):
        va = job["val_accuracy"][i-1]
        print(f"epoch {i:>2}/15  loss={l:.4f}  val_loss={job['val_loss'][i-1]:.4f}"
              f"  val_acc={va:.1%}")
    print(f"итог: loss {job['loss'][0]:.4f} -> {job['loss'][-1]:.4f}, "
          f"val accuracy {job['val_accuracy'][-1]:.0%}")
