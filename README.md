# LLM Finetune Studio — веб-студия дообучения локальных моделей

![LLM Finetune Studio](.assets/banner.svg)

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Дашборд, где бизнес загружает свои данные (тикеты поддержки, заявки, письма),
запускает дообучение модели под свои intents и смотрит метрики в реальном времени:
**loss падает, accuracy растёт — на каждом шаге**.

Демо-тренер идёт на CPU (softmax + SGD): реальное обучение с живой кривой лосса,
без GPU и без внешних API. Production-путь до QLoRA/HF описан и подготовлен в коде.

## Как это работает

```
CSV/JSONL датасет ──► Job Runner (очередь) ──► Тренер ──► метрики по эпохам
                        │                          │
                 POST /api/jobs            GET /api/jobs/{id}
                        │                          │
                        └────► дашборд (live loss chart, прогресс)
```

## Быстрый старт

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8090
# открыть http://localhost:8090
```

Дашборд: загрузить демо-датасет → задать эпохи/lr → запустить обучение →
наблюдать кривую лосса и accuracy в живом графике.

Без браузера:

```bash
python demo.py        # прогон обучения с печатью метрик по эпохам
pytest tests/ -q      # тесты движка
```

## Честность демо-режима

В README и коде явно указано: демо-тренер — линейный классификатор на CPU,
а не QLoRA. Это осознанный фолбэк: весь pipeline (джобы, метрики, дашборд)
уже работает, замена тренера на QLoRA — подмена одной функции.

## Стек

Python 3.11 · FastAPI · Pydantic · vanilla JS дашборд · pytest · GitHub Actions

## Что дальше

- [ ] QLoRA-тренер через peft (adapter-загрузка в Ollama/llama.cpp)
- [ ] Импорт датасета из CSV и Google Sheets
- [ ] Экспорт обученной модели в один файл + инференс-эндпоинт

MIT License
