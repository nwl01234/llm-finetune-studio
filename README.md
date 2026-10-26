# LLM Finetune Studio — веб-студия дообучения локальных LLM под задачи бизнеса

![LLM Finetune Studio](.assets/banner.svg)

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![CI](https://img.shields.io/github/actions/workflow/status/nwl01234/llm-finetune-studio/ci.yml?label=CI&logo=github)
![License](https://img.shields.io/badge/License-MIT-green)

## Почему это болит у работодателей

Вакансии «AI Engineer», «AI Product Builder / Vibe-coder» (180-230k ₽ на hh.ru)
повторяют одну архитектурную задачу: **каждому клиенту нужен AI-ассистент,
говорящий на языке его бизнеса**, — тикеты поддержки, заявки, переписка.
прится промптами: бизнес-специфичные интенты ("двойное списание",
"акт сверки") требуют дообучения на данных клиента.

LLM Finetune Studio — готовая инфраструктура этого процесса:
загрузил свои данные → запустил джобу → следишь за живыми loss/accuracy →
получил модель.

## Как работает

```
Данные бизнеса (JSONL: текст + label)
         │
         ▼
POST /api/jobs  ──►  Job Runner (очередь в потоках, статус/прогресс 0-100%)
         │                    │
         │                    ▼
         │            Тренер (два режима):
         │            • demo:  softmax + SGD на CPU (реальное обучение, честные метрики)
         │            • prod:  QLoRA через peft (подключается в одной функции)
         │                    │
         ▼                    ▼
GET /api/jobs/{id}  ◄── метрики по эпохам (loss[], accuracy[], progress)
         │
         ▼
Дашборд: тёмная тема, живая кривая лосса, прогресс, история джобов
```

## Дашборд

Открыть `http://localhost:8090` → «Загрузить демо-датасет» (96 тикетов поддержки:
billing / tech / refund / sales) → «Запустить обучение» → живой график лосса.

## API

```
GET  /api/seed              → создать демо-датасет
POST /api/jobs              → {dataset, epochs, lr}   → старт джобы
GET  /api/jobs              → список джобов с прогрессом
GET  /api/jobs/{id}         → loss[], accuracy[], progress (для опроса фронтом)
```

## Быстрый старт

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8090        # веб-дашборд
python demo.py                          # то же обучение без браузера, с печатью метрик
pytest tests/ -q                        # 3 теста: seed, метрики растут, по id найдётся
```

## Честность демо-режима

Дашборд и job-инфраструктура полностью рабочие. Тренер по умолчанию —
линейный классификатор на CPU: это реальное обучение (loss 0.575 → 0.019
за 8 эпох, accuracy до 100% на synthetic-датасете), а не мок. Путь до
QLoRA описан и изолирован в одной функции `_train()` — заменяется без
изменений API и дашборда.

## Стек

Python 3.11 · FastAPI · Pydantic · vanilla JS (без бандлер) · threading Job Runner · pytest · GitHub Actions CI

## Что дальше

- [ ] QLoRA-тренер (peft + Qwen2.5/Llama) с сохранением адаптера в один файл
- [ ] Импорт CSV и Google Sheets как источников данных
- [ ] Роли сотрудников: дата-сайентист запускает, заказчик смотрит прогресс
- [ ] Экспорт модели в Ollama (дальше инференс на месте заказчика)

MIT License
