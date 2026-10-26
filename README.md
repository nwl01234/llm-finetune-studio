# LLM Finetune Studio — платформа дообучения локальных моделей

![LLM Finetune Studio](.assets/banner.svg)

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![CI](https://img.shields.io/github/actions/workflow/status/nwl01234/llm-finetune-studio/ci.yml?label=CI&logo=github)
![License](https://img.shields.io/badge/License-MIT-green)

## Что внутри

| Модуль | Что делает |
|---|---|
| `app/main.py` | FastAPI: запуска джобов, чтение метрик |
| `app/trainer.py` | движок: датасет, обучение, очередь джобов в потоках, метрики по эпохам |
| `app/eval.py` | оценка качества: golden-датасет, consistency, LLM-as-a-judge (rule-based fallback), регрессия прогонов |
| `static/index.html` | дашборд: тёмная тема, vanilla JS, живой график лосса |
| `demo.py` | обучение без браузера, печать метрик в stdout |
| `tests/test_trainer.py` | 3 теста: seed-датасет, рост accuracy, поиск джобы по id |

## Датасет и обучение

Демо-датасет `data/support.jsonl` — 96 примеров, 4 класса интентов:
`billing`, `tech`, `refund`, `sales` (обращения в поддержку). Структура записи:
`{"text": "оплата не прошла", "label": "billing"}`.

Признаки: tokenizer — split по словам и lower-case set(), словарь (vocab) собирается
один раз по всей выборке. Дальше softmax-классификатор:

```
W[c][w] + b[c]              — вес класса c на слове w
score(c) = b[c] + Σ W[c][vocab[w]]   по всем словам запроса
p(y|x) = softmax(score)     — по всем классам
loss += -log(p[y])          — кросс-энтропия на примере
SGD: lr=0.3, 8 эпох (default), порядок примеров перемешивается с seed=epoch
```

Метрики по каждой эпохе пишутся в job dicts: `loss[]`, `accuracy[]`, `progress%` —
добавляются в JSON (`data/jobs.json`): дашборд читает их и рисует кривую без поллинга.

Прогон `python demo.py`: **loss 0.5754 → 0.0194, accuracy 0 → 100% за 8 эпох**
(на простом синтетичном датасете; честные метрики, не мок).

Тренер — сменный модуль: интерфейс `_train(job)` изолирован, подмена на
QLoRA (torch + peft + локальная Qwen2.5/Llama) не меняет API и дашборд.

## Job Runner

Джоба — daemon-поток; состояние держится в `data/jobs.json` и восстанавливается
после рестарта. Статус (`running`, `progress 0-100%`), метрики, время старта —
всё читается через API. Опрашивается раз в 600 мс фронтом для живой кривой.

## API

```
GET  /api/seed            — создать демо-датасет (96 примеров)
POST /api/jobs            — {"dataset":"support", "epochs":8, "lr":0.3} → id джобы
GET  /api/jobs            — список джобов со статусом/прогрессом
GET  /api/jobs/{id}       — метрики поля по id
```

## Запуск

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8090   # дашборд: http://localhost:8090
python demo.py                     # то же обучение без браузера с выводом метрик
pytest tests/ -q                   # 3 теста: seed, метрики растут, поиск по id
```

## Что дальше

- [ ] Переход на QLoRA-тренер (peft) с сохранением адаптера в один файл + экспорт в Ollama
- [ ] Импорт датасета из CSV/Google Sheets
- [ ] История обучения + сравнение прогонов (run-to-run diff)
- [ ] RBAC: дата-сайентист запускает, заказчик только читает прогресс

MIT License
