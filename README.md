# AI Agent Prod: RAG-агент на LangGraph с подтверждением действий

Бэкенд агента, который отвечает по загруженным документам (RAG), помнит диалог и **спрашивает человека перед необратимыми действиями** (Human-in-the-Loop). Токены ответа приходят потоком (SSE), работа агента трассируется в Langfuse.

Статус: учебный проект с упором на архитектуру агента. Что сделано и что нет, написано честно в разделе «Ограничения».

## Что умеет

| Возможность | Как устроена |
|---|---|
| RAG по своим документам | `.txt`, `.md`, `.pdf`, `.docx` → чанки → эмбеддинги `multilingual-e5-base` (русский поддерживается) → pgvector |
| Память диалога | Граф LangGraph с чекпоинтером в PostgreSQL; диалог продолжается по `thread_id`, в том числе после перезапуска |
| Human-in-the-Loop | Инструменты с побочным эффектом (сейчас `delete_document`) ставят граф на паузу; человек выбирает `approve`, `edit` или `reject` с комментарием |
| Стриминг | Токены ответа через SSE (`stream_mode="messages"`) |
| Защита API | Ключ `X-API-Key`, лимит запросов в минуту на Redis, ограничения на загрузку файлов, CORS по списку |
| Наблюдаемость | Langfuse на весь граф (включая вызовы инструментов), сессия = `thread_id`; без ключей отключается сам |
| Предохранители | Лимит шагов агента (`recursion_limit`), ограничение длины истории, лимит размера файла |

## Архитектура

```
Клиент ──SSE──► FastAPI (/api) ──► LangGraph
                  │  auth, rate limit        │
                  │                    ┌─────┴─────┐
                  │                    ▼           ▼
                  │                  agent ◄──── tools
                  │               (LLM, промт)   ├─ rag_search          (чтение)
                  │                              ├─ list_documents      (чтение)
                  │                              └─ delete_document     (interrupt → решение человека)
                  ▼
        PostgreSQL: pgvector (чанки) + чекпоинты LangGraph      Redis: лимит запросов
```

Принципы:

- **Модель не принимает критических решений.** Удаление выполняет код, и только после явного решения человека.
- **Подтверждение стоит до действия.** После возобновления LangGraph выполняет узел заново с начала, поэтому в инструменте `interrupt()` вызывается до любых изменений (см. `app/agent/hitl.py`).
- **Любое непонятное решение считается отказом.**
- **Содержимое документов считается данными, а не инструкциями** (правило в системном промте против prompt injection; это снижает риск, но не гарантирует защиту).

## Быстрый старт

Нужны Docker и Docker Compose, ключ любого OpenAI-совместимого провайдера.

```bash
# 1. Окружение
cp .env.example .env
# Заполните в .env: API_KEY, OPENAI_API_KEY, LANGFUSE_NEXTAUTH_SECRET, LANGFUSE_SALT
# (случайные значения: openssl rand -hex 24)

# 2. Том с данными PostgreSQL (один раз; имя задано в docker-compose.yml)
docker volume create langraph_agent_pgdata

# 3. Запуск (базы langgraph_db и langfuse_db создаются автоматически)
docker compose up -d --build
```

Первый запуск дольше обычного: скачивается модель эмбеддингов (порядка 1 ГБ для e5-base), она кэшируется в томе `fastembed_cache`. Проверка:

```bash
curl http://localhost:8000/api/health        # {"status":"healthy"}
```

Swagger UI: http://localhost:8000/docs (кнопка **Authorize** принимает `X-API-Key`). Langfuse: http://localhost:3000 (создайте проект и впишите ключи в `.env`, затем `docker compose restart backend`).

### Если вы обновляетесь со старой версии

Размерность эмбеддингов изменилась (384 → 768), существующая таблица чанков несовместима. Пересоздайте её и загрузите документы заново:

```bash
docker exec -it agent_postgres psql -U postgres -d langgraph_db -c "DROP TABLE IF EXISTS document_chunks;"
docker compose restart backend
```

## Пример использования

```bash
export KEY=ваш-API_KEY

# 1. Загрузить документ
curl -X POST http://localhost:8000/api/documents/upload \
  -H "X-API-Key: $KEY" -F "file=@examples/omega.txt"

# 2. Задать вопрос (ответ приходит потоком SSE)
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"message": "Как называется секретный проект и какой у него бюджет?"}'
```

События SSE: `{"thread_id": "..."}`, затем `{"content": "..."}` (токены), при паузе `{"interrupt": {...}}`, в конце `[DONE]`. Чтобы продолжить диалог, передайте тот же `thread_id` в теле запроса.

### Подтверждение действия (HITL)

```bash
# Просим удалить документ: агент остановится и пришлёт запрос подтверждения
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"message": "Удали документ omega.txt"}'
# ... data: {"interrupt": {"action": "delete_document", "args": {"filename": "omega.txt"},
#                          "allowed_decisions": ["approve", "edit", "reject"], ...}, "thread_id": "abc"}

# Решение человека: согласиться
curl -N -X POST http://localhost:8000/api/chat/resume \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"thread_id": "abc", "decision": "approve"}'

# Или исправить аргументы и оставить комментарий
#   {"thread_id": "abc", "decision": "edit", "args": {"filename": "other.txt"}, "comment": "не тот файл"}
# Или отклонить: комментарий увидит агент и учтёт в ответе
#   {"thread_id": "abc", "decision": "reject", "comment": "нужен для отчёта"}
```

## Эндпоинты

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/api/health` | Проверка состояния (без ключа) |
| POST | `/api/chat/stream` | Сообщение агенту, ответ потоком SSE |
| POST | `/api/chat/resume` | Решение человека для приостановленного агента |
| POST | `/api/documents/upload` | Загрузка и индексация документа |
| GET | `/api/documents` | Список документов в базе знаний |

Все, кроме `health`, требуют заголовок `X-API-Key`. Чат дополнительно ограничен `RATE_LIMIT_PER_MINUTE`.

## Структура проекта

```
.
├── backend/
│   ├── app/
│   │   ├── agent/        graph.py (граф), nodes.py (модель, промт), tools.py (инструменты),
│   │   │                 hitl.py (подтверждения), state.py
│   │   ├── api/          routes.py (чат, загрузка, SSE)
│   │   ├── core/         config, db, security (ключ API), ratelimit (Redis), tracing (Langfuse)
│   │   ├── documents/    parser (имена файлов, извлечение текста), embeddings, service, models
│   │   └── main.py       приложение, lifespan: расширение vector, таблицы, чекпоинтер
│   ├── tests/            pytest: безопасность загрузки, HITL, стриминг, лимит запросов
│   ├── Dockerfile  requirements.txt  requirements-dev.txt  pytest.ini  pyproject.toml
├── examples/omega.txt    тестовый документ
├── docker-compose.yml  Caddyfile  .env.example
└── .github/workflows/ci.yml
```

## Разработка и тесты

```bash
cd backend
pip install -r requirements-dev.txt
pytest -q          # тесты не требуют БД, сети и ключей
ruff check . && ruff format --check .
```

Тесты используют управляемую «модель» вместо реальной LLM: проверяется поведение графа (пауза до действия, три вида решений, неясные решения как отказ, память по потокам), а не качество ответов.

## Ограничения (чего здесь нет)

- **Один пользователь.** Единый API-ключ, без учётных записей и разделения данных между людьми.
- **Нет оценки качества RAG** (evals): неизвестно, как часто поиск находит нужный фрагмент. Следующий шаг: набор вопросов с эталонами и метрики.
- **Без Alembic.** Таблицы создаются через `create_all`, смена схемы требует ручной миграции.
- **Простой поиск:** без индекса по векторам (HNSW), без гибридного поиска и rerank.
- **Лимит запросов общий**, а не на пользователя; при недоступности Redis запросы пропускаются.
- **HITL только для `delete_document`.** Каркас универсален (`request_approval`), но инструментов с побочными эффектами пока мало. Параллельные вызовы нескольких опасных инструментов за один шаг не проверялись; промт просит модель вызывать один инструмент за шаг.
- **Версии зафиксированы на langchain 0.3 / langgraph 0.6**, потому что Langfuse v2 несовместим с langchain 1.x (подробности в `requirements.txt`).
- Старые файлы в томе загрузок при повторной загрузке не удаляются (в базе документ заменяется).

## Лицензия

MIT
