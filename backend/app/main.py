import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import text

from app.agent.graph import build_graph
from app.api.routes import router
from app.core.config import settings
from app.core.db import Base, engine

# Импорт модели нужен, чтобы SQLAlchemy зарегистрировал её в Base.metadata
from app.documents import models  # noqa: F401
from app.documents.embeddings import warm_up

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Расширение pgvector и таблицы приложения
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

    # Модель эмбеддингов грузим заранее, а не на первом запросе
    await warm_up()

    # Чекпоинтер хранит состояние диалогов в PostgreSQL (таблицы создаёт setup)
    async with AsyncPostgresSaver.from_conn_string(settings.checkpoint_uri) as checkpointer:
        await checkpointer.setup()
        app.state.graph = build_graph(checkpointer)
        yield

    await engine.dispose()


app = FastAPI(title="AI Agent Prod Backend", version="1.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

app.include_router(router, prefix="/api")
