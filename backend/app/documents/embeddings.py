"""Эмбеддинги для RAG.

Модель intfloat/multilingual-e5-base (768 измерений, поддерживает русский).
Требует префиксы: «passage: » для документов и «query: » для запросов.

fastembed считает на CPU синхронно, поэтому все вызовы уходят в поток
(asyncio.to_thread), иначе сервер замирает на время расчёта.
"""

import asyncio
from functools import lru_cache

from app.core.config import settings


@lru_cache(maxsize=1)
def get_embedding_model():
    # Импорт внутри функции: модель и пакет нужны только при реальном расчёте.
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    name = settings.embedding_model
    supported = {m["model"] for m in TextEmbedding.list_supported_models()}
    if name not in supported:
        # e5-base нет в стандартном реестре fastembed, регистрируем вручную.
        # Параметры (MEAN pooling, normalization) официальные для e5-моделей.
        TextEmbedding.add_custom_model(
            model=name,
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf=name),
            dim=settings.embedding_dim,
        )
    return TextEmbedding(model_name=name)


def _embed_sync(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    return [vector.tolist() for vector in model.embed(texts)]


async def embed_passages(chunks: list[str]) -> list[list[float]]:
    return await asyncio.to_thread(_embed_sync, [f"passage: {c}" for c in chunks])


async def embed_query(query: str) -> list[float]:
    vectors = await asyncio.to_thread(_embed_sync, [f"query: {query}"])
    return vectors[0]


async def warm_up() -> None:
    """Загружает модель при старте, чтобы первый запрос не ждал скачивания."""
    await asyncio.to_thread(get_embedding_model)
