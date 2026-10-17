import asyncio
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.documents.embeddings import embed_passages, embed_query
from app.documents.models import DocumentChunk
from app.documents.parser import extract_text


def split_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)


async def process_and_store_document(
    file_path: Path, filename: str, session: AsyncSession
) -> dict[str, Any]:
    """Парсит документ, режет на чанки, считает эмбеддинги и сохраняет в pgvector.

    Повторная загрузка файла с тем же именем заменяет старые чанки, а не
    дублирует их.
    """
    # 1. Извлечение текста (синхронная работа с диском и парсерами)
    text = await asyncio.to_thread(extract_text, file_path, filename)
    if not text.strip():
        raise ValueError(f"Документ '{filename}' не содержит текста.")

    # 2. Чанкинг
    chunks = split_text(text)
    if not chunks:
        raise ValueError(f"Не удалось разбить документ '{filename}' на чанки.")

    # 3. Эмбеддинги (в потоке, чтобы не блокировать цикл событий)
    embeddings = await embed_passages(chunks)

    # 4. Замена старых чанков и сохранение новых в одной транзакции
    await session.execute(delete(DocumentChunk).where(DocumentChunk.filename == filename))
    session.add_all(
        DocumentChunk(filename=filename, content=chunk, embedding=vector)
        for chunk, vector in zip(chunks, embeddings, strict=True)
    )
    await session.commit()

    return {
        "filename": filename,
        "chunks_count": len(chunks),
        "embedding_model": settings.embedding_model,
        "embedding_dim": len(embeddings[0]),
    }


async def search_documents(
    session: AsyncSession, query: str, top_k: int | None = None
) -> list[dict[str, Any]]:
    """Ищет top_k ближайших чанков по косинусному расстоянию."""
    if not query or not query.strip():
        return []

    query_embedding = await embed_query(query.strip())
    stmt = (
        select(DocumentChunk)
        .order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
        .limit(top_k or settings.rag_top_k)
    )
    result = await session.execute(stmt)
    return [{"content": row.content, "source": row.filename} for row in result.scalars()]


async def list_documents(session: AsyncSession) -> list[dict[str, Any]]:
    stmt = (
        select(DocumentChunk.filename, func.count(DocumentChunk.id))
        .group_by(DocumentChunk.filename)
        .order_by(DocumentChunk.filename)
    )
    result = await session.execute(stmt)
    return [{"filename": name, "chunks": count} for name, count in result.all()]


async def delete_document_chunks(session: AsyncSession, filename: str) -> int:
    """Удаляет все чанки документа и возвращает их число."""
    result = await session.execute(delete(DocumentChunk).where(DocumentChunk.filename == filename))
    await session.commit()
    return result.rowcount or 0
