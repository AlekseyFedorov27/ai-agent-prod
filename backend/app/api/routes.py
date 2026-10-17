import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk
from langgraph.types import Command
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.db import get_db
from app.core.ratelimit import rate_limit
from app.core.security import require_api_key
from app.core.tracing import build_run_config
from app.documents.parser import sanitize_filename
from app.documents.service import list_documents, process_and_store_document

logger = logging.getLogger(__name__)

THREAD_ID_PATTERN = r"^[\w\-]+$"

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    # Без thread_id начинается новый диалог; с ним диалог продолжается.
    thread_id: str | None = Field(default=None, max_length=64, pattern=THREAD_ID_PATTERN)


class ResumeRequest(BaseModel):
    thread_id: str = Field(max_length=64, pattern=THREAD_ID_PATTERN)
    decision: Literal["approve", "edit", "reject"]
    # Для decision="edit": новые значения аргументов инструмента
    args: dict[str, Any] | None = None
    comment: str | None = Field(default=None, max_length=1000)


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _pending_interrupts(state: Any) -> list[Any]:
    return [i.value for task in state.tasks for i in task.interrupts]


async def _stream_graph(graph: Any, graph_input: Any, thread_id: str) -> AsyncIterator[str]:
    """Запускает граф и отдаёт события SSE:
    {"thread_id"}, {"content": токены ответа}, {"interrupt": запрос подтверждения}, [DONE]."""
    config = build_run_config(thread_id)
    yield _sse({"thread_id": thread_id})
    try:
        async for chunk, meta in graph.astream(graph_input, config, stream_mode="messages"):
            if meta.get("langgraph_node") != "agent":
                continue
            if isinstance(chunk, AIMessageChunk) and isinstance(chunk.content, str):
                if chunk.content:
                    yield _sse({"content": chunk.content})

        # Граф мог остановиться на подтверждении человека
        state = await graph.aget_state(config)
        interrupts = _pending_interrupts(state)
        if interrupts:
            yield _sse({"interrupt": interrupts[0], "thread_id": thread_id})
    except Exception:
        logger.exception("Ошибка при работе агента (thread_id=%s)", thread_id)
        # Подробности только в логах, клиенту общее сообщение
        yield _sse({"error": "Внутренняя ошибка агента, попробуйте ещё раз"})
    yield "data: [DONE]\n\n"


@router.post("/chat/stream", dependencies=[Depends(require_api_key), Depends(rate_limit)])
async def chat_stream(body: ChatRequest, request: Request):
    thread_id = body.thread_id or uuid.uuid4().hex
    graph_input = {"messages": [{"role": "user", "content": body.message}]}
    return StreamingResponse(
        _stream_graph(request.app.state.graph, graph_input, thread_id),
        media_type="text/event-stream",
    )


@router.post("/chat/resume", dependencies=[Depends(require_api_key), Depends(rate_limit)])
async def chat_resume(body: ResumeRequest, request: Request):
    """Передаёт решение человека приостановленному агенту (approve / edit / reject)."""
    graph = request.app.state.graph
    state = await graph.aget_state(build_run_config(body.thread_id))
    if not _pending_interrupts(state):
        raise HTTPException(
            status_code=409, detail="Для этого диалога нет ожидающего подтверждения"
        )

    resume_value = {"decision": body.decision, "args": body.args, "comment": body.comment}
    return StreamingResponse(
        _stream_graph(graph, Command(resume=resume_value), body.thread_id),
        media_type="text/event-stream",
    )


@router.post("/documents/upload", dependencies=[Depends(require_api_key)])
async def upload_document(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    """Загрузка и индексация документа."""
    try:
        safe_name = sanitize_filename(file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Файл больше {settings.max_upload_mb} МБ")

    # Имя на диске уникально и не зависит от клиента; исходное имя храним в БД
    destination = Path(settings.upload_dir) / f"{uuid.uuid4().hex}_{safe_name}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(destination.write_bytes, data)

    try:
        details = await process_and_store_document(destination, safe_name, db)
    except ValueError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        destination.unlink(missing_ok=True)
        logger.exception("Не удалось проиндексировать документ %s", safe_name)
        raise HTTPException(status_code=500, detail="Не удалось проиндексировать документ") from exc

    return {"status": "success", "details": details}


@router.get("/documents", dependencies=[Depends(require_api_key)])
async def documents(db: AsyncSession = Depends(get_db)):
    return {"documents": await list_documents(db)}


@router.get("/health")
async def health_check():
    return {"status": "healthy"}
