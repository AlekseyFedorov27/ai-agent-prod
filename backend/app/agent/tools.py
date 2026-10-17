from langchain_core.tools import tool

from app.agent.hitl import request_approval
from app.core.db import AsyncSessionLocal
from app.documents.service import delete_document_chunks, search_documents
from app.documents.service import list_documents as list_documents_in_db


@tool
async def rag_search(query: str) -> str:
    """Ищет информацию в загруженных документах по смысловому сходству.
    Используй, когда нужно ответить на вопрос по базе знаний."""
    async with AsyncSessionLocal() as session:
        results = await search_documents(session, query)
    if not results:
        return "Информация в базе знаний не найдена."
    return "\n\n".join(f"Источник: {r['source']}\nТекст: {r['content']}" for r in results)


@tool
async def list_documents() -> str:
    """Показывает, какие документы сейчас есть в базе знаний."""
    async with AsyncSessionLocal() as session:
        documents = await list_documents_in_db(session)
    if not documents:
        return "База знаний пуста."
    return "\n".join(f"- {d['filename']} ({d['chunks']} фрагментов)" for d in documents)


@tool
async def delete_document(filename: str) -> str:
    """Удаляет документ из базы знаний по имени файла.
    Действие необратимо и выполняется только после подтверждения пользователя."""
    # Подтверждение ДО любого изменения (см. правило в app/agent/hitl.py)
    approval = request_approval(
        action="delete_document",
        args={"filename": filename},
        description=f"Удалить документ «{filename}» из базы знаний. Действие необратимо.",
    )
    if not approval.approved:
        return "Удаление отклонено пользователем." + approval.comment_suffix()

    target = str(approval.args["filename"])
    async with AsyncSessionLocal() as session:
        removed = await delete_document_chunks(session, target)
    if removed == 0:
        return f"Документ «{target}» не найден." + approval.comment_suffix()
    return f"Документ «{target}» удалён ({removed} фрагментов)." + approval.comment_suffix()


TOOLS = [rag_search, list_documents, delete_document]
