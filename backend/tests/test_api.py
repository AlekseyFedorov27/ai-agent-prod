import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agent import nodes, tools
from app.agent.graph import build_graph
from app.api import routes
from app.core.config import settings
from app.core.db import get_db
from app.core.ratelimit import rate_limit
from tests.fakes import FakeSession, ScriptedChatModel, tool_call

HEADERS = {"X-API-Key": settings.api_key}


def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(routes.router, prefix="/api")

    async def fake_db():
        yield None

    app.dependency_overrides[get_db] = fake_db
    app.dependency_overrides[rate_limit] = lambda: None
    return app


def read_sse(response) -> list[dict | str]:
    events: list[dict | str] = []
    for line in response.iter_lines():
        if not line.startswith("data: "):
            continue
        body = line[len("data: ") :]
        events.append(body if body == "[DONE]" else json.loads(body))
    return events


# ---------- загрузка документов ----------


@pytest.fixture
def upload_client(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    monkeypatch.setattr(settings, "upload_dir", str(upload_dir))
    stored: list[tuple] = []

    async def fake_process(path, filename, session):
        stored.append((path, filename))
        return {"filename": filename}

    monkeypatch.setattr(routes, "process_and_store_document", fake_process)
    client = TestClient(make_app())
    client.stored = stored
    client.upload_dir = upload_dir
    return client


def upload(client, filename, content=b"hello", headers=HEADERS):
    return client.post(
        "/api/documents/upload", files={"file": (filename, content, "text/plain")}, headers=headers
    )


def test_upload_requires_api_key(upload_client):
    assert upload(upload_client, "a.txt", headers={}).status_code == 401
    assert upload(upload_client, "a.txt", headers={"X-API-Key": "wrong"}).status_code == 401


@pytest.mark.parametrize("evil_name", ["../../evil.txt", "..\\..\\evil.txt", "/etc/evil.txt"])
def test_upload_cannot_escape_upload_dir(upload_client, tmp_path, evil_name):
    response = upload(upload_client, evil_name)

    assert response.status_code == 200
    path, filename = upload_client.stored[-1]
    assert filename == "evil.txt"
    assert path.resolve().parent == upload_client.upload_dir.resolve()
    assert not (tmp_path / "evil.txt").exists()
    assert not (tmp_path.parent / "evil.txt").exists()


def test_upload_rejects_unsupported_format(upload_client):
    assert upload(upload_client, "virus.exe").status_code == 400
    assert upload_client.stored == []


def test_upload_rejects_too_large_file(upload_client, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 1)
    response = upload(upload_client, "big.txt", content=b"x" * (1024 * 1024 + 1))
    assert response.status_code == 413
    assert upload_client.stored == []


def test_upload_removes_file_when_indexing_fails(upload_client, monkeypatch):
    async def failing_process(path, filename, session):
        raise ValueError("Документ пуст")

    monkeypatch.setattr(routes, "process_and_store_document", failing_process)
    response = upload(upload_client, "empty.txt")

    assert response.status_code == 422
    assert list(upload_client.upload_dir.glob("*")) == []


def test_health_is_public():
    client = TestClient(make_app())
    assert client.get("/api/health").json() == {"status": "healthy"}


# ---------- чат: стриминг и подтверждения ----------


@pytest.fixture
def chat_client(monkeypatch):
    app = make_app()
    app.state.graph = build_graph(MemorySaver())
    monkeypatch.setattr(tools, "AsyncSessionLocal", lambda: FakeSession())
    return TestClient(app)


def set_model(monkeypatch, replies):
    model = ScriptedChatModel(replies=replies)
    monkeypatch.setattr(nodes, "get_llm", lambda: model)


def test_chat_requires_api_key(chat_client):
    response = chat_client.post("/api/chat/stream", json={"message": "привет"})
    assert response.status_code == 401


def test_chat_streams_tokens_and_finishes(chat_client, monkeypatch):
    set_model(monkeypatch, [AIMessage(content="Привет мир")])
    with chat_client.stream(
        "POST", "/api/chat/stream", json={"message": "привет"}, headers=HEADERS
    ) as response:
        events = read_sse(response)

    assert "thread_id" in events[0]
    text = "".join(e["content"] for e in events if isinstance(e, dict) and "content" in e)
    assert text.strip() == "Привет мир"
    assert events[-1] == "[DONE]"


def test_chat_interrupt_then_resume_with_approval(chat_client, monkeypatch):
    deleted: list[str] = []

    async def fake_delete(session, filename):
        deleted.append(filename)
        return 2

    monkeypatch.setattr(tools, "delete_document_chunks", fake_delete)
    set_model(
        monkeypatch,
        [tool_call("delete_document", {"filename": "a.txt"}), AIMessage(content="Удалено")],
    )

    with chat_client.stream(
        "POST", "/api/chat/stream", json={"message": "удали a.txt"}, headers=HEADERS
    ) as response:
        events = read_sse(response)

    thread_id = events[0]["thread_id"]
    interrupt_event = next(e for e in events if isinstance(e, dict) and "interrupt" in e)
    assert interrupt_event["interrupt"]["action"] == "delete_document"
    assert deleted == []

    with chat_client.stream(
        "POST",
        "/api/chat/resume",
        json={"thread_id": thread_id, "decision": "approve"},
        headers=HEADERS,
    ) as response:
        resumed = read_sse(response)

    assert deleted == ["a.txt"]
    text = "".join(e["content"] for e in resumed if isinstance(e, dict) and "content" in e)
    assert "Удалено" in text
    assert resumed[-1] == "[DONE]"


def test_resume_without_pending_interrupt_returns_409(chat_client):
    response = chat_client.post(
        "/api/chat/resume",
        json={"thread_id": "no-such-thread", "decision": "approve"},
        headers=HEADERS,
    )
    assert response.status_code == 409


def test_chat_validates_input(chat_client):
    url = "/api/chat/stream"
    assert chat_client.post(url, json={"message": ""}, headers=HEADERS).status_code == 422
    bad_thread = {"message": "привет", "thread_id": "../x"}
    assert chat_client.post(url, json=bad_thread, headers=HEADERS).status_code == 422
