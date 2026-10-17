import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.agent import nodes, tools
from app.agent.graph import build_graph
from tests.fakes import FakeSession, ScriptedChatModel, tool_call

CONFIG = {"configurable": {"thread_id": "t1"}}
USER_MSG = {"messages": [{"role": "user", "content": "удали a.txt"}]}


@pytest.fixture
def deleted(monkeypatch):
    """Список имён, которые реально «удалили». Пусто, пока человек не подтвердил."""
    calls: list[str] = []

    async def fake_delete(session, filename):
        calls.append(filename)
        return 3

    monkeypatch.setattr(tools, "delete_document_chunks", fake_delete)
    monkeypatch.setattr(tools, "AsyncSessionLocal", lambda: FakeSession())
    return calls


def make_graph(monkeypatch, replies):
    model = ScriptedChatModel(replies=replies)
    monkeypatch.setattr(nodes, "get_llm", lambda: model)
    return build_graph(MemorySaver())


async def pending(graph):
    state = await graph.aget_state(CONFIG)
    return [i.value for task in state.tasks for i in task.interrupts]


async def test_graph_pauses_before_deleting(monkeypatch, deleted):
    graph = make_graph(monkeypatch, [tool_call("delete_document", {"filename": "a.txt"})])
    await graph.ainvoke(USER_MSG, CONFIG)

    requests = await pending(graph)
    assert requests[0]["action"] == "delete_document"
    assert requests[0]["args"] == {"filename": "a.txt"}
    assert requests[0]["allowed_decisions"] == ["approve", "edit", "reject"]
    assert deleted == [], "до подтверждения ничего удаляться не должно"


async def test_approve_executes_the_action(monkeypatch, deleted):
    graph = make_graph(
        monkeypatch,
        [tool_call("delete_document", {"filename": "a.txt"}), AIMessage(content="Готово")],
    )
    await graph.ainvoke(USER_MSG, CONFIG)
    result = await graph.ainvoke(Command(resume={"decision": "approve"}), CONFIG)

    assert deleted == ["a.txt"]
    assert result["messages"][-1].content == "Готово"
    assert await pending(graph) == []


async def test_edit_changes_arguments_before_execution(monkeypatch, deleted):
    graph = make_graph(
        monkeypatch,
        [tool_call("delete_document", {"filename": "a.txt"}), AIMessage(content="Ок")],
    )
    await graph.ainvoke(USER_MSG, CONFIG)
    resume = {"decision": "edit", "args": {"filename": "b.txt"}, "comment": "не тот файл"}
    result = await graph.ainvoke(Command(resume=resume), CONFIG)

    assert deleted == ["b.txt"]
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "не тот файл" in tool_msg.content


async def test_reject_skips_action_and_passes_comment_to_agent(monkeypatch, deleted):
    graph = make_graph(
        monkeypatch,
        [tool_call("delete_document", {"filename": "a.txt"}), AIMessage(content="Понял")],
    )
    await graph.ainvoke(USER_MSG, CONFIG)
    resume = {"decision": "reject", "comment": "нужен для отчёта"}
    result = await graph.ainvoke(Command(resume=resume), CONFIG)

    assert deleted == []
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "отклонено" in tool_msg.content
    assert "нужен для отчёта" in tool_msg.content


@pytest.mark.parametrize(
    "resume",
    [
        {"decision": "что-то странное"},
        {"decision": "edit"},  # правка без новых аргументов
        {"decision": "edit", "args": {}},
        "approve",  # не словарь
    ],
)
async def test_unclear_decisions_are_treated_as_reject(monkeypatch, deleted, resume):
    graph = make_graph(
        monkeypatch,
        [tool_call("delete_document", {"filename": "a.txt"}), AIMessage(content="Ок")],
    )
    await graph.ainvoke(USER_MSG, CONFIG)
    await graph.ainvoke(Command(resume=resume), CONFIG)
    assert deleted == []


async def test_read_only_tool_does_not_ask_for_approval(monkeypatch):
    async def fake_search(session, query, top_k=None):
        return [{"content": "Проект называется Омега", "source": "omega.txt"}]

    monkeypatch.setattr(tools, "search_documents", fake_search)
    monkeypatch.setattr(tools, "AsyncSessionLocal", lambda: FakeSession())
    graph = make_graph(
        monkeypatch,
        [tool_call("rag_search", {"query": "название проекта"}), AIMessage(content="Омега")],
    )
    result = await graph.ainvoke(USER_MSG, CONFIG)

    assert await pending(graph) == []
    assert result["messages"][-1].content == "Омега"
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "omega.txt" in tool_msg.content


async def test_conversation_memory_is_kept_per_thread(monkeypatch):
    graph = make_graph(monkeypatch, [AIMessage(content="первый"), AIMessage(content="второй")])
    await graph.ainvoke({"messages": [{"role": "user", "content": "раз"}]}, CONFIG)
    result = await graph.ainvoke({"messages": [{"role": "user", "content": "два"}]}, CONFIG)

    contents = [m.content for m in result["messages"]]
    assert contents == ["раз", "первый", "два", "второй"]
