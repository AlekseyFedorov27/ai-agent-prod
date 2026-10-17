from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.nodes import call_model
from app.agent.state import AgentState
from app.agent.tools import TOOLS


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Собирает граф агента.

    agent -> (есть вызовы инструментов? tools : конец) -> agent ...

    Подтверждения человека (interrupt) происходят внутри инструментов с
    побочными эффектами, поэтому граф остаётся простым. Чекпоинтер хранит
    состояние диалога по thread_id и нужен для пауз и возобновления.
    """
    builder = StateGraph(AgentState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=checkpointer)
