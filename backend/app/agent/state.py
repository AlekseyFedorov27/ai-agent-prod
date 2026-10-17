from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # add_messages дописывает сообщения в историю, а не заменяет её.
    messages: Annotated[list[AnyMessage], add_messages]
