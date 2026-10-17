"""Тестовые двойники: управляемая «модель» и поддельная сессия БД."""

import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


class ScriptedChatModel(BaseChatModel):
    """Отдаёт заранее заготовленные ответы по очереди и умеет стримить их."""

    replies: list[AIMessage]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs: Any):
        reply = self.replies.pop(0)
        if reply.tool_calls:
            chunks = [
                {
                    "name": call["name"],
                    "args": json.dumps(call["args"]),
                    "id": call["id"],
                    "index": index,
                }
                for index, call in enumerate(reply.tool_calls)
            ]
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=chunks))
            return
        for word in str(reply.content).split(" "):
            yield ChatGenerationChunk(message=AIMessageChunk(content=word + " "))


def tool_call(name: str, args: dict, call_id: str = "call-1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


class FakeSession:
    """Заменяет AsyncSessionLocal(): годится для `async with ... as session`."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False
