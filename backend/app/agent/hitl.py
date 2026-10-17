"""Универсальное подтверждение действий человеком (Human-in-the-Loop).

Любой инструмент с побочным эффектом вызывает request_approval() ДО того,
как что-то изменит. Граф приостанавливается; клиент показывает запрос и
присылает решение:

    {"decision": "approve"}
    {"decision": "edit", "args": {...}, "comment": "..."}
    {"decision": "reject", "comment": "..."}

ВАЖНО: после возобновления LangGraph выполняет узел (инструмент) с самого
начала. Поэтому всё, что стоит ДО request_approval(), должно быть без
побочных эффектов, а само действие выполняется строго ПОСЛЕ неё.
"""

from dataclasses import dataclass
from typing import Any

from langgraph.types import interrupt

APPROVE = "approve"
EDIT = "edit"
REJECT = "reject"
ALLOWED_DECISIONS = [APPROVE, EDIT, REJECT]


@dataclass
class ApprovalResult:
    approved: bool
    args: dict[str, Any]
    comment: str = ""

    def comment_suffix(self) -> str:
        return f" Комментарий пользователя: {self.comment}" if self.comment else ""


def request_approval(action: str, args: dict[str, Any], description: str) -> ApprovalResult:
    """Приостанавливает граф и возвращает решение человека.

    Любое непонятное решение трактуется как отказ: по умолчанию безопасно.
    """
    decision = interrupt(
        {
            "action": action,
            "args": args,
            "description": description,
            "allowed_decisions": ALLOWED_DECISIONS,
        }
    )
    if not isinstance(decision, dict):
        return ApprovalResult(False, args)

    kind = decision.get("decision")
    comment = str(decision.get("comment") or "")

    if kind == APPROVE:
        return ApprovalResult(True, args, comment)

    if kind == EDIT:
        edited = decision.get("args")
        if isinstance(edited, dict) and edited:
            return ApprovalResult(True, {**args, **edited}, comment)
        # «Правка» без новых аргументов бессмысленна, считаем отказом
        return ApprovalResult(False, args, comment or "Правка без новых аргументов")

    return ApprovalResult(False, args, comment)
