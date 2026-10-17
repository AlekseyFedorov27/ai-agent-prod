import logging
from functools import lru_cache
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _langfuse_handler() -> Any | None:
    """Один обработчик Langfuse на процесс. Если ключей нет или пакет
    несовместим, трассировка тихо отключается, а чат продолжает работать."""
    if not settings.langfuse_enabled:
        return None
    try:
        from langfuse.callback import CallbackHandler

        return CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except Exception:
        logger.warning("Langfuse недоступен, трассировка отключена", exc_info=True)
        return None


def build_run_config(thread_id: str) -> dict[str, Any]:
    """Конфиг запуска графа: поток диалога, трассировка, предохранитель от циклов.

    Обработчик передаётся всему графу (а не только вызову модели), поэтому
    в трассе видны и вызовы инструментов. Сессия Langfuse = thread_id.
    """
    handler = _langfuse_handler()
    return {
        "configurable": {"thread_id": thread_id},
        "callbacks": [handler] if handler else [],
        "recursion_limit": settings.max_agent_steps,
        "metadata": {"langfuse_session_id": thread_id},
    }
