import logging
import time
from functools import lru_cache

import redis.asyncio as aioredis
from fastapi import HTTPException

from app.core.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_redis() -> aioredis.Redis:
    return aioredis.from_url(settings.redis_uri, decode_responses=True)


async def rate_limit() -> None:
    """Общий лимит запросов к чату в минуту (фиксированное окно в Redis).

    Если Redis недоступен, запрос пропускается: сбой кэша не должен
    ронять чат. Это осознанный компромисс, а не ошибка.
    """
    window = int(time.time() // 60)
    key = f"rl:chat:{window}"
    try:
        redis = get_redis()
        count = await redis.incr(key)
        if count == 1:
            await redis.expire(key, 70)
    except Exception:
        logger.warning("Redis недоступен, лимит запросов пропущен", exc_info=True)
        return

    if count > settings.rate_limit_per_minute:
        raise HTTPException(
            status_code=429,
            detail="Слишком много запросов, повторите через минуту",
            headers={"Retry-After": "60"},
        )
