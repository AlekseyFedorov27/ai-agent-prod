import pytest
from fastapi import HTTPException

from app.core import ratelimit
from app.core.config import settings


class FakeRedis:
    def __init__(self):
        self.counts: dict[str, int] = {}
        self.expires: dict[str, int] = {}

    async def incr(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    async def expire(self, key, seconds):
        self.expires[key] = seconds


class BrokenRedis:
    async def incr(self, key):
        raise ConnectionError("redis is down")


async def test_allows_requests_up_to_limit_then_returns_429(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(ratelimit, "get_redis", lambda: fake)
    monkeypatch.setattr(settings, "rate_limit_per_minute", 3)

    for _ in range(3):
        await ratelimit.rate_limit()

    with pytest.raises(HTTPException) as exc_info:
        await ratelimit.rate_limit()
    assert exc_info.value.status_code == 429
    assert exc_info.value.headers["Retry-After"] == "60"
    assert fake.expires, "у окна должен быть TTL, иначе ключи копятся"


async def test_fails_open_when_redis_is_down(monkeypatch):
    monkeypatch.setattr(ratelimit, "get_redis", lambda: BrokenRedis())
    await ratelimit.rate_limit()  # не должно бросать исключение
