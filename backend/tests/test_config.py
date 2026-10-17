import pytest
from pydantic import ValidationError

from app.core.config import Settings

BASE = {
    "postgres_uri": "postgresql+asyncpg://u:p@db:5432/x",
    "redis_uri": "redis://redis:6379",
    "openai_api_key": "k",
}


def make(**overrides):
    # _env_file=None: тест не должен зависеть от настоящего .env разработчика
    return Settings(_env_file=None, **{**BASE, **overrides})


def test_placeholder_api_key_is_rejected():
    with pytest.raises(ValidationError):
        make(api_key="замените-на-длинную-случайную-строку")


def test_short_api_key_is_rejected():
    with pytest.raises(ValidationError):
        make(api_key="short")


def test_checkpoint_uri_uses_psycopg_scheme():
    settings = make(api_key="a-real-key-1234567890")
    assert settings.checkpoint_uri == "postgresql://u:p@db:5432/x"


def test_cors_origins_are_split_and_trimmed():
    settings = make(api_key="a-real-key-1234567890", cors_origins="http://a.com, http://b.com ,")
    assert settings.cors_origin_list == ["http://a.com", "http://b.com"]


def test_langfuse_is_disabled_without_keys():
    settings = make(api_key="a-real-key-1234567890")
    assert settings.langfuse_enabled is False
