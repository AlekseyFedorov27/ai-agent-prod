import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.core.config import settings

# auto_error=False, чтобы вернуть единое сообщение на русском и код 401.
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(key: str | None = Security(_api_key_header)) -> None:
    """Проверяет заголовок X-API-Key (сравнение за постоянное время)."""
    if not key or not secrets.compare_digest(key.encode(), settings.api_key.encode()):
        raise HTTPException(status_code=401, detail="Неверный или отсутствующий API-ключ")
