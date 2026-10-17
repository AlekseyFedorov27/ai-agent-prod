import os

# Настройки читаются при импорте приложения, поэтому задаём их до любых импортов app.*
os.environ.setdefault("POSTGRES_URI", "postgresql+asyncpg://user:pass@localhost:5432/test_db")
os.environ.setdefault("REDIS_URI", "redis://localhost:6379")
os.environ.setdefault("API_KEY", "test-api-key-1234567890")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
# В тестах трассировка всегда выключена
os.environ["LANGFUSE_HOST"] = ""
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""
