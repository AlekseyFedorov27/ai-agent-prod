from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Инфраструктура ---
    postgres_uri: str
    redis_uri: str

    # --- Доступ к API ---
    # Единый ключ для заголовка X-API-Key. Минимум 16 символов.
    api_key: str = Field(min_length=16)
    # Разрешённые источники для CORS, через запятую.
    cors_origins: str = "http://localhost:3000,http://localhost:5173"
    # Общий лимит запросов к чату в минуту (защита бюджета LLM).
    rate_limit_per_minute: int = 30

    # --- LLM (любой OpenAI-совместимый провайдер) ---
    openai_api_key: str
    openai_base_url: str = "https://tokenharbor.ai/v1"
    llm_model: str = "deepseek-v4.1-flash:free"
    llm_temperature: float = 0.1
    # Сколько последних сообщений диалога отправлять модели.
    max_history_messages: int = 40
    # Предохранитель от бесконечных циклов агента (recursion_limit LangGraph).
    max_agent_steps: int = 12

    # --- Langfuse (необязателен: без ключей трассировка выключена) ---
    langfuse_host: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # --- RAG ---
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_dim: int = 768
    chunk_size: int = 500
    chunk_overlap: int = 50
    rag_top_k: int = 3

    # --- Загрузка файлов ---
    upload_dir: str = "/app/uploads"
    max_upload_mb: int = 10

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @field_validator("api_key")
    @classmethod
    def _api_key_is_not_placeholder(cls, value: str) -> str:
        # Защита от запуска с заглушкой из .env.example
        if value.lower().startswith(("замените", "change")):
            raise ValueError("Задайте настоящий API_KEY в .env (заглушка из примера не подходит)")
        return value

    @property
    def checkpoint_uri(self) -> str:
        """Строка подключения для чекпоинтера LangGraph (psycopg, а не asyncpg)."""
        return self.postgres_uri.replace("postgresql+asyncpg://", "postgresql://", 1)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_host and self.langfuse_public_key and self.langfuse_secret_key)


settings = Settings()
