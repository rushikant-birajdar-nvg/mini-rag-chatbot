from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    jwt_secret: str = Field(..., alias="JWT_SECRET")
    jwt_algorithm: str = "HS256"

    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_collection: str = "policy_chunks"

    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL")

    llm_provider: Literal["ollama", "openai", "groq", "gemini", "anthropic"] = Field(
        default="ollama", alias="LLM_PROVIDER"
    )
    llm_temperature: float = Field(default=0.2, alias="LLM_TEMPERATURE")

    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.2", alias="OLLAMA_MODEL")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")

    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1", alias="GROQ_BASE_URL")

    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    gemini_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta",
        alias="GEMINI_BASE_URL",
    )

    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(
        default="claude-haiku-4-5-20251001",
        alias="ANTHROPIC_MODEL",
    )
    
    chunk_size: int = 300
    chunk_overlap: int = 50

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

