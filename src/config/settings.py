from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Anthropic
    anthropic_api_key: str
    claude_model: str = "claude-sonnet-4-20250514"

    # Voyage AI (embeddings) — voyage-code-2 produces 1536-dim vectors
    voyage_api_key: str
    voyage_model: str = "voyage-code-2"
    voyage_embedding_dim: int = 1536

    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: str = "kyoto-codebase"
    pinecone_environment: str = "us-east-1"

    # Redis (for Celery, used in Phase 3+)
    redis_url: str = "redis://localhost:6379/0"

    # Ingestion defaults
    max_chunk_tokens: int = 512
    chunk_overlap_tokens: int = 64

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()