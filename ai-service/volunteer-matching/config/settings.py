from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    backend_base_url: str = "http://localhost:5000"
    campaign_endpoint: str = "/api/events/"
    users_endpoint: str = "/api/users/volunteers"
    vector_store_dir: str = "./vector_store"
    top_n_candidates: int = 50
    top_k_recommendations: int = 10
    embedding_model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2"
    max_distance_km: float = 50.0

    # LLM Settings
    groq_api_key: Optional[str] = None
    groq_model: str = "llama-3.3-70b-versatile"
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    llm_provider: str = "groq"  # "groq", "openai", "gemini", or "fallback"

    model_config = SettingsConfigDict(env_file=(".env", "../.env", "../../.env"), env_file_encoding="utf-8", extra="ignore")
