"""Central settings. Defaults match the previous hardcoded values but are
repo-relative, so the project no longer depends on C:\\NeuroVisionCombined4.
Override via environment variables (NV_*) or .env.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_prefix="NV_")

    models_dir: Path = PROJECT_ROOT / "models"
    chroma_db_dir: Path = PROJECT_ROOT / "RAG" / "chroma_db"

    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "admin"
    minio_secret_key: str = "password123"
    minio_bucket: str = "neuro-vision"
    minio_secure: bool = False


settings = Settings()