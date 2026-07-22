"""Application configuration"""
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List

PROJECT_ROOT = Path(__file__).parent.parent.parent

class Settings(BaseSettings):
    api_host: str = "127.0.0.1"
    api_port: int = 8001
    database_url: str = f"sqlite:///{PROJECT_ROOT}/recruitment.db"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    ner_model: str = "xlm-roberta-base-finetuned-conll03-english"
    custom_ner_model_path: Path = PROJECT_ROOT / "models" / "ner"
    models_dir: Path = PROJECT_ROOT / "models"
    min_skill_match_threshold: float = 60.0
    match_shortlist_threshold: float = 0.75
    match_review_threshold: float = 0.60
    embedding_weight: float = 0.7
    skill_weight: float = 0.3
    logs_dir: Path = PROJECT_ROOT / "logs"
    uploads_dir: Path = PROJECT_ROOT / "uploads"
    
    # Missing these - add them:
    # Add these to your Settings class in app/core/config.py
    allowed_extensions: List[str] = ['.pdf', '.docx', '.doc', '.txt']
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    
    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()