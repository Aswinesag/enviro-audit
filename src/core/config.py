# src/core/config.py - Update with production settings
import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    app_name: str = "EnviroAudit"
    environment: str = os.getenv("ENVIRONMENT", "development")
    
    # Server settings
    api_host: str = "0.0.0.0"
    api_port: int = int(os.getenv("PORT", 8000))
    
    # Database (will use Supabase in production)
    database_url: str = os.getenv(
        "DATABASE_URL", 
        "sqlite:///data/enviroaudit.db"
    )
    
    # Model cache (for faster loading)
    model_cache_dir: str = "/tmp/huggingface_cache"
    
    # Optional: HuggingFace token (for faster downloads)
    huggingface_token: Optional[str] = os.getenv("HUGGINGFACE_TOKEN")
    
    # CORS settings for production
    cors_origins: list = [
        "https://enviroaudit.streamlit.app",
        "http://localhost:8501",
        "http://localhost:8000"
    ]
    
    class Config:
        env_file = ".env"

settings = Settings()