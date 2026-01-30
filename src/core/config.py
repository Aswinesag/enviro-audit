# src/core/config.py
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    app_name: str = "EnviroAudit"
    environment: str = "development"
    
    # Model settings
    clip_model: str = "openai/clip-vit-base-patch32"
    blip_model: str = "Salesforce/blip-image-captioning-base"
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    
    # Security (for production)
    api_key: Optional[str] = None
    cors_origins: list = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # External Services
    huggingface_token: Optional[str] = None
    
    # Sentinel Hub
    sentinelhub_client_id: Optional[str] = None
    sentinelhub_client_secret: Optional[str] = None
    sentinelhub_instance_id: Optional[str] = None
    
    # Database
    database_url: str = "sqlite:///./data/enviroaudit.db"
    
    class Config:
        env_file = ".env"

settings = Settings()