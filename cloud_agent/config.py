import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # LLM Choice: "mock", "openai", "gemini"
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "mock")
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    
    # Simple WebSocket security token
    DEVICE_AUTH_TOKEN: str = os.getenv("DEVICE_AUTH_TOKEN", "super-secret-device-token")

    class Config:
        env_file = ".env"

settings = Settings()
