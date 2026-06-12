from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Set

# Absolute path: bfai-assessment/backend/core/config.py → ../  = backend/ → ../ = project root
# .env lives in backend/, two levels up from this file
_ENV_FILE = Path(__file__).parent.parent / ".env"


class Settings(BaseSettings):
    GROQ_API_KEY: str = ""
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000", "https://doc-mind-three.vercel.app"]
    MAX_FILE_SIZE_MB: int = 20
    ALLOWED_EXTENSIONS: Set[str] = {".pdf", ".png", ".jpg", ".jpeg", ".tiff"}
    STORAGE_PATH: str = "./storage"
    CHROMA_PATH: str = "./storage/chroma_db"
    IMAGES_PATH: str = "./storage/page_images"
    RATE_LIMIT_PER_MINUTE: int = 30

    class Config:
        env_file = str(_ENV_FILE)
        env_file_encoding = "utf-8"


settings = Settings()
