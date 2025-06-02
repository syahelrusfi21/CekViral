# cekviral_project/app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "CekViral API"
    PROJECT_VERSION: str = "0.1.0"
    MODEL_PATH: str = "models/" # Path ke folder model ML
    YDL_TEMP_DIR: str = "temp_downloads/" # Direktori sementara untuk yt-dlp

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

settings = Settings()