import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App Configuration
    APP_NAME: str = "Game ASR and Command API"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Whisper ASR Configuration
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "large-v2")
    USE_VAD: bool = os.getenv("USE_VAD", "True").lower() in ("true", "1", "t")

    # Ollama Configuration
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")

    # WebSocket Configuration
    WS_PING_INTERVAL: float = float(os.getenv("WS_PING_INTERVAL", "20"))
    WS_PING_TIMEOUT: float = float(os.getenv("WS_PING_TIMEOUT", "20"))

    # Command Processing
    CMD_RECOGNITION_CONFIDENCE_THRESHOLD: float = float(
        os.getenv("CMD_RECOGNITION_CONFIDENCE_THRESHOLD", "0.6")
    )

    # CUDA Configuration
    CUDA_VISIBLE_DEVICES: str = os.getenv("CUDA_VISIBLE_DEVICES", None)

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
