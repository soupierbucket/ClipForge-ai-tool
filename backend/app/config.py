from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    frontend_origins: str = "http://localhost:5173"
    clipforge_temp_dir: str | None = None
    max_video_size_mb: int = 1024
    max_video_duration_seconds: int = 10800
    ffmpeg_location: str | None = None
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    llm_provider: str = "ollama"
    llm_api_key: str | None = None
    llm_model: str = "qwen2.5:7b"
    ollama_base_url: str = "http://localhost:11434"
    job_retention_hours: int = 24
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def origins(self) -> list[str]:
        return [origin.strip() for origin in self.frontend_origins.split(",") if origin.strip()]

    @property
    def media_dir(self) -> Path:
        location = self.clipforge_temp_dir or str(Path(__file__).resolve().parents[1] / "temp_media")
        return Path(location).resolve()

    @property
    def jobs_dir(self) -> Path:
        return self.media_dir / "jobs"

    @property
    def clips_dir(self) -> Path:
        return self.media_dir / "clips"


settings = Settings()
