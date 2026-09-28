"""
Application settings for AgriVoice.

One class — AgriVoiceConfig — reads env vars and provides typed access
to every setting and derived path the app needs.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgriVoiceConfig(BaseSettings):
    """Single config object for the entire AgriVoice app."""

    # App settings
    app_name: str = 'agrivoice'
    environment: str = 'development'

    # Model settings
    llm_model: str = 'gpt-4o-mini'
    embedding_model: str = 'text-embedding-3-small'
    top_k: int = 4
    temperature: float = 0.0

    # Paths
    project_root: str = '.'
    chroma_dir: str = ''

    # Logging
    log_level: str = 'INFO'

    model_config = SettingsConfigDict(
        env_prefix='AGRIVOICE_',
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    @property
    def root_path(self) -> Path:
        """Project root as a Path object."""
        return Path(self.project_root)

    @property
    def chroma_path(self) -> Path:
        """Chroma store path — from env var or default."""
        if self.chroma_dir:
            return Path(self.chroma_dir)
        return self.root_path / 'chroma_db_domain'

    @property
    def logs_path(self) -> Path:
        """Logs folder."""
        return self.root_path / 'logs'

    @property
    def is_production(self) -> bool:
        """True when running in production."""
        return self.environment.lower() == 'production'
