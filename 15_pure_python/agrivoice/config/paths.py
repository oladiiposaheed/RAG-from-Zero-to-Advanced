"""
Filesystem paths for AgriVoice.

One class, one source of truth — every path in the app comes from here.
"""

from pathlib import Path


class ConfigPaths:
    """Central place for all AgriVoice file paths."""

    def __init__(self, project_root: str):
        """Store the project root and derive every path from it."""
        self.root = Path(project_root)

    @property
    def chroma_dir(self) -> Path:
        """Where the Chroma vector store lives."""
        return self.root / 'chroma_db_domain'

    @property
    def data_dir(self) -> Path:
        """Where raw data files live."""
        return self.root / '04_data_ingestion_document_processing' / 'data'

    @property
    def model_dir(self) -> Path:
        """Where trained models live."""
        return self.root / 'models'

    @property
    def logs_dir(self) -> Path:
        """Where log files are written."""
        return self.root / 'logs'

    @property
    def tests_dir(self) -> Path:
        """Where tests and fixtures live."""
        return self.root / '15_pure_python' / 'agrivoice' / 'tests'

    def ensure_dirs(self) -> None:
        """Create every folder that should exist."""
        for folder in [self.model_dir, self.logs_dir]:
            folder.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f'ConfigPaths(root={self.root})'
