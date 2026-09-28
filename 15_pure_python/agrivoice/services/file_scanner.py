"""
Simple file scanner for AgriVoice.

Scans a folder for images and logs every action.
Demonstrates pathlib + logging working together.
"""

import logging
from pathlib import Path

# Module-level logger — name shows up in every log line
logger = logging.getLogger(__name__)


class FileScanner:
    """Scans a folder for image files and logs each step."""

    IMAGE_PATTERNS = ['*.jpg', '*.jpeg', '*.png', '*.JPG']

    def __init__(self, folder: str):
        """Store the folder to scan."""
        self.folder = Path(folder)
        logger.debug(f'FileScanner created for {self.folder}')

    def scan(self) -> list[Path]:
        """Return a list of image files in the folder."""
        # Log the start of the scan
        logger.info(f'Scanning folder: {self.folder}')

        # Refuse to scan if the folder doesn't exist
        if not self.folder.exists():
            logger.error(f'Folder does not exist: {self.folder}')
            return []

        # Collect all images matching any pattern
        images = []
        for pattern in self.IMAGE_PATTERNS:
            found = list(self.folder.glob(pattern))
            images.extend(found)
            if found:
                logger.debug(f'  Pattern {pattern}: {len(found)} file(s)')

        # Log the final count
        logger.info(f'Found {len(images)} image(s)')
        return images
