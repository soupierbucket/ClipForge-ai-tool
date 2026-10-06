import logging
import shutil

from app.config import settings

logger = logging.getLogger(__name__)


def cleanup_temporary_media() -> None:
    """Remove only ClipForge's dedicated job and clip directories."""
    media_root = settings.media_dir.resolve()
    for directory in (settings.jobs_dir, settings.clips_dir):
        # Never follow a user-created symlink or delete outside the media root.
        if directory.is_symlink() or directory.resolve().parent != media_root:
            logger.warning("Skipping temporary media cleanup for unsafe path: %s", directory)
            continue
        if not directory.exists():
            continue
        try:
            shutil.rmtree(directory)
            logger.info("Removed temporary media directory: %s", directory)
        except OSError:
            logger.exception("Could not remove temporary media directory: %s", directory)
