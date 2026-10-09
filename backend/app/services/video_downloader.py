from pathlib import Path
import logging
from collections.abc import Callable
import yt_dlp
from app.config import settings
from app.utils.ffmpeg import get_ffmpeg_path, probe_video_dimensions
from app.utils.urls import normalize_youtube_url


class DownloadError(RuntimeError):
    pass


logger = logging.getLogger(__name__)


def get_video_info(url: str) -> dict:
    normalized = normalize_youtube_url(url)
    options = {"quiet": True, "no_warnings": True, "noplaylist": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            data = ydl.extract_info(normalized, download=False)
    except Exception as exc:
        raise DownloadError("We couldn't retrieve this video. Check the link and try again.") from exc
    duration = float(data.get("duration") or 0)
    if duration <= 0:
        raise DownloadError("This video has no usable duration information.")
    return {"title": data.get("title") or "Untitled video", "channel": data.get("channel") or data.get("uploader") or "Unknown channel", "duration": duration, "thumbnail": data.get("thumbnail") or "", "url": normalized}


def _source_files(job_dir: Path) -> list[Path]:
    return [
        item for item in job_dir.glob("source.*")
        if item.is_file() and item.suffix.lower() not in {".part", ".ytdl"}
    ]


def _remove_download_attempt(job_dir: Path) -> None:
    # This fixed template only covers downloads owned by the current job.
    for item in job_dir.glob("source.*"):
        if item.is_file():
            item.unlink(missing_ok=True)


def download_video(url: str, job_dir: Path, on_progress: Callable[[str], None] | None = None) -> Path:
    job_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = settings.max_video_size_mb * 1024 * 1024
    ffmpeg_path = get_ffmpeg_path()
    last_error: Exception | None = None
    selectors = [
        "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "bestvideo[height<=480]+bestaudio/best[height<=480]",
        "bestvideo[height<=360]+bestaudio/best[height<=360]",
        "bestvideo[height<=240]+bestaudio/best[height<=240]",
    ]
    for selector in selectors:
        height_limit = selector.split("<=", 1)[1].split("]", 1)[0]
        if on_progress:
            on_progress(f"Downloading a video stream at up to {height_limit}p within the {settings.max_video_size_mb} MB limit.")
        _remove_download_attempt(job_dir)
        options = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "format": selector,
            "outtmpl": str(job_dir / "source.%(ext)s"),
            "merge_output_format": "mp4",
            "max_filesize": max_bytes,
            "ffmpeg_location": ffmpeg_path,
        }
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([normalize_youtube_url(url)])
        except Exception as exc:
            last_error = exc
            logger.warning("yt-dlp download failed at selector %s; retrying at lower resolution.", selector, exc_info=True)
            if on_progress:
                on_progress(f"Download failed at {height_limit}p; retrying at a lower resolution.")
            continue

        completed_video_files = [
            path for path in _source_files(job_dir)
            if probe_video_dimensions(path) is not None and path.stat().st_size <= max_bytes
        ]
        if completed_video_files:
            source = max(completed_video_files, key=lambda path: path.stat().st_size)
            height_limit = selector.split("<=", 1)[1].split("]", 1)[0]
            logger.info("Selected a valid video source at <=%sp: %s (%d bytes).", height_limit, source.name, source.stat().st_size)
            if on_progress:
                on_progress(f"A playable video stream is ready at up to {height_limit}p.")
            return source

        logger.warning("yt-dlp produced no complete video stream under %d MB at selector %s; trying lower resolution.", settings.max_video_size_mb, selector)
        if on_progress:
            on_progress(f"No complete video stream fit at {height_limit}p; retrying at a lower resolution.")

    _remove_download_attempt(job_dir)
    message = f"Could not download a complete video stream within the {settings.max_video_size_mb} MB limit, even at 240p. Try a shorter video or raise MAX_VIDEO_SIZE_MB."
    if last_error:
        logger.error("All yt-dlp resolution attempts failed; last error: %s", last_error)
    raise DownloadError(message) from last_error
