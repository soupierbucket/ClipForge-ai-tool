from pathlib import Path
import logging
import yt_dlp
from app.config import settings
from app.utils.ffmpeg import get_ffmpeg_path
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


def download_video(url: str, job_dir: Path) -> Path:
    job_dir.mkdir(parents=True, exist_ok=True)
    options = {"quiet": True, "no_warnings": True, "noplaylist": True, "format": "bestvideo*+bestaudio/best", "outtmpl": str(job_dir / "source.%(ext)s"), "merge_output_format": "mp4", "max_filesize": settings.max_video_size_mb * 1024 * 1024, "ffmpeg_location": get_ffmpeg_path()}
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([normalize_youtube_url(url)])
    except Exception as exc:
        logger.exception("yt-dlp failed while downloading the source video")
        raise DownloadError("The source video download failed. Check access to the video and try again.") from exc
    files = [item for item in job_dir.glob("source.*") if item.is_file() and item.suffix not in {".part", ".ytdl"}]
    if not files:
        raise DownloadError("The download completed without producing a video file.")
    source = files[0]
    if source.stat().st_size > settings.max_video_size_mb * 1024 * 1024:
        source.unlink(missing_ok=True)
        raise DownloadError(f"This video exceeds the {settings.max_video_size_mb} MB processing limit.")
    return source
