from pathlib import Path
import shutil

from app.config import settings


def get_ffmpeg_path() -> str:
    """Return an FFmpeg executable path, including common Windows installs."""
    configured = settings.ffmpeg_location
    if configured:
        path = Path(configured).expanduser()
        if path.is_dir():
            path = path / ("ffmpeg.exe" if path.drive else "ffmpeg")
        return str(path)

    found = shutil.which("ffmpeg")
    if found:
        return found

    # WinGet installs are often not added to Git Bash's inherited PATH.
    winget_packages = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    matches = sorted(winget_packages.glob("Gyan.FFmpeg.Shared_*/ffmpeg-*-full_build-shared/bin/ffmpeg.exe"))
    if matches:
        return str(matches[-1])
    return "ffmpeg"


def get_ffprobe_path() -> str:
    """Resolve FFprobe next to the configured/discovered FFmpeg executable."""
    ffmpeg = Path(get_ffmpeg_path())
    probe_name = "ffprobe.exe" if ffmpeg.suffix.lower() == ".exe" else "ffprobe"
    if ffmpeg.parent != Path("."):
        sibling = ffmpeg.with_name(probe_name)
        if sibling.is_file():
            return str(sibling)
    return shutil.which(probe_name) or probe_name


def probe_video_dimensions(path: Path) -> tuple[int, int] | None:
    """Read the first video stream dimensions without relying on OpenCV codecs."""
    import json
    import subprocess

    try:
        result = subprocess.run(
            [get_ffprobe_path(), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30, check=False,
        )
        streams = json.loads(result.stdout).get("streams", []) if result.returncode == 0 else []
        if not streams:
            return None
        width, height = int(streams[0].get("width", 0)), int(streams[0].get("height", 0))
        return (width, height) if width > 1 and height > 1 else None
    except (OSError, subprocess.TimeoutExpired, ValueError, TypeError):
        return None
