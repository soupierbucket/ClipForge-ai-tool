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
