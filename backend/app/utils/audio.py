from pathlib import Path
import subprocess

import numpy as np

from app.utils.ffmpeg import get_ffmpeg_path


def decode_mono_audio(path: Path, start: float, duration: float, sample_rate: int) -> tuple[np.ndarray, int]:
    """Decode a bounded mono float stream with FFmpeg, without librosa/SciPy."""
    command = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-nostdin",
        "-ss", str(max(0.0, start)), "-i", str(path), "-t", str(max(0.1, duration)),
        "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "pipe:1",
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=180, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("FFmpeg could not decode the audio segment.") from exc
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip().splitlines()
        raise RuntimeError(detail[-1][:240] if detail else "FFmpeg could not decode the audio segment.")
    samples = np.frombuffer(result.stdout, dtype="<f4")
    return samples, sample_rate
