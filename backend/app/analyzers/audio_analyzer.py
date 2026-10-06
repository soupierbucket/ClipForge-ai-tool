import numpy as np
from pathlib import Path

from app.utils.audio import decode_mono_audio


def analyze_audio(path: Path, start: float, end: float) -> dict:
    sr = 22050
    y, sr = decode_mono_audio(path, start, max(.1, end - start), sr)
    frame_length = max(1, int(sr * .025))
    usable = len(y) - (len(y) % frame_length)
    if usable:
        frames = y[:usable].reshape(-1, frame_length)
        rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    else:
        rms = np.array([0.0])
    peak_level = float(np.max(rms)) if len(rms) else 0.0
    if peak_level <= 1e-8:
        db = np.full_like(rms, -80.0)
    else:
        db = 20 * np.log10(np.maximum(rms, 1e-8) / peak_level)
    active = float(np.mean(db > -35)) if len(db) else 0.0
    peaks = float(np.mean(db > -8)) if len(db) else 0.0
    score = max(0.0, min(10.0, (active * 7.0) + (peaks * 30.0)))
    return {"score": round(score, 1), "active_ratio": round(active, 2), "peak_ratio": round(peaks, 2), "duration": round(len(y) / sr, 2) if sr else 0}
