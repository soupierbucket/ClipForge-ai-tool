from pathlib import Path

import cv2
import numpy as np

from app.utils.audio import decode_mono_audio


def detect_action_events(path: Path, start: float, end: float) -> list[dict]:
    """Find a few synchronized audio/video intensity bursts for brief overlays.

    This is a conservative action-cue heuristic, not semantic object or event
    recognition. Dialogue keywords are handled separately by the captioner.
    """
    duration = max(0.0, end - start)
    if duration < .5:
        return []

    audio_peaks = _audio_peaks(path, start, duration)
    visual_peaks = _visual_peaks(path, start, end)
    events = []
    for audio_time in audio_peaks:
        nearby = [time for time in visual_peaks if abs(time - audio_time) <= .75]
        if nearby:
            events.append({"time": audio_time, "duration": .9, "emoji": "💥"})

    # Avoid a burst of repeated overlays during a rapid sequence.
    selected = []
    for event in sorted(events, key=lambda item: item["time"]):
        if not selected or event["time"] - selected[-1]["time"] >= 2.0:
            selected.append(event)
    return selected[:6]


def _audio_peaks(path: Path, start: float, duration: float) -> list[float]:
    samples, sample_rate = decode_mono_audio(path, start, duration, 11025)
    step = int(sample_rate * .25)
    if len(samples) < step * 2:
        return []
    rms = np.array([
        float(np.sqrt(np.mean(np.square(samples[index:index + step]))))
        for index in range(0, len(samples) - step + 1, step)
    ])
    if len(rms) < 3:
        return []
    threshold = max(.12, float(np.median(rms)) * 2.5, float(np.percentile(rms, 90)) * 1.2)
    peaks = []
    for index in range(1, len(rms) - 1):
        if rms[index] >= threshold and rms[index] >= rms[index - 1] and rms[index] >= rms[index + 1]:
            peaks.append(start + (index + .5) * .25)
    return peaks


def _visual_peaks(path: Path, start: float, end: float) -> list[float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return []
    differences = []
    previous = None
    times = np.arange(start + .25, end, .5)
    for second in times:
        capture.set(cv2.CAP_PROP_POS_MSEC, float(second) * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        gray = cv2.cvtColor(cv2.resize(frame, (96, 54)), cv2.COLOR_BGR2GRAY)
        if previous is not None:
            difference = float(np.mean(cv2.absdiff(gray, previous))) / 255
            differences.append((float(second), difference))
        previous = gray
    capture.release()
    if not differences:
        return []
    values = np.array([value for _, value in differences])
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = max(.18, median + 3.0 * mad, float(np.percentile(values, 90)) * 1.25)
    return [time for time, value in differences if value >= threshold]
