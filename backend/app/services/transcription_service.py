from pathlib import Path
from collections.abc import Callable
from faster_whisper import WhisperModel
from app.config import settings

_model: WhisperModel | None = None


def transcribe_video(video_path: Path, on_progress: Callable[[str], None] | None = None) -> list[dict]:
    global _model
    try:
        if _model is None:
            if on_progress:
                on_progress(f"Loading the local Whisper {settings.whisper_model} model on {settings.whisper_device}.")
            _model = WhisperModel(settings.whisper_model, device=settings.whisper_device, compute_type="int8" if settings.whisper_device == "cpu" else "float16")
        if on_progress:
            on_progress("Transcribing the video's audio into timestamped speech segments.")
        segments, _info = _model.transcribe(str(video_path), word_timestamps=True, vad_filter=True)
        transcript = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                words = [
                    {"start": round(float(word.start), 3), "end": round(float(word.end), 3), "text": word.word}
                    for word in (segment.words or [])
                    if word.word and word.end > word.start
                ]
                transcript.append({"start": round(float(segment.start), 2), "end": round(float(segment.end), 2), "text": text, "words": words})
                if on_progress and len(transcript) % 25 == 0:
                    on_progress(f"Transcribed {len(transcript)} speech segments so far.")
    except Exception as exc:
        if on_progress:
            on_progress(f"Local transcription was unavailable ({type(exc).__name__}); continuing with timeline samples.")
        return []
    if not transcript:
        if on_progress:
            on_progress("No timestamped speech was detected; continuing with clearly labeled timeline samples.")
    return transcript
