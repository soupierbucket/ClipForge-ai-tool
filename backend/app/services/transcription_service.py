from pathlib import Path
from collections.abc import Callable
from faster_whisper import WhisperModel
from app.config import settings

_model: WhisperModel | None = None


def transcribe_video(video_path: Path, on_progress: Callable[[str], None] | None = None) -> list[dict]:
    global _model
    try:
        if _model is None:
            model_name = settings.whisper_model
            # Translation requires a multilingual Whisper checkpoint. If the
            # user selected an English-only alias, use its multilingual peer.
            if model_name.endswith(".en"):
                model_name = model_name[:-3]
            if on_progress:
                on_progress(f"Loading the local multilingual Whisper {model_name} model on {settings.whisper_device} for English captions.")
            _model = WhisperModel(model_name, device=settings.whisper_device, compute_type="int8" if settings.whisper_device == "cpu" else "float16")
        if on_progress:
            on_progress("Transcribing speech and translating it into English captions with word timestamps.")
        segments, _info = _model.transcribe(
            str(video_path),
            task="translate",
            language=None,
            word_timestamps=True,
            vad_filter=True,
        )
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
