from pathlib import Path
import logging
import subprocess
import cv2
import numpy as np

from app.analyzers.event_analyzer import detect_action_events
from app.analyzers.video_analyzer import detect_face_tracks
from app.services.caption_generator import write_ass
from app.models.schemas import MAX_SHORT_DURATION_SECONDS
from app.utils.ffmpeg import get_ffmpeg_path


class ClipGenerationError(RuntimeError):
    pass


logger = logging.getLogger(__name__)


def _piecewise_expression(points: list[tuple[float, float]], default: float) -> str:
    if not points:
        return f"{default:.3f}"
    expression = f"{points[-1][1]:.3f}"
    for index in range(len(points) - 2, -1, -1):
        t0, value0 = points[index]
        t1, value1 = points[index + 1]
        duration = max(0.001, t1 - t0)
        interpolation = f"({value0:.3f}+({value1 - value0:.3f})*(t-{t0:.3f})/{duration:.3f})"
        expression = f"if(lt(t\\,{t1:.3f})\\,{interpolation}\\,{expression})"
    if points[0][0] > 0:
        expression = f"if(lt(t\\,{points[0][0]:.3f})\\,{points[0][1]:.3f}\\,{expression})"
    return expression


def _zoom_windows(points: list[tuple], duration: float, min_gap: float, max_count: int, window: float, prioritize_size: bool = False) -> list[tuple[float, float]]:
    candidates = sorted(points, key=lambda point: point[3] if prioritize_size else point[0], reverse=prioritize_size)
    selected = []
    for point in candidates:
        time = float(point[0])
        if any(abs(time - existing) < min_gap for existing in selected):
            continue
        selected.append(time)
        if len(selected) >= max_count:
            break
    return [
        (max(0.0, time - .18), min(duration, time + window - .18))
        for time in sorted(selected)
    ]


def _windows_expression(windows: list[tuple[float, float]]) -> str:
    if not windows:
        return "0"
    return "+".join(f"between(t\\,{start:.3f}\\,{end:.3f})" for start, end in windows)


def _zoom_crop_plan(source: Path, start: float, end: float) -> tuple:
    capture = cv2.VideoCapture(str(source))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if width < 2 or height < 2:
        raise ClipGenerationError("Could not read the source video dimensions for portrait framing.")

    face_track, focus_track = detect_face_tracks(source, start, end)
    fit_scale = min(1080 / width, 1920 / height)

    def bounded_zoom(scale: float) -> tuple[int, int, int, int]:
        zoom_width = max(2, int(width * fit_scale * scale) // 2 * 2)
        zoom_height = max(2, int(height * fit_scale * scale) // 2 * 2)
        return zoom_width, zoom_height, min(1080, zoom_width) // 2 * 2, min(1920, zoom_height) // 2 * 2

    mild_width, mild_height, mild_view_width, mild_view_height = bounded_zoom(1.18)
    focus_width, focus_height, focus_view_width, focus_view_height = bounded_zoom(1.30)

    mild_x_points = [(time, max(0.0, min(mild_width - mild_view_width, x * mild_width - mild_view_width / 2))) for time, x, _y in face_track]
    mild_y_points = [(time, max(0.0, min(mild_height - mild_view_height, y * mild_height - mild_view_height / 2))) for time, _x, y in face_track]
    mild_x = _piecewise_expression(mild_x_points, (mild_width - mild_view_width) / 2)
    mild_y = _piecewise_expression(mild_y_points, (mild_height - mild_view_height) / 2)

    # Reserve only a couple of the clearest stable-face moments for a portrait
    # crop that fills the canvas. Ordinary face moments use the smaller 18–30%
    # punch-ins. Space the full-screen shots apart to keep the edit varied.
    clip_duration = end - start
    screen_windows = _zoom_windows(focus_track, clip_duration, min_gap=18.0, max_count=2, window=1.25, prioritize_size=True)
    focus_windows = _zoom_windows(focus_track, clip_duration, min_gap=8.0, max_count=6, window=1.25, prioritize_size=True)
    focus_windows = [item for item in focus_windows if not any(max(item[0], screen[0]) < min(item[1], screen[1]) for screen in screen_windows)][:4]
    occupied = screen_windows + focus_windows
    mild_windows = _zoom_windows(face_track, clip_duration, min_gap=6.0, max_count=8, window=1.35)
    mild_windows = [item for item in mild_windows if not any(max(item[0], other[0]) < min(item[1], other[1]) for other in occupied)]

    focus_x_points = [(time, max(0.0, min(focus_width - focus_view_width, x * focus_width - focus_view_width / 2))) for time, x, _y, _size in focus_track]
    focus_y_points = [(time, max(0.0, min(focus_height - focus_view_height, y * focus_height - focus_view_height / 2))) for time, _x, y, _size in focus_track]
    focus_x = _piecewise_expression(focus_x_points, (focus_width - focus_view_width) / 2)
    focus_y = _piecewise_expression(focus_y_points, (focus_height - focus_view_height) / 2)

    screen_x = screen_y = "0"
    screen_crop_width = screen_crop_height = 2
    if focus_track:
        face_height = float(np.median([point[3] for point in focus_track])) * height
        screen_crop_height = min(height - height % 2, max(int(height * .34), int(face_height * 2.25)) // 2 * 2)
        screen_crop_width = max(2, int(screen_crop_height * 9 / 16) // 2 * 2)
        if screen_crop_width > width:
            screen_crop_width = width - width % 2
            screen_crop_height = min(height - height % 2, int(screen_crop_width * 16 / 9) // 2 * 2)
        screen_x_points = [(time, max(0.0, min(width - screen_crop_width, x * width - screen_crop_width / 2))) for time, x, _y, _size in focus_track]
        # Place the detected face in the upper third so the crop can include
        # more of the character below the face while tracking their movement.
        screen_y_points = [(time, max(0.0, min(height - screen_crop_height, y * height - screen_crop_height * .32))) for time, _x, y, _size in focus_track]
        screen_x = _piecewise_expression(screen_x_points, (width - screen_crop_width) / 2)
        screen_y = _piecewise_expression(screen_y_points, (height - screen_crop_height) / 2)

    return (
        mild_view_width, mild_view_height, mild_width, mild_height, mild_x, mild_y,
        _windows_expression(mild_windows), focus_view_width, focus_view_height, focus_width, focus_height,
        focus_x, focus_y, _windows_expression(focus_windows), screen_crop_width, screen_crop_height,
        screen_x, screen_y, _windows_expression(screen_windows), bool(face_track), bool(focus_track), bool(screen_windows),
    )


def generate_short(source: Path, transcript: list[dict], start: float, end: float, output: Path, work_dir: Path) -> tuple[str, int]:
    if start < 0 or end <= start or end - start > MAX_SHORT_DURATION_SECONDS:
        raise ClipGenerationError(f"A Short must be longer than zero and no more than {MAX_SHORT_DURATION_SECONDS} seconds.")
    captions = work_dir / "captions.ass"
    try:
        action_events = detect_action_events(source, start, end)
        logger.info("Detected %d synchronized action beat(s) for brief emoji overlays.", len(action_events))
    except Exception:
        action_events = []
        logger.exception("Action beat detection failed; continuing with transcript-timed captions and emoji cues.")
    emoji_events = write_ass(transcript, start, end, captions, action_events=action_events)
    logger.info("Prepared %d visible emoji overlay(s) for this Short.", len(emoji_events))
    subtitle_path = str(captions.resolve()).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    (
        view_width, view_height, zoom_width, zoom_height, crop_x, crop_y,
        mild_zoom_window, focus_view_width, focus_view_height, focus_zoom_width, focus_zoom_height, focus_x, focus_y,
        full_zoom_window, screen_crop_width, screen_crop_height, screen_x, screen_y, screen_window,
        face_framed, focus_framed, has_screen_zoom,
    ) = _zoom_crop_plan(source, start, end)

    # The fitted foreground preserves the portrait canvas and blurred fill.
    # Face-gated punch-ins briefly enlarge that foreground by 15% or at most
    # 30%, keeping the tracked subject inside the crop.
    video_filter = (
        "[0:v]setpts=PTS-STARTPTS,fps=25,split=5[background_source][fit_source][close_source][focus_source][screen_source];"
        "[background_source]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,boxblur=24:2,eq=brightness=-0.16:saturation=0.82[background];"
        "[fit_source]scale=1080:1920:force_original_aspect_ratio=decrease:force_divisible_by=2[fit];"
        "[background][fit]overlay=x=(W-w)/2:y=(H-h)/2:shortest=1[canvas];"
        f"[close_source]scale={zoom_width}:{zoom_height},crop={view_width}:{view_height}:x='{crop_x}':y='{crop_y}':exact=1,setsar=1[close];"
        f"[focus_source]scale={focus_zoom_width}:{focus_zoom_height},crop={focus_view_width}:{focus_view_height}:x='{focus_x}':y='{focus_y}':exact=1,setsar=1[focus];"
        f"[canvas][close]overlay=x=(W-w)/2:y=(H-h)/2:enable='{mild_zoom_window}':shortest=1[mild_canvas];"
        f"[mild_canvas][focus]overlay=x=(W-w)/2:y=(H-h)/2:enable='{full_zoom_window}':shortest=1[focus_canvas];"
        f"[screen_source]crop={screen_crop_width}:{screen_crop_height}:x='{screen_x}':y='{screen_y}':exact=1,scale=1080:1920,setsar=1[screen];"
        f"[focus_canvas][screen]overlay=x=0:y=0:enable='{screen_window}':shortest=1,"
        f"subtitles='{subtitle_path}'[captioned]"
    )
    command = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(start), "-i", str(source), "-t", str(end - start),
    ]
    current_label = "captioned"
    for index, event in enumerate(emoji_events):
        emoji_path = work_dir / f"emoji-{index}.png"
        render_emoji_overlay(event["emoji"], emoji_path)
        command.extend(["-loop", "1", "-framerate", "25", "-i", str(emoji_path)])
        next_label = f"emoji{index}"
        local_start = event["time"]
        local_end = local_start + event["duration"]
        video_filter += (
            f";[{current_label}][{index + 1}:v]overlay=x=(W-w)/2:y=H*0.17:"
            f"enable='between(t\\,{local_start:.3f}\\,{local_end:.3f})':shortest=1[{next_label}]"
        )
        current_label = next_label
    video_filter += f";[{current_label}]null[outv]"
    command.extend([
        "-filter_complex", video_filter, "-map", "[outv]", "-map", "0:a:0?",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output),
    ])
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=900, check=False)
    except FileNotFoundError as exc:
        raise ClipGenerationError("FFmpeg is missing. Install FFmpeg and restart the backend.") from exc
    except subprocess.TimeoutExpired as exc:
        raise ClipGenerationError("Short rendering timed out. Try again.") from exc
    if completed.returncode or not output.is_file() or output.stat().st_size == 0:
        details = (completed.stderr or "").strip().splitlines()
        detail = details[-1][:240] if details else "Check that the source has a video and audio track."
        raise ClipGenerationError(f"FFmpeg could not render this Short. {detail}")
    framing = "full-screen" if has_screen_zoom else "face-tracked" if focus_framed else "no-zoom" if not face_framed else "group-framed"
    return framing, len(emoji_events)
