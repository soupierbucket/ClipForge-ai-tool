from pathlib import Path
import re
from PIL import Image, ImageDraw, ImageFont


EMOJI_RULES = [
    ("💥", ("blast", "explosion", "explode", "exploded", "exploding", "bomb", "boom", "grenade", "detonate")),
    ("⚔️", ("attack", "attacked", "attacking", "fight", "fighting", "punch", "punched", "struck")),
    ("🔫", ("gunshot", "gunfire", "shoot", "shooting", "shot")),
    ("😂", ("haha", "hahaha", "lol", "funny", "hilarious", "laugh", "laughs", "laughed", "laughing", "laughter", "joke", "giggle", "giggling")),
    ("😮", ("wow", "no way", "surprise", "surprised", "shocked", "unexpected", "unbelievable")),
    ("🔥", ("amazing", "insane", "incredible", "epic", "wild", "awesome")),
    ("🏆", ("winner", "winning", "victory", "champion", "won")),
    ("❤️", ("love", "beautiful", "favorite", "favourite")),
    ("💡", ("tip", "trick", "secret", "lesson", "learn")),
    ("😬", ("awkward", "embarrassing", "oops", "mistake", "fail", "failed", "failure")),
]
EMOJI_MIN_SPACING_SECONDS = 2.0


def _ass_time(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remain = divmod(centiseconds, 360000)
    minutes, remain = divmod(remain, 6000)
    secs, cs = divmod(remain, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _safe_text(value: str) -> str:
    # Prevent transcript characters from being interpreted as ASS formatting tags.
    return re.sub(r"[{}\\]", "", value).replace("\n", " ").replace("\r", " ")


def _fallback_words(segment: dict) -> list[dict]:
    text = re.sub(r"\s+", " ", str(segment.get("text", ""))).strip()
    tokens = text.split()
    if not tokens:
        return []
    start, end = float(segment["start"]), float(segment["end"])
    span = max(0.01, end - start) / len(tokens)
    return [{"start": start + index * span, "end": start + (index + 1) * span, "text": token} for index, token in enumerate(tokens)]


def _emoji_for(text: str) -> str:
    normalized = text.lower()
    for emoji, keywords in EMOJI_RULES:
        if any(re.search(rf"\b{re.escape(keyword)}\w*\b", normalized) for keyword in keywords):
            return emoji
    return ""


def _emoji_events_from_words(words: list[dict]) -> list[dict]:
    events = []
    for index, word in enumerate(words):
        emoji = _emoji_for(str(word.get("text", "")))
        if not emoji:
            continue
        if emoji == "💥":
            duration = 1.0
        elif emoji in ("⚔️", "🔫"):
            duration = 0.55
        elif emoji == "😂":
            duration = 0.85
        else:
            duration = 0.7
        events.append({"time": float(word["start"]), "duration": duration, "emoji": emoji})
    for first, second in zip(words, words[1:]):
        if _safe_text(str(first.get("text", ""))).strip().lower() == "no" and _safe_text(str(second.get("text", ""))).strip().lower() == "way":
            events.append({"time": float(first["start"]), "duration": .7, "emoji": "😮"})
    return events


def write_ass(transcript: list[dict], start: float, end: float, destination: Path, action_events: list[dict] | None = None) -> list[dict]:
    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "\n[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Default,Arial,66,&H0045FFC7,&H00FFFFFF,&H00101010,&H88000000,-1,0,1,5,2,2,80,80,410,1",
        "\n[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    lines = list(header)
    clip_duration = end - start
    emoji_events = list(action_events or [])
    for segment in transcript:
        segment_start = max(start, float(segment["start"]))
        segment_end = min(end, float(segment["end"]))
        if segment_end <= segment_start:
            continue
        candidates = segment.get("words") or _fallback_words(segment)
        words = []
        for word in candidates:
            word_start = max(segment_start, float(word["start"]))
            word_end = min(segment_end, float(word["end"]))
            if word_end > word_start and str(word.get("text", "")).strip():
                words.append({"start": word_start, "end": word_end, "text": str(word["text"])})
        if not words:
            continue
        emoji_events.extend(_emoji_events_from_words(words))
        # Render one token per timed event. ASS karaoke normally displays the
        # whole phrase before its later words are spoken, so it is unsuitable
        # for the strict word-by-word timing requested here.
        for word in words:
            cue_start = word["start"]
            cue_end = min(segment_end, word["end"])
            caption = _safe_text(str(word.get("text", ""))).strip()
            if not caption or cue_end <= cue_start:
                continue
            local_start = cue_start - start
            local_end = min(clip_duration, cue_end - start)
            pop = r"{\fscx88\fscy88\t(0,80,\fscx100\fscy100)}"
            lines.append(f"Dialogue: 0,{_ass_time(local_start)},{_ass_time(local_end)},Default,,0,0,0,,{pop}{caption}")

    # Keep action/reaction emojis independent from speech captions. Return
    # timed events for PNG overlays because FFmpeg/libass builds often lack
    # dependable color-emoji font support.
    last_event_time = -EMOJI_MIN_SPACING_SECONDS
    rendered_events = []
    for event in sorted(emoji_events, key=lambda item: float(item.get("time", 0))):
        event_time = float(event.get("time", -1))
        emoji = _safe_text(str(event.get("emoji", "")))
        duration = max(.5, min(1.0, float(event.get("duration", .7))))
        if not emoji or event_time < start or event_time >= end or event_time - last_event_time < EMOJI_MIN_SPACING_SECONDS:
            continue
        local_start = event_time - start
        visible_duration = min(duration, clip_duration - local_start)
        if visible_duration <= 0:
            continue
        rendered_events.append({"time": local_start, "duration": visible_duration, "emoji": emoji})
        last_event_time = event_time
    destination.write_text("\n".join(lines), encoding="utf-8")
    return rendered_events


def render_emoji_overlay(emoji: str, destination: Path) -> None:
    """Render a small, high-contrast sticker so emoji visibility is font-safe."""
    image = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((20, 20, 236, 236), radius=76, fill=(18, 20, 28, 232), outline=(190, 242, 100, 255), width=7)
    font_candidates = [
        Path("C:/Windows/Fonts/seguiemj.ttf"),
        Path("/System/Library/Fonts/Apple Color Emoji.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"),
    ]
    font_path = next((path for path in font_candidates if path.is_file()), None)
    font = ImageFont.truetype(str(font_path), 118) if font_path else ImageFont.load_default()
    bounds = draw.textbbox((0, 0), emoji, font=font, embedded_color=font_path is not None)
    x = (256 - (bounds[2] - bounds[0])) / 2 - bounds[0]
    y = (256 - (bounds[3] - bounds[1])) / 2 - bounds[1] - 5
    try:
        draw.text((x, y), emoji, font=font, embedded_color=font_path is not None)
    except (TypeError, ValueError):
        draw.text((x, y), emoji, font=font, fill=(255, 255, 255, 255))
    image.save(destination)
