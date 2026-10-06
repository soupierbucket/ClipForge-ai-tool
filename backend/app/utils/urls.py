import re
from urllib.parse import parse_qs, urlparse


class InvalidYouTubeURL(ValueError):
    pass


def normalize_youtube_url(url: str) -> str:
    try:
        parsed = urlparse(url.strip())
    except ValueError as exc:
        raise InvalidYouTubeURL("Enter a valid YouTube video URL.") from exc
    host = (parsed.hostname or "").lower().removeprefix("www.")
    if parsed.scheme not in {"http", "https"} or host not in {"youtube.com", "m.youtube.com", "youtu.be", "youtube-nocookie.com"}:
        raise InvalidYouTubeURL("Enter a valid YouTube video URL.")
    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
    elif parsed.path == "/watch":
        video_id = parse_qs(parsed.query).get("v", [""])[0]
    else:
        match = re.match(r"/(?:shorts|embed|live)/([^/?]+)", parsed.path)
        video_id = match.group(1) if match else ""
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise InvalidYouTubeURL("That URL does not appear to contain a valid YouTube video ID.")
    return f"https://www.youtube.com/watch?v={video_id}"
