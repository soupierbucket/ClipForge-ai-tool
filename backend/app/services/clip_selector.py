import uuid

from app.models.schemas import Candidate, CandidateScores, MAX_SHORT_DURATION_SECONDS


ENGAGEMENT_WEIGHTS = {
    "hook": .18,
    "curiosity": .12,
    "emotional_intensity": .12,
    "novelty": .10,
    "entertainment": .12,
    "visual_potential": .10,
    "payoff": .12,
    "context": .06,
    "standalone": .05,
    "rewatch_potential": .03,
}


def build_candidates(raw: list[dict], transcript: list[dict], video_path, video_duration: float) -> list[Candidate]:
    from app.analyzers.audio_analyzer import analyze_audio
    from app.analyzers.video_analyzer import analyze_video

    candidates = []
    for item in raw:
        try:
            analysis_method = item.get("analysis_method", "ai")
            if analysis_method == "sample":
                start, end = float(item["start"]), float(item["end"])
            else:
                start, end = _align_to_transcript(item, transcript, video_duration)
            duration = end - start
            if start < 0 or end > video_duration or duration < 1 or duration > MAX_SHORT_DURATION_SECONDS:
                continue
            transcript_match = [segment for segment in transcript if segment["end"] > start and segment["start"] < end]
            if not transcript_match and analysis_method != "sample":
                continue

            source_scores = item["scores"]
            try:
                audio = analyze_audio(video_path, start, end)
                visual = analyze_video(video_path, start, end)
                media_evidence = [
                    f"Measured visual activity: {visual['motion']:.3f} average frame change with {visual['scene_changes']} scene-change signals.",
                    f"Measured audio activity: {round(audio['active_ratio'] * 100)}% active audio; {round(audio['peak_ratio'] * 100)}% high-intensity samples.",
                ]
                # Ollama is transcript-only; blend its visual-potential estimate
                # with measured motion/cuts without implying semantic vision.
                source_scores = dict(source_scores)
                source_scores["visual_potential"] = (
                    _score(source_scores.get("visual_potential")) * .6 + _score(visual["score"]) * .4
                )
            except Exception:
                media_evidence = ["Audio/video activity analysis was unavailable; the remaining transcript-based score dimensions were used."]

            scores = CandidateScores(**{key: _score(source_scores.get(key)) for key in ENGAGEMENT_WEIGHTS})
            engagement_score = round(sum(getattr(scores, key) * weight for key, weight in ENGAGEMENT_WEIGHTS.items()) * 10)
            excerpt = " ".join(segment["text"] for segment in transcript_match)[:700]
            if analysis_method == "sample":
                excerpt = excerpt or "No timestamped speech was detected in this video."
                evidence = ["This interval was selected from the source timeline. No story, visual event, or highlight is claimed.", *media_evidence]
            else:
                evidence = [
                    f"Transcript-supported section: {item['hook_summary']} → {item['context_summary']} → {item['payoff_summary']}",
                    *media_evidence,
                ]
            candidates.append(Candidate(
                id=uuid.uuid4().hex[:12],
                start=round(start, 2),
                end=round(end, 2),
                duration=round(duration, 2),
                hook_summary=item["hook_summary"],
                context_summary=item["context_summary"],
                payoff_summary=item["payoff_summary"],
                reason=item["reason"],
                transcript_excerpt=excerpt,
                evidence=evidence,
                scores=scores,
                engagement_potential_score=engagement_score,
                analysis_method=analysis_method,
            ))
        except (ValueError, TypeError, KeyError, IndexError):
            continue

    # AI candidates retain model rank. Timeline samples stay in chronological
    # order and are visibly labeled as samples, never as inferred highlights.
    if candidates and candidates[0].analysis_method == "ai":
        candidates[0].recommended = True
    return candidates[:15]


def _align_to_transcript(item: dict, transcript: list[dict], video_duration: float) -> tuple[float, float]:
    """Snap proposed timestamps to real Whisper segment boundaries."""
    start = float(item["start"])
    end = float(item["end"])
    starts = [float(segment["start"]) for segment in transcript]
    ends = [float(segment["end"]) for segment in transcript]
    start_boundary = min(starts, key=lambda value: abs(value - start))
    end_boundary = min(ends, key=lambda value: abs(value - end))
    if abs(start_boundary - start) <= 2.0:
        start = start_boundary
    if abs(end_boundary - end) <= 2.0:
        end = end_boundary
    return max(0.0, start), min(video_duration, end)


def _score(value) -> float:
    try:
        return max(0.0, min(10.0, float(value)))
    except (ValueError, TypeError):
        return 0.0
