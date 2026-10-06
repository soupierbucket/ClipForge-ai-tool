import json
import logging
from collections.abc import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from openai import OpenAI

from app.config import settings
from app.prompts.clip_analysis import (
    CANDIDATE_PROMPT as STRUCTURED_CANDIDATE_PROMPT,
    RANKING_PROMPT as STRUCTURED_RANKING_PROMPT,
)

logger = logging.getLogger(__name__)


class AnalysisProviderError(RuntimeError):
    pass


class CandidateAnalyzer:
    def analyze(self, transcript: list[dict], video_duration: float, on_progress: Callable[[str], None] | None = None) -> list[dict]:
        raise NotImplementedError


class OllamaCandidateAnalyzer(CandidateAnalyzer):
    def analyze(self, transcript: list[dict], video_duration: float, on_progress: Callable[[str], None] | None = None) -> list[dict]:
        if on_progress:
            on_progress("Splitting the timestamped transcript into overlapping story windows across the full video.")
        windows = _select_transcript_windows(transcript, video_duration)
        if not windows:
            raise AnalysisProviderError("The transcript did not contain enough timestamped material to analyze clip moments.")

        target_count = candidate_target(video_duration)
        logger.info("Preparing %d transcript windows for %d candidate recommendations.", len(windows), target_count)
        if on_progress:
            on_progress(f"Prepared {len(windows)} transcript windows; sending bounded sections to local Ollama model '{settings.llm_model}'.")
        user_message = _candidate_analysis_message(windows, video_duration, target_count)
        payload = self._chat_json(CANDIDATE_PROMPT, user_message)
        candidates = _normalize_candidates(payload, transcript, video_duration)
        if on_progress:
            on_progress(f"Ollama proposed {len(payload.get('candidates', []))} moments; {len(candidates)} passed segment-boundary, duration, and duplicate checks.")
        if not candidates:
            raise AnalysisProviderError("Ollama did not return any valid complete moments. Try another video or analyze it again.")

        # A compact second pass compares complete story arcs without resending
        # the source transcript, and places the strongest standalone moment first.
        logger.info("Asking configured Ollama model %s to rank %d validated candidates.", settings.llm_model, len(candidates))
        if on_progress:
            on_progress(f"Asking Ollama to compare {len(candidates)} complete candidate arcs and choose the strongest standalone moment.")
        try:
            ranking = self._chat_json(RANKING_PROMPT, _ranking_message(candidates, target_count))
            candidates = _apply_ranking(candidates, ranking)
            if on_progress:
                on_progress("Final Ollama ranking complete; the strongest complete standalone moment is ranked first.")
        except AnalysisProviderError:
            logger.warning("Ollama final ranking was invalid; using the validated candidates' weighted scores.")
            candidates.sort(key=lambda item: item["engagement_score"], reverse=True)
            if on_progress:
                on_progress("Final ranking response was invalid; ordered validated candidates by the weighted score instead.")
        return candidates[:target_count]

    @staticmethod
    def _chat_json(system_prompt: str, user_message: str) -> dict:
        body = json.dumps({
            "model": settings.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.15},
        }).encode("utf-8")
        request = Request(
            f"{settings.ollama_base_url.rstrip('/')}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=600) as response:
                response_data = json.loads(response.read().decode("utf-8"))
            content = response_data["message"]["content"]
            payload = json.loads(content)
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object")
            return payload
        except (URLError, HTTPError, TimeoutError) as exc:
            raise AnalysisProviderError("Could not get a response from local Ollama. Start Ollama and check the configured model.") from exc
        except (KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
            raise AnalysisProviderError("Ollama returned malformed JSON. Try the analysis again; no clip timestamps were accepted.") from exc


class OpenAICandidateAnalyzer(CandidateAnalyzer):
    """Keep the existing optional provider compatible with the staged analyzer."""

    def analyze(self, transcript: list[dict], video_duration: float, on_progress: Callable[[str], None] | None = None) -> list[dict]:
        if not settings.llm_api_key:
            raise AnalysisProviderError("The OpenAI provider needs LLM_API_KEY in backend/.env.")
        windows = _select_transcript_windows(transcript, video_duration)
        if not windows:
            raise AnalysisProviderError("The transcript did not contain enough timestamped material to analyze clip moments.")
        user_message = _candidate_analysis_message(windows, video_duration, candidate_target(video_duration))
        try:
            payload = self._chat_json(CANDIDATE_PROMPT, user_message)
            candidates = _normalize_candidates(payload, transcript, video_duration)
            if not candidates:
                raise AnalysisProviderError("The model did not return any valid complete moments.")
            ranking = self._chat_json(RANKING_PROMPT, _ranking_message(candidates, candidate_target(video_duration)))
            return _apply_ranking(candidates, ranking)[:candidate_target(video_duration)]
        except AnalysisProviderError:
            raise
        except Exception as exc:
            raise AnalysisProviderError("The configured analysis provider returned an invalid response.") from exc

    @staticmethod
    def _chat_json(system_prompt: str, user_message: str) -> dict:
        try:
            response = OpenAI(api_key=settings.llm_api_key).chat.completions.create(
                model=settings.llm_model,
                temperature=0.15,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            payload = json.loads(response.choices[0].message.content or "{}")
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object")
            return payload
        except Exception as exc:
            raise AnalysisProviderError("The configured analysis provider returned malformed JSON or could not be reached.") from exc


CANDIDATE_PROMPT = """Select clip boundaries from the supplied timestamped speech transcript. It is the only source of story facts. Do not invent words, events, visuals, characters, reactions, or timestamps. Use only supplied segment IDs and choose contiguous ranges. Aim for 20–55 seconds when possible; any valid complete range from 1–60 seconds is acceptable. Prefer a clear passage with a beginning and ending. If there is no dramatic story, choose ordinary useful, funny, informative, or complete speech rather than returning nothing. Return up to the requested number of distinct candidates; avoid near-duplicates. You cannot see the frames: make no visual claims. Choose ranges only; the application will display the actual transcript words for those ranges. Scores are rough transcript-based estimates, not facts or view predictions; keep visual_potential conservative. Return only this JSON shape: {\"candidates\":[{\"start_segment\":integer,\"end_segment\":integer,\"scores\":{\"hook\":number,\"curiosity\":number,\"emotional_intensity\":number,\"novelty\":number,\"entertainment\":number,\"visual_potential\":number,\"payoff\":number,\"context\":number,\"standalone\":number,\"rewatch_potential\":number}}]}. IDs are inclusive. Scores must be numeric from 0 to 10. Never output markdown or text outside the JSON."""

RANKING_PROMPT = """You are selecting the strongest complete standalone short-form story, not merely the most dramatic sentence. Compare the provided validated candidates using HOOK → DEVELOPMENT → PAYOFF, clarity without source context, satisfying ending, pacing implied by the transcript, and entertainment value. Favor a complete 20–55 second story over a weak longer candidate; never claim to predict views. Return strict JSON only: {\"ranked_ids\":[string],\"recommended_id\":string}. Include every supplied candidate ID exactly once, strongest first, and choose recommended_id as the strongest complete moment."""

# Prompt text is maintained in app/prompts/clip_analysis.py. These assignments
# keep the provider code compatible while making the structured templates active.
CANDIDATE_PROMPT = STRUCTURED_CANDIDATE_PROMPT
RANKING_PROMPT = STRUCTURED_RANKING_PROMPT

SCORE_KEYS = (
    "hook", "curiosity", "emotional_intensity", "novelty", "entertainment",
    "visual_potential", "payoff", "context", "standalone", "rewatch_potential",
)
SCORE_WEIGHTS = {
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


def candidate_target(video_duration: float) -> int:
    # Keep the result focused and the review UI compact across source lengths.
    return 5


def build_timeline_samples(transcript: list[dict], video_duration: float) -> list[dict]:
    """Create honest, evenly spaced clip options when speech/model output is unavailable."""
    if video_duration < 1:
        return []
    count = min(5, max(1, int(video_duration // 15)))
    clip_duration = min(30.0, video_duration / count)
    starts = [
        (video_duration - clip_duration) * index / (count - 1) if count > 1 else (video_duration - clip_duration) / 2
        for index in range(count)
    ]
    samples = []
    for index, raw_start in enumerate(starts):
        start = round(max(0.0, raw_start), 2)
        end = round(min(video_duration, raw_start + clip_duration), 2)
        if end - start < 1:
            continue
        excerpt = " ".join(
            str(segment.get("text", "")).strip()
            for segment in transcript
            if float(segment.get("end", 0)) > start and float(segment.get("start", 0)) < end
        ).strip()[:500]
        stamp = lambda value: f"{int(value // 60):02d}:{int(value % 60):02d}"
        samples.append({
            "start": start,
            "end": end,
            "analysis_method": "sample",
            "hook_summary": f"Source interval {stamp(start)}–{stamp(end)}",
            "context_summary": f"Transcript excerpt: {excerpt[:180]}" if excerpt else "No timestamped speech was available for this interval.",
            "payoff_summary": "Preview this source segment; no highlight was inferred.",
            "reason": "Timeline sample selected directly from the video duration. It is a clip option, not an AI-identified highlight.",
            "scores": {key: 0.0 for key in SCORE_KEYS},
        })
    return samples


def _select_transcript_windows(transcript: list[dict], video_duration: float) -> list[dict]:
    """Create bounded, overlapping logical sections across the whole video."""
    if not transcript:
        return []
    width, stride = 100.0, 80.0
    starts = list(range(0, max(1, int(video_duration)), int(stride)))
    sections = []
    for section_index, start in enumerate(starts):
        end = min(video_duration, start + width)
        segments = [
            {"id": index, "start": float(item["start"]), "end": float(item["end"]), "text": str(item["text"])[:500]}
            for index, item in enumerate(transcript)
            if float(item["end"]) > start and float(item["start"]) < end
        ]
        if segments:
            sections.append({"id": section_index, "start": start, "end": end, "segments": segments, "priority": _window_priority(segments)})

    if len(sections) <= 18:
        return sections

    # Keep timeline coverage while prioritizing expressive or story-rich text.
    selected = []
    bin_count = 12
    for bin_index in range(bin_count):
        low = bin_index * video_duration / bin_count
        high = (bin_index + 1) * video_duration / bin_count
        choices = [section for section in sections if low <= section["start"] < high]
        if choices:
            selected.append(max(choices, key=lambda item: item["priority"]))
    selected_ids = {item["id"] for item in selected}
    remaining = sorted((item for item in sections if item["id"] not in selected_ids), key=lambda item: item["priority"], reverse=True)
    selected.extend(remaining[:18 - len(selected)])
    return sorted(selected, key=lambda item: item["start"])


def _window_priority(segments: list[dict]) -> float:
    text = " ".join(item["text"].lower() for item in segments)
    signals = (
        "but then", "suddenly", "wait", "what if", "how did", "why did", "no way", "i can't believe",
        "i cannot believe", "oh my god", "never expected", "turns out", "finally", "the problem",
        "the secret", "actually", "however", "because", "so i", "and then", "boss", "clutch", "killed",
        "died", "rage", "fail", "lost", "won", "victory", "laugh", "funny", "insane", "explosion",
    )
    signal_count = sum(text.count(signal) for signal in signals)
    punctuation = sum(item["text"].count("?") + item["text"].count("!") for item in segments)
    return signal_count * 2 + punctuation * .6 + min(len(segments), 20) * .05


def _candidate_analysis_message(windows: list[dict], video_duration: float, target_count: int) -> str:
    payload = []
    for window in windows:
        segments = "\n".join(
            f"[{item['id']}] {item['start']:.2f}-{item['end']:.2f}: {item['text']}"
            for item in window["segments"]
        )
        payload.append(f"SECTION {window['id']} ({window['start']:.1f}-{window['end']:.1f}s)\n{segments}")
    return (
        f"Source duration: {video_duration:.2f} seconds. Return up to {target_count} distinct clip ranges. "
        "For ordinary speech, select the clearest complete passages; do not require a dramatic story. Use only the listed segment IDs.\n\n"
        + "\n\n".join(payload)
    )


def _normalize_candidates(payload: dict, transcript: list[dict], video_duration: float) -> list[dict]:
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        raise AnalysisProviderError("The model response did not include a valid candidates array.")
    normalized = []
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            continue
        try:
            start_index = int(raw["start_segment"])
            end_index = int(raw["end_segment"])
            if not 0 <= start_index <= end_index < len(transcript):
                continue
            start = float(transcript[start_index]["start"])
            end = float(transcript[end_index]["end"])
            duration = end - start
            if start < 0 or end > video_duration or duration < 1 or duration > 60:
                continue
            scores_data = raw["scores"]
            if not isinstance(scores_data, dict):
                continue
            scores = {key: max(0.0, min(10.0, float(scores_data[key]))) for key in SCORE_KEYS}
            selected_segments = transcript[start_index:end_index + 1]
            first_text = str(selected_segments[0].get("text", "")).strip()
            middle_text = str(selected_segments[len(selected_segments) // 2].get("text", "")).strip()
            last_text = str(selected_segments[-1].get("text", "")).strip()
            summaries = {
                "hook_summary": first_text[:240] or "Transcript begins here.",
                "context_summary": middle_text[:240] or "See the timestamped transcript excerpt.",
                "payoff_summary": last_text[:240] or "Transcript ends here.",
                "reason": "Selected timestamp range from the transcript. The displayed words are the source for this suggestion; automatic transcription may contain errors.",
            }
            candidate = {
                "llm_id": f"c{len(normalized)}",
                "start": round(start, 2),
                "end": round(end, 2),
                "duration": round(duration, 2),
                "scores": scores,
                "engagement_score": round(sum(scores[key] * weight for key, weight in SCORE_WEIGHTS.items()) * 10),
                **summaries,
            }
            if any(_interval_overlap(candidate, item) >= .8 for item in normalized):
                continue
            normalized.append(candidate)
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    return normalized


def _ranking_message(candidates: list[dict], target_count: int) -> str:
    compact = [{key: item[key] for key in ("llm_id", "start", "end", "duration", "hook_summary", "context_summary", "payoff_summary", "reason", "scores", "engagement_score")} for item in candidates]
    return f"Requested recommendation count: {target_count}. Rank these complete candidate stories:\n{json.dumps(compact, ensure_ascii=False)}"


def _apply_ranking(candidates: list[dict], ranking: dict) -> list[dict]:
    ids = [item["llm_id"] for item in candidates]
    ranked_ids = ranking.get("ranked_ids")
    if not isinstance(ranked_ids, list) or set(ranked_ids) != set(ids) or len(ranked_ids) != len(ids):
        raise AnalysisProviderError("Ollama's ranking did not match the validated candidates.")
    if ranking.get("recommended_id") not in ids:
        raise AnalysisProviderError("Ollama's recommended candidate was not in the validated list.")
    by_id = {item["llm_id"]: item for item in candidates}
    ranked = [by_id[item_id] for item_id in ranked_ids]
    # The model must recommend its highest ranked moment; reconcile malformed
    # ordering instead of trusting a mismatched ID.
    recommended_id = ranking["recommended_id"]
    if ranked[0]["llm_id"] != recommended_id:
        ranked.remove(by_id[recommended_id])
        ranked.insert(0, by_id[recommended_id])
    return ranked


def _interval_overlap(first: dict, second: dict) -> float:
    intersection = max(0.0, min(first["end"], second["end"]) - max(first["start"], second["start"]))
    shorter = min(first["duration"], second["duration"])
    return intersection / shorter if shorter > 0 else 0.0


def get_candidate_analyzer() -> CandidateAnalyzer:
    provider = settings.llm_provider.lower()
    if provider == "ollama":
        return OllamaCandidateAnalyzer()
    if provider == "openai":
        if not settings.llm_api_key:
            raise AnalysisProviderError("The OpenAI provider needs LLM_API_KEY in backend/.env.")
        return OpenAICandidateAnalyzer()
    raise AnalysisProviderError(f"Unsupported LLM_PROVIDER '{settings.llm_provider}'.")
