from pathlib import Path
import shutil

from app.config import settings
from app.models.schemas import AnalysisResult, VideoInfo
from app.services.clip_generator import generate_short
from app.services.clip_selector import build_candidates
from app.services.job_store import jobs
from app.services.llm_analysis_service import AnalysisProviderError, build_timeline_samples, get_candidate_analyzer
from app.services.transcription_service import transcribe_video
from app.services.video_downloader import download_video, get_video_info


def run_analysis(job_id: str, url: str) -> None:
    work_dir = settings.jobs_dir / job_id
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        jobs.update(job_id, status="processing", progress=4, stage="Loading video information", work_dir=str(work_dir))
        jobs.log(job_id, "Checking the YouTube URL and retrieving video metadata.")
        video_data = get_video_info(url)
        jobs.log(job_id, f"Video found: {video_data['title']} · {video_data['channel']} · {video_data['duration']:.0f} seconds.", "success")
        jobs.log(job_id, f"Checking the configured maximum video length ({settings.max_video_duration_seconds // 60} minutes).")
        if video_data["duration"] > settings.max_video_duration_seconds:
            raise RuntimeError(f"Videos longer than {settings.max_video_duration_seconds // 3600} hours are not supported.")
        jobs.update(job_id, video=video_data, progress=10, stage="Downloading video")
        jobs.log(job_id, "Downloading the source video with yt-dlp; FFmpeg may merge separate audio and video streams.")
        source = download_video(video_data["url"], work_dir)
        jobs.log(job_id, f"Source download complete ({source.stat().st_size / (1024 * 1024):.1f} MB).", "success")
        jobs.update(job_id, progress=38, stage="Transcribing audio")
        jobs.log(job_id, "Checking the source audio and preparing local speech transcription.")
        transcript = transcribe_video(source, on_progress=lambda message: jobs.log(job_id, message))
        jobs.log(job_id, f"Transcription complete: {len(transcript)} timestamped speech segments.", "success")
        jobs.update(job_id, progress=63, stage="Analyzing candidate moments")
        if transcript:
            jobs.log(job_id, f"Asking {settings.llm_provider} model '{settings.llm_model}' to select only grounded, timestamped transcript sections.")
            try:
                raw_candidates = get_candidate_analyzer().analyze(
                    transcript,
                    video_data["duration"],
                    on_progress=lambda message: jobs.log(job_id, message),
                )
            except AnalysisProviderError as exc:
                jobs.log(job_id, f"AI did not return a valid grounded clip ({str(exc)[:160]}). Using clearly labeled source timeline samples instead.", "warning")
                raw_candidates = build_timeline_samples(transcript, video_data["duration"])
        else:
            jobs.log(job_id, "No speech transcript is available. Creating clearly labeled source timeline samples without claiming they are highlights.", "warning")
            raw_candidates = build_timeline_samples([], video_data["duration"])
        if not raw_candidates:
            raise RuntimeError("This source is too short to create a clip of at least one second.")
        jobs.log(job_id, f"Prepared {len(raw_candidates)} clip option(s).", "success")
        jobs.update(job_id, progress=78, stage="Scoring audio and visual activity")
        jobs.log(job_id, "Checking source timestamps and measuring available audio activity and visual motion for each clip option.")
        candidates = build_candidates(raw_candidates, transcript, source, video_data["duration"])
        if not candidates:
            jobs.log(job_id, "No candidates passed the duration, timestamp, and transcript checks.", "warning")
            raise RuntimeError("No valid clip ranges could be prepared from this source video.")
        for index, candidate in enumerate(candidates, 1):
            scores = candidate.scores
            jobs.log(job_id, f"Candidate {index} scored {candidate.engagement_potential_score}/100 — hook {scores.hook:.1f}, curiosity {scores.curiosity:.1f}, emotion {scores.emotional_intensity:.1f}, novelty {scores.novelty:.1f}, entertainment {scores.entertainment:.1f}, visual potential {scores.visual_potential:.1f}, payoff {scores.payoff:.1f}, context {scores.context:.1f}, standalone {scores.standalone:.1f}, rewatch {scores.rewatch_potential:.1f}.", "success")
        result = AnalysisResult(video=VideoInfo(**video_data), transcript=transcript, candidates=candidates)
        jobs.update(job_id, status="completed", progress=100, stage="Analysis complete", result=result.model_dump())
        jobs.log(job_id, f"Analysis complete. Ranked {len(candidates)} moments; the top score is a comparison of signals, not a view prediction.", "success")
    except Exception as exc:
        jobs.log(job_id, f"Analysis stopped: {str(exc)[:300]}", "error")
        jobs.update(job_id, status="failed", progress=100, stage="Analysis failed", error=str(exc)[:500])
        shutil.rmtree(work_dir, ignore_errors=True)
        jobs.update(job_id, work_dir=None)


def run_generation(job_id: str, source_job_id: str, candidate_id: str) -> None:
    record = jobs.get(source_job_id)
    if not record or record.get("status") != "completed" or not record.get("result"):
        jobs.update(job_id, status="failed", progress=100, stage="Generation failed", error="The analysis job is no longer available.")
        return
    try:
        result = record["result"]
        candidate = next(item for item in result["candidates"] if item["id"] == candidate_id)
        work_dir = Path(record["work_dir"])
        source_candidates = [file for file in work_dir.glob("source.*") if file.is_file() and file.suffix not in {".part", ".ytdl"}]
        if not source_candidates:
            raise RuntimeError("The source video expired. Analyze the video again before rendering a Short.")
        settings.clips_dir.mkdir(parents=True, exist_ok=True)
        clip_id = job_id
        output = settings.clips_dir / f"{clip_id}.mp4"
        jobs.update(job_id, status="processing", progress=15, stage="Preparing captions", work_dir=str(work_dir))
        jobs.log(job_id, f"Preparing captions for the selected {candidate['end'] - candidate['start']:.1f}-second moment.")
        jobs.update(job_id, progress=40, stage="Rendering vertical video")
        jobs.log(job_id, "Rendering a 9:16 clip with speech-timed captions, brief emoji cues, varied face-aware punch-ins, and occasional character close-ups.")
        framing, emoji_count = generate_short(Path(source_candidates[0]), result["transcript"], candidate["start"], candidate["end"], output, work_dir)
        if emoji_count:
            jobs.log(job_id, f"Added {emoji_count} timed emoji overlay(s) for relevant dialogue or action cues.")
        else:
            jobs.log(job_id, "No strong dialogue or action cue matched the emoji rules, so no emoji overlays were added.")
        if framing == "full-screen":
            jobs.log(job_id, "Added one or more brief full-frame portrait crops around a stable, tracked character, alongside shorter punch-ins.")
        elif framing == "face-tracked":
            jobs.log(job_id, "Using brief face-tracked punch-ins (18% and up to 30%) centered on detected characters.")
        elif framing == "group-framed":
            jobs.log(job_id, "Using more frequent brief 18% group punch-ins; no stable single-face close-up was selected.")
        else:
            jobs.log(job_id, "No clear face was detected, so the clip keeps the full fitted frame without a center-crop zoom.")
        jobs.update(job_id, status="completed", progress=100, stage="Short ready", clip_url=f"/api/clips/{clip_id}", clip_path=str(output))
        jobs.log(job_id, "Clip render complete and ready to preview or download.", "success")
    except Exception as exc:
        if 'output' in locals():
            output.unlink(missing_ok=True)
        jobs.log(job_id, f"Clip generation stopped: {str(exc)[:300]}", "error")
        jobs.update(job_id, status="failed", progress=100, stage="Generation failed", error=str(exc)[:500])
