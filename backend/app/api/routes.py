from concurrent.futures import ThreadPoolExecutor
import logging
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pathlib import Path

from app.config import settings
from app.models.schemas import AnalyzeRequest, JobStatus, VideoInfo
from app.services.effect_planner import EffectsPlanRequest, GenerateClipRequestWithEffects, build_effects_plan, tag_moments_with_local_llm
from app.analyzers.event_analyzer import detect_action_events
from app.services.job_store import jobs
from app.services.llm_analysis_service import AnalysisProviderError, get_candidate_analyzer
from app.services.pipeline import run_analysis, run_generation
from app.services.video_downloader import DownloadError, get_video_info
from app.utils.urls import InvalidYouTubeURL

router = APIRouter(prefix="/api")
executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="clipforge-job")
logger = logging.getLogger(__name__)


@router.get("/video-info", response_model=VideoInfo)
def video_info(url: str = Query(min_length=1, max_length=2048)):
    try:
        return get_video_info(url)
    except (DownloadError, InvalidYouTubeURL) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/analyze", status_code=202)
def analyze_video(request: AnalyzeRequest):
    try:
        get_candidate_analyzer()
    except AnalysisProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    job_id = jobs.create("analysis", url=request.url)
    logger.info("Analysis job accepted: job_id=%s provider=%s model=%s", job_id, settings.llm_provider, settings.llm_model)
    executor.submit(run_analysis, job_id, request.url)
    return {"job_id": job_id, "status": "queued"}


@router.post("/effects-plan")
def effects_plan(request: EffectsPlanRequest):
    record = jobs.get(request.job_id)
    if not record or record.get("status") != "completed" or not record.get("result"):
        raise HTTPException(status_code=404, detail="Analysis is unavailable. Analyze the video again.")
    result = record["result"]
    candidate = next((item for item in result.get("candidates", []) if item.get("id") == request.candidate_id), None)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found for this analysis.")
    source_candidates = [file for file in Path(record["work_dir"]).glob("source.*") if file.is_file() and file.suffix not in {".part", ".ytdl"}]
    signals = []
    if source_candidates:
        try:
            signals = detect_action_events(source_candidates[0], candidate["start"], candidate["end"])
        except Exception:
            logger.exception("Effects plan signal detection failed; using transcript cues only.")
    duration = candidate["end"] - candidate["start"]
    llm_events = tag_moments_with_local_llm(result.get("transcript", []), candidate["start"], candidate["end"], signals)
    effects = build_effects_plan(result.get("transcript", []), candidate["start"], candidate["end"], signals, settings.effects_intensity, llm_events)
    for effect in effects:
        if not settings.effects_sfx_enabled: effect["sfx"] = None
        if not settings.effects_filters_enabled: effect["filter"] = "none"
        if not settings.effects_overlays_enabled: effect["overlay"] = ""
    return {"duration": duration, "effects": effects}

@router.post("/generate-clip", status_code=202)
def generate_clip(request: GenerateClipRequestWithEffects):
    record = jobs.get(request.job_id)
    if not record:
        logger.warning("Clip request rejected: analysis job is missing or expired (analysis_job_id=%s candidate_id=%s)", request.job_id, request.candidate_id)
        raise HTTPException(status_code=404, detail="This analysis is no longer available. The backend may have restarted or the job expired. Analyze the video again, then generate the clip.")
    status = record.get("status")
    if status != "completed":
        logger.warning("Clip request rejected: analysis is not complete (analysis_job_id=%s status=%s stage=%s)", request.job_id, status, record.get("stage"))
        if status == "failed":
            detail = f"The analysis failed: {record.get('error') or 'unknown error'}. Start a new analysis before generating a clip."
        else:
            detail = f"The analysis is still {status or 'unavailable'} ({record.get('stage') or 'no stage reported'}). Wait for it to finish, then try again."
        raise HTTPException(status_code=409, detail=detail)
    result = record.get("result")
    if not result or not isinstance(result.get("candidates"), list):
        logger.error("Clip request rejected: completed analysis has no candidate result (analysis_job_id=%s)", request.job_id)
        raise HTTPException(status_code=409, detail="The analysis completed without a usable candidate list. Analyze the video again before generating a clip.")
    if not any(item.get("id") == request.candidate_id for item in result["candidates"]):
        logger.warning("Clip request rejected: candidate does not belong to analysis (analysis_job_id=%s candidate_id=%s candidate_count=%d)", request.job_id, request.candidate_id, len(result["candidates"]))
        raise HTTPException(status_code=404, detail="Candidate not found for this analysis.")
    job_id = jobs.create("generation", source_job_id=request.job_id, candidate_id=request.candidate_id)
    logger.info("Clip generation accepted: job_id=%s analysis_job_id=%s candidate_id=%s", job_id, request.job_id, request.candidate_id)
    executor.submit(run_generation, job_id, request.job_id, request.candidate_id, [item.model_dump() for item in request.effects])
    return {"job_id": job_id, "status": "queued"}


@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    if len(job_id) != 32 or any(char not in "0123456789abcdef" for char in job_id):
        raise HTTPException(status_code=404, detail="Job not found.")
    record = jobs.get(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found or it expired.")
    return {key: record.get(key) for key in ("job_id", "status", "progress", "stage", "logs", "error", "result", "clip_url")}


@router.get("/clips/{clip_id}")
def get_clip(clip_id: str):
    if len(clip_id) != 32 or any(char not in "0123456789abcdef" for char in clip_id):
        raise HTTPException(status_code=404, detail="Clip not found.")
    path = settings.clips_dir / f"{clip_id}.mp4"
    if not path.is_file() or path.resolve().parent != settings.clips_dir:
        raise HTTPException(status_code=404, detail="Clip not found.")
    return FileResponse(path, media_type="video/mp4")
