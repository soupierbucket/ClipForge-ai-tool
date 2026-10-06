from typing import Literal
from pydantic import BaseModel, Field

MAX_SHORT_DURATION_SECONDS = 60


class VideoInfo(BaseModel):
    title: str
    channel: str
    duration: float
    thumbnail: str
    url: str


class AnalyzeRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)


class GenerateClipRequest(BaseModel):
    job_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    candidate_id: str = Field(pattern=r"^[a-f0-9]{12}$")


class CandidateScores(BaseModel):
    hook: float = Field(ge=0, le=10)
    curiosity: float = Field(ge=0, le=10)
    emotional_intensity: float = Field(ge=0, le=10)
    novelty: float = Field(ge=0, le=10)
    entertainment: float = Field(ge=0, le=10)
    visual_potential: float = Field(ge=0, le=10)
    payoff: float = Field(ge=0, le=10)
    context: float = Field(ge=0, le=10)
    standalone: float = Field(ge=0, le=10)
    rewatch_potential: float = Field(ge=0, le=10)


class Candidate(BaseModel):
    id: str
    start: float
    end: float
    duration: float = Field(ge=1, le=MAX_SHORT_DURATION_SECONDS)
    hook_summary: str
    context_summary: str
    payoff_summary: str
    reason: str
    transcript_excerpt: str
    evidence: list[str]
    scores: CandidateScores
    engagement_potential_score: int = Field(ge=0, le=100)
    analysis_method: Literal["ai", "sample"] = "ai"
    recommended: bool = False


class AnalysisResult(BaseModel):
    video: VideoInfo
    transcript: list[dict]
    candidates: list[Candidate]


class JobLog(BaseModel):
    timestamp: str
    level: Literal["info", "success", "warning", "error"] = "info"
    message: str


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    progress: int = Field(ge=0, le=100)
    stage: str
    logs: list[JobLog] = Field(default_factory=list)
    error: str | None = None
    result: AnalysisResult | None = None
    clip_url: str | None = None
