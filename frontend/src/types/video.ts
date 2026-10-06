export interface VideoInfo {
  title: string
  channel: string
  duration: number
  thumbnail: string
  url: string
}

export interface CandidateScores {
  hook: number
  curiosity: number
  emotional_intensity: number
  novelty: number
  entertainment: number
  visual_potential: number
  payoff: number
  context: number
  standalone: number
  rewatch_potential: number
}

export interface Candidate {
  id: string
  start: number
  end: number
  duration: number
  hook_summary: string
  context_summary: string
  payoff_summary: string
  reason: string
  transcript_excerpt: string
  evidence: string[]
  scores: CandidateScores
  engagement_potential_score: number
  analysis_method: 'ai' | 'sample'
  recommended: boolean
}

export interface AnalysisResult {
  video: VideoInfo
  transcript: { start: number; end: number; text: string }[]
  candidates: Candidate[]
}

export interface JobLog {
  timestamp: string
  level: 'info' | 'success' | 'warning' | 'error'
  message: string
}

export interface JobStatus {
  job_id: string
  status: 'queued' | 'processing' | 'completed' | 'failed'
  progress: number
  stage: string
  logs: JobLog[]
  error: string | null
  result: AnalysisResult | null
  clip_url: string | null
}
