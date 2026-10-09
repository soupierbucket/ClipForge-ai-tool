import type { AnalysisResult, ClipEffect, JobStatus, VideoInfo } from '../types/video'

const apiBase = import.meta.env.VITE_API_BASE_URL ?? ''

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBase}${path}`, { ...init, headers: { 'Content-Type': 'application/json', ...init?.headers } })
  if (!response.ok) {
    const body = await response.json().catch(() => null) as { detail?: string } | null
    throw new Error(body?.detail ?? 'The request failed. Please try again.')
  }
  return response.json() as Promise<T>
}

export const api = {
  videoInfo: (url: string) => request<VideoInfo>(`/api/video-info?url=${encodeURIComponent(url)}`),
  analyze: (url: string) => request<{ job_id: string; status: string }>('/api/analyze', { method: 'POST', body: JSON.stringify({ url }) }),
  effectsPlan: (job_id: string, candidate_id: string) => request<{ duration: number; effects: ClipEffect[] }>('/api/effects-plan', { method: 'POST', body: JSON.stringify({ job_id, candidate_id }) }),
  generateClip: (job_id: string, candidate_id: string, effects: ClipEffect[]) => request<{ job_id: string; status: string }>('/api/generate-clip', { method: 'POST', body: JSON.stringify({ job_id, candidate_id, effects }) }),
  job: (jobId: string) => request<JobStatus>(`/api/jobs/${encodeURIComponent(jobId)}`),
  mediaUrl: (path: string) => `${apiBase}${path}`,
}

export type { AnalysisResult }
