import { useMemo, useState } from 'react'
import { Film, Scissors, Sparkles, WandSparkles } from 'lucide-react'
import { CandidateList } from '../components/CandidateList'
import { ClipResult } from '../components/ClipResult'
import { ErrorMessage } from '../components/ErrorMessage'
import { JobProgress } from '../components/JobProgress'
import { JobActivity } from '../components/JobActivity'
import { LoadingState } from '../components/LoadingState'
import { UrlInput } from '../components/UrlInput'
import { VideoInfo } from '../components/VideoInfo'
import { VideoPlayer } from '../components/VideoPlayer'
import { useJobPolling } from '../hooks/useJobPolling'
import { api } from '../services/api'
import type { Candidate, JobLog, VideoInfo as VideoInfoType } from '../types/video'
import { formatTime } from '../utils'

export default function Home() {
  const [video, setVideo] = useState<VideoInfoType | null>(null)
  const [analysisJobId, setAnalysisJobId] = useState('')
  const [generationJobId, setGenerationJobId] = useState('')
  const [selectedId, setSelectedId] = useState('')
  const [starting, setStarting] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [error, setError] = useState('')
  const [generationRequestLogs, setGenerationRequestLogs] = useState<JobLog[]>([])
  const { job: analysisJob, error: analysisPollError } = useJobPolling(analysisJobId)
  const { job: generationJob, error: generationPollError } = useJobPolling(generationJobId)
  const result = analysisJob?.result ?? null
  const selected = useMemo<Candidate | undefined>(() => result?.candidates.find(candidate => candidate.id === selectedId) ?? result?.candidates[0], [result, selectedId])
  const isRendering = generating || generationJob?.status === 'queued' || generationJob?.status === 'processing'
  const clipUrl = generationJob?.clip_url ? api.mediaUrl(generationJob.clip_url) : ''

  const analyze = async (url: string) => {
    setStarting(true); setError(''); setVideo(null); setAnalysisJobId(''); setGenerationJobId(''); setSelectedId(''); setGenerationRequestLogs([])
    try {
      const info = await api.videoInfo(url)
      setVideo(info)
      const job = await api.analyze(info.url)
      setAnalysisJobId(job.job_id)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unable to start video analysis.')
    } finally { setStarting(false) }
  }

  const generate = async () => {
    if (!analysisJobId || !selected) return
    setGenerating(true); setError(''); setGenerationJobId('')
    setGenerationRequestLogs([{ timestamp: new Date().toISOString(), level: 'info', message: `Requesting a clip for candidate ${selected.id} from analysis ${analysisJobId.slice(0, 8)}…` }])
    try {
      const job = await api.generateClip(analysisJobId, selected.id)
      setGenerationJobId(job.job_id)
      setGenerationRequestLogs(previous => [...previous, { timestamp: new Date().toISOString(), level: 'success', message: `Backend accepted the request as generation job ${job.job_id.slice(0, 8)}.` }])
    } catch (e) {
      const message = e instanceof Error ? e.message : 'Unable to render the Short.'
      setError(message)
      setGenerationRequestLogs(previous => [...previous, { timestamp: new Date().toISOString(), level: 'error', message: `Backend rejected the generation request: ${message}` }])
    } finally { setGenerating(false) }
  }

  const reset = () => { setVideo(null); setAnalysisJobId(''); setGenerationJobId(''); setSelectedId(''); setError(''); setGenerationRequestLogs([]) }

  return <div className="min-h-screen bg-ink text-white"><header className="border-b border-white/[.06]"><div className="mx-auto flex max-w-6xl items-center justify-between px-5 py-5 sm:px-8"><a href="#" onClick={reset} className="flex items-center gap-2.5"><span className="grid h-9 w-9 place-items-center rounded-xl bg-lime-300 text-zinc-950"><Scissors className="h-5 w-5"/></span><span className="text-lg font-bold tracking-tight">ClipForge</span></a><div className="hidden items-center gap-2 rounded-full border border-white/[.08] px-3 py-1.5 text-xs text-zinc-500 sm:flex"><span className="h-1.5 w-1.5 rounded-full bg-lime-300"/>AI-assisted Shorts studio</div></div></header>
    <main className="mx-auto max-w-4xl px-5 pb-20 pt-12 sm:px-8 sm:pt-16"><div className="mb-10 text-center"><div className="mx-auto mb-5 flex h-12 w-12 items-center justify-center rounded-2xl border border-lime-300/20 bg-lime-300/[.08] text-lime-300"><Sparkles className="h-5 w-5"/></div><h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">Turn long videos <span className="text-lime-300">into Shorts.</span></h1><p className="mt-3 text-sm text-zinc-500 sm:text-base">Find the moments worth sharing, then make them yours.</p></div>
      <div className="space-y-5"><UrlInput loading={starting} onSubmit={analyze}/><ErrorMessage message={error || analysisPollError || generationPollError || analysisJob?.error || generationJob?.error || ''}/>{generationRequestLogs.length > 0 && !generationJob && <div className="rounded-2xl border border-white/[.08] bg-panel px-5"><JobActivity logs={generationRequestLogs}/></div>}{starting && <div className="flex justify-center py-4"><LoadingState label="Loading video details and starting analysis…"/></div>}{video && <div className="rounded-2xl border border-white/[.08] bg-panel p-4 sm:p-5"><VideoInfo video={video}/><div className="mt-5"><VideoPlayer src={video.url} poster={video.thumbnail}/></div></div>}
        {analysisJob && <JobProgress job={analysisJob}/>}
        {result && selected && <div className="space-y-5"><section className="space-y-4 rounded-2xl border border-white/[.08] bg-panel p-5 sm:p-6"><div className="flex items-start justify-between gap-4"><div><div className="flex items-center gap-2 text-lime-300"><WandSparkles className="h-4 w-4"/><span className="text-xs font-semibold uppercase tracking-[.13em]">{selected.analysis_method === 'sample' ? 'Video samples' : 'AI recommendation'}</span></div><h2 className="mt-2 text-lg font-semibold text-white">Choose a moment to turn into a Short</h2></div><span className="rounded-lg bg-white/[.05] px-3 py-2 text-xs text-zinc-400">{selected.analysis_method === 'sample' ? 'Samples' : 'Top'} {result.candidates.length}</span></div><p className="text-xs leading-5 text-zinc-500">{selected.analysis_method === 'sample' ? 'These are timeline samples, not AI-identified highlights. Preview each section to choose one.' : 'Engagement Potential Score compares transcript and media signals; it does not predict views or guarantee reach.'}</p><CandidateList candidates={result.candidates} selected={selected.id} onSelect={setSelectedId}/><div className="space-y-3 rounded-xl border border-white/[.06] bg-black/20 p-3"><div className="flex items-center justify-between text-xs text-zinc-500"><span>Selected moment preview</span><span>{formatTime(selected.start)} → {formatTime(selected.end)}</span></div><VideoPlayer src={result.video.url} poster={result.video.thumbnail} startAt={selected.start} endAt={selected.end}/></div><button disabled={isRendering} onClick={() => void generate()} className="flex h-12 w-full items-center justify-center gap-2 rounded-xl bg-lime-300 text-sm font-semibold text-zinc-950 transition hover:bg-lime-200 disabled:cursor-wait disabled:opacity-60"><Film className="h-4 w-4"/>{isRendering ? 'Generating Short…' : 'Generate Short'}</button></section>
          {generationJob && <div className="rounded-2xl border border-white/[.08] bg-panel p-5"><div className="mb-3 flex justify-between text-sm"><span className="text-white">{generationJob.status === 'completed' ? 'Clip generation complete' : generationJob.status === 'failed' ? 'Clip generation stopped' : generationJob.stage}</span><span className="text-lime-300">{generationJob.progress}%</span></div><div className="h-1.5 overflow-hidden rounded-full bg-white/[.07]"><div className="h-full rounded-full bg-lime-300 transition-all" style={{ width: `${generationJob.progress}%` }}/></div><JobActivity logs={generationJob.logs ?? []}/></div>}
          {generationJob?.status === 'completed' && clipUrl && <ClipResult src={clipUrl}/>}
        </div>}
      </div>
      {!video && !starting && <div className="mt-14 grid gap-4 sm:grid-cols-3">{[['01', 'Analyze the story', 'Transcribe the video and surface promising moments.'], ['02', 'Compare candidates', 'Review each score and the signals behind it.'], ['03', 'Create your Short', 'Render a vertical clip with synchronized captions.']].map(([n, title, desc]) => <div key={n} className="rounded-2xl border border-white/[.06] bg-white/[.02] p-5"><div className="mb-4 flex items-center justify-between"><span className="text-xs font-semibold text-lime-300">{n}</span><Film className="h-4 w-4 text-zinc-700"/></div><h3 className="text-sm font-medium text-zinc-200">{title}</h3><p className="mt-1.5 text-xs leading-5 text-zinc-600">{desc}</p></div>)}</div>}
      <footer className="mt-14 flex items-center justify-center gap-2 text-center text-xs text-zinc-700"><Sparkles className="h-3.5 w-3.5 shrink-0"/>Engagement signals help compare clips; they don’t predict views.</footer>
    </main></div>
}
