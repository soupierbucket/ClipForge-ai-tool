import { Check, LoaderCircle } from 'lucide-react'
import type { JobStatus } from '../types/video'
import { JobActivity } from './JobActivity'

const stages = ['Loading video information', 'Downloading video', 'Transcribing audio', 'Analyzing candidate moments', 'Scoring audio and visual activity']

export function JobProgress({ job }: { job: JobStatus }) {
  const current = stages.findIndex(stage => stage.toLowerCase() === job.stage.toLowerCase())
  const completed = job.status === 'completed'
  return <section className="rounded-2xl border border-white/[.08] bg-panel p-5 sm:p-6">
    <div className="flex items-center justify-between gap-4"><div><h3 className="text-sm font-semibold text-white">{completed ? 'Analysis activity' : job.status === 'failed' ? 'Analysis stopped' : 'Analyzing your video'}</h3><p className="mt-1 text-xs text-zinc-500">{job.stage || 'Queued'}</p></div><span className="shrink-0 text-sm font-semibold tabular-nums text-lime-300">{job.progress}%</span></div>
    <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-white/[.07]"><div className="h-full rounded-full bg-lime-300 transition-all duration-500" style={{ width: `${job.progress}%` }}/></div>
    <div className="mt-5 grid gap-2 sm:grid-cols-2">{stages.map((stage, index) => {
      const done = completed || (current >= 0 && current > index)
      const active = !completed && current === index && job.status !== 'failed'
      return <div key={stage} className="flex items-center gap-2 text-xs"><span className={`grid h-5 w-5 place-items-center rounded-full ${done || active ? 'bg-lime-300/15 text-lime-300' : 'bg-white/[.04] text-zinc-700'}`}>{active ? <LoaderCircle className="h-3 w-3 animate-spin"/> : done ? <Check className="h-3 w-3"/> : index + 1}</span><span className={done || active ? 'text-zinc-300' : 'text-zinc-700'}>{stage}</span></div>
    })}</div>
    {completed && <details className="mt-4 rounded-xl border border-white/[.06] bg-black/20 px-3 py-2.5">
      <summary className="cursor-pointer text-xs font-medium text-zinc-400">How the score is calculated</summary>
      <p className="mt-2 text-xs leading-5 text-zinc-500">Hook 25%, emotion 20%, novelty 15%, visual activity 15%, audio activity 10%, payoff 10%, and context 5%. Each signal is scored from 0–10 and combined into a 0–100 ranking aid; it does not predict views.</p>
    </details>}
    <JobActivity logs={job.logs ?? []}/>
  </section>
}
