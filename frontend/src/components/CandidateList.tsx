import { Check, Sparkles } from 'lucide-react'
import type { Candidate } from '../types/video'
import { formatTime } from '../utils'

const metrics: [keyof Candidate['scores'], string][] = [
  ['hook', 'Hook'],
  ['curiosity', 'Curiosity'],
  ['emotional_intensity', 'Emotion'],
  ['novelty', 'Novelty'],
  ['entertainment', 'Entertainment'],
  ['visual_potential', 'Visual potential'],
  ['payoff', 'Payoff'],
  ['context', 'Context'],
  ['standalone', 'Standalone'],
  ['rewatch_potential', 'Rewatch'],
]

export function CandidateList({ candidates, selected, onSelect }: { candidates: Candidate[]; selected: string; onSelect: (id: string) => void }) {
  return <div className="space-y-3">
    {candidates.map((candidate, index) => {
      const active = candidate.id === selected
      return <article key={candidate.id} className={`rounded-xl border p-3 transition ${active ? 'border-lime-300/40 bg-lime-300/[.045]' : 'border-white/[.08] bg-white/[.025] hover:border-white/15'}`}>
        <button onClick={() => onSelect(candidate.id)} className="w-full text-left" aria-pressed={active}>
          <div className="flex items-start gap-3">
            <span className={`mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded-full border ${active ? 'border-lime-300 bg-lime-300 text-zinc-950' : 'border-zinc-600 text-transparent'}`}>
              {active && <Check className="h-3 w-3"/>}
            </span>
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-sm font-semibold text-white">{candidate.analysis_method === 'sample' ? `Timeline sample ${index + 1}` : candidate.recommended ? 'Recommended complete moment' : `Candidate ${index + 1}`}</span>
                {candidate.recommended && candidate.analysis_method === 'ai' && <span className="rounded-full border border-lime-300/20 bg-lime-300/10 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-lime-200">Best story arc</span>}
              </div>
              <p className="mt-1 text-xs text-zinc-500">{formatTime(candidate.start)} → {formatTime(candidate.end)} <span className="px-1 text-zinc-700">·</span> {formatTime(candidate.duration)}</p>
            </div>
            {candidate.analysis_method === 'ai' ? <div className="shrink-0 text-right">
              <div className="text-xl font-semibold tracking-tight text-lime-300">{candidate.engagement_potential_score}</div>
              <div className="text-[10px] text-zinc-600">/ 100</div>
            </div> : <span className="shrink-0 rounded-md bg-white/[.05] px-2 py-1 text-[10px] font-medium text-zinc-500">Sample</span>}
          </div>
          <div className="mt-3 grid grid-cols-3 gap-2">
            {[
              [candidate.analysis_method === 'sample' ? 'Source interval' : 'Hook', candidate.hook_summary],
              [candidate.analysis_method === 'sample' ? 'Transcript' : 'Context & development', candidate.context_summary],
              [candidate.analysis_method === 'sample' ? 'Clip note' : 'Payoff', candidate.payoff_summary],
            ].map(([label, value]) => <div key={label} className="min-w-0 rounded-lg border border-white/[.05] bg-black/20 px-2.5 py-2">
              <div className="truncate text-[9px] font-semibold uppercase tracking-wide text-lime-300/70">{label}</div>
              <p className="mt-0.5 text-[11px] leading-4 text-zinc-300">{value}</p>
            </div>)}
          </div>
        </button>
        {candidate.analysis_method === 'ai' && <div className="mt-3 grid grid-cols-2 gap-1.5 sm:grid-cols-5">
          {metrics.map(([key, label]) => <div key={key} className="flex min-w-0 items-center justify-between gap-1.5 rounded-md bg-black/20 px-2 py-1.5">
            <span className="truncate text-[9px] uppercase tracking-wide text-zinc-500">{label}</span>
            <span className="shrink-0 text-[11px] font-medium tabular-nums text-zinc-300">{candidate.scores[key].toFixed(1)}<span className="text-zinc-600">/10</span></span>
          </div>)}
        </div>}
        <details className="group mt-2">
          <summary className="cursor-pointer list-none text-[10px] font-medium text-zinc-500 transition hover:text-zinc-300">Why this works <span className="text-zinc-700 group-open:hidden">· show details</span><span className="hidden text-zinc-700 group-open:inline">· hide details</span></summary>
          <div className="mt-2 space-y-2 border-l border-white/10 pl-3">
            <p className="text-xs leading-5 text-zinc-300">{candidate.reason}</p>
            <blockquote className="text-[11px] italic leading-5 text-zinc-500">“{candidate.transcript_excerpt}”</blockquote>
            {candidate.evidence.map((item, evidenceIndex) => <p key={evidenceIndex} className="text-[10px] leading-4 text-zinc-600"><Sparkles className="mr-1 inline h-3 w-3 text-zinc-700"/>{item}</p>)}
          </div>
        </details>
      </article>
    })}
  </div>
}
