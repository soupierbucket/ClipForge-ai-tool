import { Music2, Sparkles, WandSparkles } from 'lucide-react'
import type { ClipEffect, EffectFilter, EffectSound } from '../types/video'

const sounds: (EffectSound | 'none')[] = ['none', 'hit', 'whoosh', 'rimshot', 'record_scratch', 'ding', 'crickets']
const filters: EffectFilter[] = ['none', 'vignette', 'saturation', 'desaturate', 'contrast']
const title = (value: string) => value.replace(/_/g, ' ').replace(/^\w/, (character: string) => character.toUpperCase())

export function EffectsEditor({ effects, onChange, loading }: { effects: ClipEffect[]; onChange: (effects: ClipEffect[]) => void; loading: boolean }) {
  const update = (index: number, changes: Partial<ClipEffect>) => onChange(effects.map((item, i) => i === index ? { ...item, ...changes } : item))
  return <section className="space-y-3 rounded-xl border border-white/[.06] bg-black/20 p-3">
    <div className="flex items-center gap-2"><WandSparkles className="h-4 w-4 text-lime-300"/><div><h3 className="text-sm font-medium text-white">Effects timeline</h3><p className="text-[11px] text-zinc-500">Review optional cues before rendering.</p></div></div>
    {loading ? <p className="text-xs text-zinc-500">Checking transcript cues and audio peaks…</p> : effects.length === 0 ? <p className="text-xs text-zinc-500">No supported moments found. This clip will render without extra effects.</p> : <div className="space-y-2">
      {effects.map((effect, index) => <div key={`${effect.time}-${index}`} className="grid gap-2 rounded-lg border border-white/[.06] bg-white/[.025] p-2 sm:grid-cols-[1fr_1.1fr_1fr_auto] sm:items-center">
        <label className="flex items-center gap-2"><input type="checkbox" checked={effect.enabled} onChange={e => update(index, { enabled: e.target.checked })} className="accent-lime-300"/><span className="text-xs text-zinc-200">{effect.time.toFixed(1)}s · {title(effect.moment_type)}</span></label>
        <label className="flex items-center gap-1.5 text-[10px] text-zinc-500"><Music2 className="h-3 w-3"/><select value={effect.sfx ?? 'none'} onChange={e => update(index, { sfx: e.target.value === 'none' ? null : e.target.value as EffectSound })} className="min-w-0 rounded-md border border-white/10 bg-zinc-900 px-2 py-1.5 text-xs text-zinc-200">{sounds.map(value => <option key={value} value={value}>{title(value)}</option>)}</select></label>
        <label className="flex items-center gap-1.5 text-[10px] text-zinc-500"><Sparkles className="h-3 w-3"/><select value={effect.filter} onChange={e => update(index, { filter: e.target.value as EffectFilter })} className="min-w-0 rounded-md border border-white/10 bg-zinc-900 px-2 py-1.5 text-xs text-zinc-200">{filters.map(value => <option key={value} value={value}>{title(value)}</option>)}</select></label>
        <span className="text-[10px] text-zinc-600">{effect.duration.toFixed(1)}s</span>
      </div>)}
    </div>}
  </section>
}