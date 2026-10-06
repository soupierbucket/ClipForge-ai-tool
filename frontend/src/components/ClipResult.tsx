import { ArrowDownToLine, CheckCircle2 } from 'lucide-react'
import { VideoPlayer } from './VideoPlayer'

export function ClipResult({ src }: { src: string }) {
  return <section className="space-y-4 rounded-2xl border border-lime-300/20 bg-lime-300/[.035] p-5 sm:p-6"><div className="flex flex-wrap items-center justify-between gap-3"><div><h3 className="flex items-center gap-2 font-semibold text-white"><CheckCircle2 className="h-4 w-4 text-lime-300"/>Your Short is ready</h3><p className="mt-1 text-sm text-zinc-400">9:16 vertical · Word-synced English captions · Blurred fill · Framed zooms · MP4</p></div><a href={src} download className="flex items-center gap-2 rounded-xl bg-lime-300 px-4 py-2.5 text-sm font-semibold text-zinc-950 transition hover:bg-lime-200"><ArrowDownToLine className="h-4 w-4"/>Download Short</a></div><VideoPlayer src={src}/></section>
}
