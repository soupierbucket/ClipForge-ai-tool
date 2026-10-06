import { ArrowUpRight, Link2 } from 'lucide-react'
import { useState, type FormEvent } from 'react'

export function UrlInput({ loading, onSubmit }: { loading: boolean; onSubmit: (url: string) => void }) {
  const [url, setUrl] = useState('')
  const submit = (event: FormEvent) => { event.preventDefault(); onSubmit(url.trim()) }
  return <form onSubmit={submit} className="flex flex-col gap-3 rounded-2xl border border-white/[.09] bg-white/[.035] p-2 sm:flex-row sm:items-center">
    <div className="flex min-w-0 flex-1 items-center gap-3 px-3"><Link2 className="h-5 w-5 shrink-0 text-zinc-500" /><input aria-label="YouTube URL" type="url" required value={url} onChange={e => setUrl(e.target.value)} placeholder="Paste a YouTube link here..." className="h-12 w-full bg-transparent text-sm text-white outline-none placeholder:text-zinc-600" /></div>
    <button disabled={loading} className="flex h-12 items-center justify-center gap-2 rounded-xl bg-lime-300 px-5 text-sm font-semibold text-zinc-950 transition hover:bg-lime-200 disabled:cursor-wait disabled:opacity-60">{loading ? 'Starting analysis…' : 'Analyze video'}<ArrowUpRight className="h-4 w-4" /></button>
  </form>
}
