import { Play } from 'lucide-react'

export function VideoPlayer({ src, poster, startAt = 0, endAt }: { src: string; poster?: string; startAt?: number; endAt?: number }) {
  const youtubeId = (() => {
    try {
      const url = new URL(src)
      if (url.hostname.endsWith('youtu.be')) return url.pathname.slice(1)
      return url.searchParams.get('v')
    } catch { return null }
  })()
  const windowArgs = new URLSearchParams({ start: String(Math.floor(startAt)), ...(endAt ? { end: String(Math.floor(endAt)) } : {}) })
  return <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-black">{youtubeId ? <iframe key={`${youtubeId}-${startAt}`} className="aspect-video w-full" src={`https://www.youtube-nocookie.com/embed/${encodeURIComponent(youtubeId)}?${windowArgs}`} title="YouTube video preview" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerPolicy="strict-origin-when-cross-origin" allowFullScreen /> : <video key={src} className="aspect-video w-full" controls playsInline preload="metadata" poster={poster}><source src={src} type="video/mp4" />Your browser does not support video playback.</video>}<div className="pointer-events-none absolute left-4 top-4 flex items-center gap-2 rounded-full border border-white/10 bg-black/60 px-3 py-1.5 text-xs text-white/80"><Play className="h-3.5 w-3.5 fill-current" />Preview</div></div>
}
