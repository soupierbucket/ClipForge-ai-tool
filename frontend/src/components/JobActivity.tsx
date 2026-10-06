import { AlertTriangle, CheckCircle2, CircleAlert, Info } from 'lucide-react'
import type { JobLog } from '../types/video'

const styles = {
  info: { icon: Info, color: 'text-zinc-500' },
  success: { icon: CheckCircle2, color: 'text-lime-300' },
  warning: { icon: AlertTriangle, color: 'text-amber-300' },
  error: { icon: CircleAlert, color: 'text-rose-300' },
}

export function JobActivity({ logs }: { logs: JobLog[] }) {
  if (!logs.length) return null
  return <div className="mt-5 border-t border-white/[.06] pt-4">
    <h4 className="mb-3 text-[10px] font-semibold uppercase tracking-[.14em] text-zinc-600">Activity log</h4>
    <ol className="max-h-56 space-y-2 overflow-y-auto pr-1" aria-live="polite" aria-label="Analysis activity log">
      {logs.map((entry, index) => {
        const { icon: Icon, color } = styles[entry.level]
        const time = new Date(entry.timestamp)
        return <li key={`${entry.timestamp}-${index}`} className="flex items-start gap-2.5 text-xs leading-5">
          <Icon className={`mt-1 h-3.5 w-3.5 shrink-0 ${color}`} />
          <time className="shrink-0 pt-px font-mono text-[10px] tabular-nums text-zinc-700" dateTime={entry.timestamp}>{Number.isNaN(time.valueOf()) ? '' : time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</time>
          <span className={entry.level === 'error' ? 'text-rose-200' : 'text-zinc-400'}>{entry.message}</span>
        </li>
      })}
    </ol>
  </div>
}
