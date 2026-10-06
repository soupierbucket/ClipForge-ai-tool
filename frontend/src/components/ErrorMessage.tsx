import { AlertCircle } from 'lucide-react'

export function ErrorMessage({ message }: { message: string }) {
  if (!message) return null
  return <div role="alert" className="flex gap-3 rounded-xl border border-rose-400/20 bg-rose-400/[.07] p-4 text-sm text-rose-200"><AlertCircle className="h-5 w-5 shrink-0" />{message}</div>
}
