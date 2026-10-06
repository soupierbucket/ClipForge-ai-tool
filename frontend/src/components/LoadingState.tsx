import { LoaderCircle } from 'lucide-react'

export function LoadingState({ label }: { label: string }) {
  return <div className="flex items-center gap-2 text-sm text-lime-300"><LoaderCircle className="h-4 w-4 animate-spin" />{label}</div>
}
