import { useEffect, useState } from 'react'
import { api } from '../services/api'
import type { JobStatus } from '../types/video'

export function useJobPolling(jobId: string) {
  const [job, setJob] = useState<JobStatus | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    setJob(null)
    setError('')
    if (!jobId) return
    let cancelled = false
    let timer: ReturnType<typeof setTimeout>
    const poll = async () => {
      try {
        const next = await api.job(jobId)
        if (cancelled) return
        setJob(next)
        setError('')
        if (next.status === 'processing' || next.status === 'queued') timer = setTimeout(poll, 1400)
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : 'Unable to check job progress.')
          timer = setTimeout(poll, 3000)
        }
      }
    }
    void poll()
    return () => { cancelled = true; clearTimeout(timer) }
  }, [jobId])

  return { job, error }
}
