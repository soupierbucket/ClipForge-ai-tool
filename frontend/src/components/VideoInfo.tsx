import { CirclePlay, Clock3 } from 'lucide-react'
import type { VideoInfo as VideoInfoType } from '../types/video'
import { formatTime } from '../utils'

export function VideoInfo({ video }: { video: VideoInfoType }) {
  return <div className="flex min-w-0 items-center gap-4"><img src={video.thumbnail} alt="" className="h-16 w-28 shrink-0 rounded-lg object-cover"/><div className="min-w-0"><h2 className="line-clamp-2 text-sm font-semibold leading-5 text-white">{video.title}</h2><p className="mt-1 truncate text-xs text-zinc-500">{video.channel}</p><div className="mt-2 flex gap-3 text-xs text-zinc-400"><span className="flex items-center gap-1"><Clock3 className="h-3.5 w-3.5"/>{formatTime(video.duration)}</span><span className="flex items-center gap-1"><CirclePlay className="h-3.5 w-3.5"/>YouTube video</span></div></div></div>
}
