# ClipForge

ClipForge analyzes long-form YouTube videos and surfaces up to five short-form candidates, each capped at 60 seconds. It transcribes speech, asks a configurable LLM provider for transcript-based candidates, combines those scores with measured audio and video activity, and renders a selected moment as a 9:16 MP4 with word-timed animated captions, occasional context-matched emojis, a blurred fill behind fitted frames, and face-tracked portrait punch-ins where possible.

The **Engagement Potential Score** is a transparent ranking of signals. It is not a prediction of views and does not guarantee reach.

## Project structure

```text
frontend/
  src/components/       UI components for metadata, progress, candidates, and output
  src/hooks/            reusable job-polling hook
  src/pages/            home workflow
  src/services/         typed API client
  src/types/            API types
backend/app/
  api/                  HTTP routes and job submission
  analyzers/            measured audio and video features
  models/               request and response models
  services/             downloading, transcription, LLM, selection, captions, rendering, jobs
  utils/                URL validation
```

Analysis and rendering run as background jobs. The frontend polls `GET /api/jobs/{job_id}` for progress, so it does not hold one HTTP request open for minutes. Jobs are stored in memory for this initial local version; media is stored under `backend/temp_media/` by default. Expired jobs and media are cleaned the next time a job is created or queried. On a graceful backend shutdown (including Ctrl+C or a Uvicorn reload), ClipForge waits for active worker tasks, cancels queued tasks, then deletes the dedicated `jobs/` and `clips/` directories and clears the in-memory job registry. If the process is force-killed or the computer loses power, shutdown cleanup cannot run and temporary files may remain. A server restart clears job state and removes old clips. Use a persistent queue and object storage before deploying multiple backend instances.

## Requirements

- Node.js 20.19+ (or 22.12+) with npm (used once to install pnpm)
- pnpm
- Python 3.10+
- FFmpeg installed and available on `PATH`
- Ollama for local, no-API-bill analysis (default)

Install Ollama from its [official Windows download page](https://ollama.com/download/windows). Then download the local model once:

```powershell
ollama pull qwen2.5:7b
```

The model download is about 4.7 GB. Ollama runs it locally, so analysis uses your computer's memory and CPU/GPU. [Ollama's model page](https://ollama.com/library/qwen2.5%3A7b) lists the model details.

Install FFmpeg on Windows with WinGet:

```powershell
winget install --id Gyan.FFmpeg.Shared --exact
```

Confirm the installation in a new terminal with `ffmpeg -version`.

## Run locally (Git Bash on Windows)

Open two terminals at the project root.

### Backend

```bash
cd backend
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp -n .env.example .env
```

Edit `backend/.env` and ensure `LLM_PROVIDER=ollama` and `LLM_MODEL=qwen2.5:7b` are set.

```bash
uvicorn app.main:app --reload
```

The API is at `http://localhost:8000`, API docs at `http://localhost:8000/docs`, and health check at `http://localhost:8000/health`.

### Frontend

```bash
cd frontend
npm install --global pnpm
pnpm install
cp -n .env.example .env
pnpm run dev
```

Open `http://localhost:5173`. The Vite development server proxies `/api` to `http://localhost:8000`. Set `VITE_API_BASE_URL` in `frontend/.env` only when using a different API host.

The first transcription downloads the configured faster-whisper model. On CPU, `WHISPER_MODEL=small` can take several minutes and needs extra disk space; use `base` in `backend/.env` for a lighter local setup. `WHISPER_DEVICE=cpu` is the default.

## Configuration

Backend settings are loaded from `backend/.env`:

- `LLM_PROVIDER`: defaults to `ollama` for local analysis. Set it to `openai` to use the paid API instead. The `CandidateAnalyzer` interface isolates provider-specific analysis.
- `LLM_MODEL`: defaults to the local `qwen2.5:7b` model; change it to a model already pulled in Ollama.
- `OLLAMA_BASE_URL`: local Ollama server URL, default `http://localhost:11434`.
- `LLM_API_KEY`: only needed when `LLM_PROVIDER=openai`. Keep it in `backend/.env`, never in the frontend. OpenAI API requests are usage-billed; see the [current API pricing](https://developers.openai.com/api/docs/pricing).
- `WHISPER_MODEL`, `WHISPER_DEVICE`: faster-whisper model and runtime device.
- `MAX_VIDEO_SIZE_MB`, `MAX_VIDEO_DURATION_SECONDS`: download and source duration limits.
- `CLIPFORGE_TEMP_DIR`: optional location for temporary source videos and generated clips.
- `FRONTEND_ORIGINS`: comma-separated CORS origins.
- `JOB_RETENTION_HOURS`: in-memory job/media retention window.

## API

- `GET /api/video-info?url=<youtube-url>` — validate a YouTube URL and fetch metadata.
- `POST /api/analyze` with `{"url":"https://youtube.com/watch?v=..."}` — returns a job ID.
- `GET /api/jobs/{job_id}` — returns status, stage, progress, candidate scores, and evidence.
- `POST /api/generate-clip` with `{"job_id":"...","candidate_id":"..."}` — starts the selected clip render job.
- `GET /api/clips/{clip_id}` — streams the rendered MP4.

Candidate scoring weights: hook 25%, emotion 20%, novelty 15%, visual activity 15%, audio intensity 10%, payoff 10%, and standalone context 5%. Transcript scores come from the LLM's analysis of transcript content. Visual and audio scores come from sampled frame differences/scene-change signals and RMS audio activity. These are engagement-oriented heuristics, not view predictions.

## Notes

Use videos you own or have permission to process, and follow YouTube's terms and applicable copyright rules. This local prototype has no authentication, rate limiting, durable job queue, or cloud storage. Avoid exposing it publicly without adding those controls.
