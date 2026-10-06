import asyncio
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import executor, router
from app.config import settings
from app.services.job_store import jobs
from app.services.media_cleanup import cleanup_temporary_media

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("ClipForge API started; temporary media will be deleted on graceful shutdown.")
    try:
        yield
    finally:
        logger.info("Shutting down ClipForge jobs before cleaning temporary media.")
        try:
            # Let active downloads/renders finish and cancel jobs that have not started,
            # so workers cannot recreate files while cleanup is running.
            await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)
        finally:
            jobs.clear()
            cleanup_temporary_media()

app = FastAPI(title="ClipForge AI Shorts API", version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.origins, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])
app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}
