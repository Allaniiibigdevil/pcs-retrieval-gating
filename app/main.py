from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import decision, rerank, search
from app.config import get_settings
from app.logging_config import configure_logging

configure_logging()

settings = get_settings()
app = FastAPI(title=settings.APP_NAME)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(search.router)
app.include_router(decision.router)
app.include_router(rerank.router)

frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/frontend", StaticFiles(directory=frontend_dir, html=True), name="frontend")
