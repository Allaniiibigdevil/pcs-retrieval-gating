import logging

from fastapi import FastAPI

from app.api import decision, search
from app.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

settings = get_settings()
app = FastAPI(title=settings.APP_NAME)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(search.router)
app.include_router(decision.router)
