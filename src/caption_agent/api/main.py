from __future__ import annotations

from fastapi import FastAPI

from caption_agent.api.routers.caption import router as caption_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Caption Agent API",
        version="0.1.0",
        description="MVP API for person-instance structured captioning.",
    )
    app.include_router(caption_router)
    return app


app = create_app()

