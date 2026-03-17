from __future__ import annotations

import uvicorn

from caption_agent.shared.config import AppSettings


if __name__ == "__main__":
    settings = AppSettings.from_env()
    uvicorn.run(
        "caption_agent.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )

