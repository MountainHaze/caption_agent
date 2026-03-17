from __future__ import annotations

from functools import lru_cache

from caption_agent.application.usecases.generate_instance_caption import (
    GenerateInstanceCaptionUseCase,
)


@lru_cache(maxsize=1)
def get_caption_usecase() -> GenerateInstanceCaptionUseCase:
    return GenerateInstanceCaptionUseCase()

