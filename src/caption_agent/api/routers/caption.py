from __future__ import annotations

import base64
import json
from typing import TypeAlias

from fastapi import APIRouter, Depends
from fastapi import File, Form, HTTPException, UploadFile

from caption_agent.api.dependencies import get_caption_usecase
from caption_agent.api.schemas.request import CaptionRequest
from caption_agent.api.schemas.request import PersonInstanceInput
from caption_agent.api.schemas.response import CaptionResponse
from caption_agent.application.usecases.generate_instance_caption import (
    GenerateInstanceCaptionUseCase,
)

router = APIRouter(prefix="/v1", tags=["caption"])
CaptionApiResponse: TypeAlias = CaptionResponse | list[CaptionResponse]


def _run_caption_for_targets(
    usecase: GenerateInstanceCaptionUseCase,
    base_request: CaptionRequest,
) -> CaptionApiResponse:
    instance_ids = [item.id for item in base_request.instances]
    if len(instance_ids) <= 1:
        result, errors = usecase.execute(base_request)
        return CaptionResponse(
            instance_id=result.instance_id,
            attributes=result.attributes,
            person_relations=[item.model_dump() for item in result.person_relations],
            object_relations=[item.model_dump() for item in result.object_relations],
            summary=result.summary,
            errors=errors,
        )

    responses: list[CaptionResponse] = []
    for instance_id in instance_ids:
        request = base_request.model_copy(update={"target_instance_id": instance_id})
        result, errors = usecase.execute(request)
        responses.append(
            CaptionResponse(
                instance_id=result.instance_id,
                attributes=result.attributes,
                person_relations=[item.model_dump() for item in result.person_relations],
                object_relations=[item.model_dump() for item in result.object_relations],
                summary=result.summary,
                errors=errors,
            )
        )
    return responses


@router.post("/caption", response_model=CaptionResponse | list[CaptionResponse])
def caption(
    request: CaptionRequest,
    usecase: GenerateInstanceCaptionUseCase = Depends(get_caption_usecase),
) -> CaptionApiResponse:
    try:
        return _run_caption_for_targets(usecase, request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/caption/upload", response_model=CaptionResponse | list[CaptionResponse])
async def caption_upload(
    image_file: UploadFile = File(...),
    instances_json: str = Form(..., description="JSON array of person instances."),
    target_instance_id: str = Form(...),
    language: str = Form("zh"),
    include_summary: bool = Form(True),
    usecase: GenerateInstanceCaptionUseCase = Depends(get_caption_usecase),
) -> CaptionApiResponse:
    try:
        instances_raw = json.loads(instances_json)
        instances = [PersonInstanceInput.model_validate(item) for item in instances_raw]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="instances_json is invalid") from exc

    image_bytes = await image_file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="image_file is empty")
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = image_file.content_type or "image/jpeg"

    request = CaptionRequest(
        image_base64=image_base64,
        image_mime_type=mime_type,
        instances=instances,
        target_instance_id=target_instance_id,
        language=language,
        include_summary=include_summary,
    )
    try:
        return _run_caption_for_targets(usecase, request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
