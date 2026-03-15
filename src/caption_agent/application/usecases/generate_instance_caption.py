from __future__ import annotations

import base64
from uuid import uuid4

from caption_agent.api.schemas.request import CaptionRequest
from caption_agent.domain.entities.bbox import BBox
from caption_agent.domain.entities.caption_result import CaptionResult
from caption_agent.domain.entities.person_instance import PersonInstance
from caption_agent.domain.policies.verification_policy import VerificationPolicy
from caption_agent.graph.builder import build_caption_graph
from caption_agent.infrastructure.llm.factory import build_multimodal_client
from caption_agent.infrastructure.observability.logging import get_logger
from caption_agent.shared.config import AppSettings

logger = get_logger("caption_agent.usecase")


class GenerateInstanceCaptionUseCase:
    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or AppSettings.from_env()
        self.llm_client = build_multimodal_client(self.settings)
        self.policy = VerificationPolicy(
            person_relation_threshold=self.settings.person_relation_threshold,
            object_relation_threshold=self.settings.object_relation_threshold,
        )
        self.graph = build_caption_graph(
            llm_client=self.llm_client,
            policy=self.policy,
        )

    def execute(self, request: CaptionRequest) -> tuple[CaptionResult, list[str]]:
        image_ref = self._resolve_image_ref(request)
        request_id = uuid4().hex
        persons = [
            PersonInstance(
                id=item.id,
                bbox=BBox.from_list(item.bbox),
                score=item.score,
            )
            for item in request.instances
        ]
        initial_state = {
            "request_id": request_id,
            "image_ref": image_ref,
            "instances": persons,
            "target_instance_id": request.target_instance_id,
            "language": request.language,
            "include_summary": request.include_summary,
            "artifact_dir": self.settings.artifact_dir,
            "errors": [],
        }
        final_state = self.graph.invoke(initial_state)
        errors = final_state.get("errors", [])
        if errors:
            logger.error(
                "Caption flow finished with errors. request_id=%s target_instance_id=%s errors=%s",
                request_id,
                request.target_instance_id,
                errors,
            )
        result_payload = final_state.get(
            "final_result",
            {
                "instance_id": request.target_instance_id,
                "attributes": {},
                "person_relations": [],
                "object_relations": [],
                "summary": None,
            },
        )
        return CaptionResult(**result_payload), errors

    @staticmethod
    def _resolve_image_ref(request: CaptionRequest) -> str:
        if request.image_base64:
            raw = request.image_base64.strip()
            try:
                base64.b64decode(raw, validate=True)
            except Exception as exc:  # noqa: BLE001
                raise ValueError("image_base64 is not valid base64") from exc
            mime = request.image_mime_type or "image/jpeg"
            return f"data:{mime};base64,{raw}"
        if request.image:
            return request.image
        raise ValueError("missing image input")
