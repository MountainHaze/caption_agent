# Caption Agent MVP

This project implements a first MVP for describing a target person instance in an image.

## Scope

- Input: `image`, all person `bbox` instances, and `target_instance_id`.
- Output: structured fields for:
  - person attributes
  - person-to-person relations
  - conservative person-to-object relations
  - optional short summary in Chinese

## Why LangGraph

The workflow is deterministic and stateful:

1. validate input
2. build visual context
3. extract attributes
4. extract relations
5. verify facts
6. compose summary

LangGraph is used as the orchestration layer and is intentionally designed for long-term iteration.
This v2 codebase requires LangGraph and does not include non-framework fallbacks.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
python scripts/run_dev.py
```

Then open:

- `http://127.0.0.1:8000/docs`

## API

- `POST /v1/caption`
- `POST /v1/caption/upload` (multipart file upload)

See request/response models in:

- `src/caption_agent/api/schemas/request.py`
- `src/caption_agent/api/schemas/response.py`

Example payload:

```json
{
  "image_base64": "<base64-image-bytes>",
  "image_mime_type": "image/jpeg",
  "instances": [
    {"id": "p1", "bbox": [0.620039, 0.5939, 0.172415, 0.14608], "bbox_format": "norm_xywh"}
  ],
  "target_instance_id": "p1",
  "language": "zh",
  "include_summary": true
}
```

Notes:

- For external users, prefer `image_base64` or `/v1/caption/upload`.
- `image` field still supports `http(s)` URL and local path for debugging.
- If `instances` contains one item, API returns one object.
- If `instances` contains multiple items, API returns an array (one result per instance id).

`/v1/caption/upload` form fields:

- `image_file`: binary image file
- `instances_json`: JSON string, same schema as `instances`
- `target_instance_id`
- `language` (optional, default `zh`)
- `include_summary` (optional, default `true`)

`bbox` input formats:

- `norm_xywh`: normalized `[cx, cy, w, h]`, each value in `0-1`
- `xyxy`: absolute pixel `[x1, y1, x2, y2]`
- `xywh`: absolute pixel `[x, y, w, h]`
- `auto` (default): inferred from value range

## Notes

- This MVP v2 includes a LangChain Runnable-based client (`LangChainMultimodalClient`) as a conservative baseline.
- You can replace runnables with real model-backed chains in `src/caption_agent/infrastructure/llm/multimodal_client.py`.
- Use `src/caption_agent/infrastructure/llm/factory.py` as the provider wiring entry.

## Online Model Setup

Copy `.env.example` to `.env`, then set provider and keys.

OpenAI:

```env
CAPTION_AGENT_LLM_PROVIDER=openai
CAPTION_AGENT_LLM_MODEL=gpt-4.1-mini
OPENAI_API_KEY=your_openai_api_key
```

Qwen (DashScope OpenAI-compatible endpoint):

```env
CAPTION_AGENT_LLM_PROVIDER=qwen
CAPTION_AGENT_LLM_MODEL=qwen-vl-max
DASHSCOPE_API_KEY=your_dashscope_api_key
# optional override:
# CAPTION_AGENT_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

Generic override (works for both):

```env
CAPTION_AGENT_LLM_API_KEY=your_key
CAPTION_AGENT_LLM_BASE_URL=your_openai_compatible_base_url
```

Priority:

- API key: `CAPTION_AGENT_LLM_API_KEY` > provider-specific key
- Base URL: `CAPTION_AGENT_LLM_BASE_URL` > provider default

## Prompt Tuning

This project now uses a centralized prompt bundle for online model calls.

- Prompt source file: `configs/prompts.yaml`
- Override path via env: `CAPTION_AGENT_PROMPTS_CONFIG=your_path`
- Loaded at startup in `build_multimodal_client(...)`

Prompt keys:

- `prompts.attribute`
- `prompts.person_relation`
- `prompts.object_relation`
- `prompts.summary`

After editing prompts, restart the API service to apply changes.
