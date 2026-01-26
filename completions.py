"""OpenAI-compatible completions API for agenda items."""

import json
import os
import time
import uuid

# Set default URLs for local development (outside Docker)
if not os.getenv("QDRANT_URL"):
    os.environ["QDRANT_URL"] = "http://localhost:6333"
if not os.getenv("FLAGSERVE_URL"):
    os.environ["FLAGSERVE_URL"] = "http://localhost:8273"

from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from src.agent import run_agent

app = FastAPI(title="ByrådsGPT Completions API")

API_KEY = os.getenv("COMPLETIONS_API_KEY", "")


def verify_api_key(authorization: str | None = Header(None)):
    """Verify API key from Authorization header."""
    if not API_KEY:
        return  # No key configured, allow all
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    # Support both "Bearer <key>" and just "<key>"
    key = authorization.removeprefix("Bearer ").strip()
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: str | None = Header(None)):
    """OpenAI-compatible chat completions endpoint."""
    verify_api_key(authorization)
    body = await request.json()

    messages = body.get("messages", [])
    stream = body.get("stream", False)

    # Extract conversation history and current query
    history = []
    query = ""
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            continue  # We use our own system prompt
        elif role in ("user", "assistant"):
            if role == "user" and msg == messages[-1]:
                query = content
            else:
                history.append({"role": role, "content": content})

    if not query:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "No user message found", "type": "invalid_request_error"}},
        )

    request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model = "byraadsgpt"

    if stream:
        return StreamingResponse(
            _stream_response(query, history, request_id, created, model),
            media_type="text/event-stream",
        )
    else:
        return _sync_response(query, history, request_id, created, model)


def _sync_response(query: str, history: list, request_id: str, created: int, model: str) -> JSONResponse:
    """Generate non-streaming response."""
    content = ""
    for update in run_agent(query, history):
        if update["type"] == "answer":
            content = update["content"]

    return JSONResponse({
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


def _stream_response(query: str, history: list, request_id: str, created: int, model: str):
    """Generate streaming response in SSE format."""
    # Send initial chunk with role
    initial_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": ""},
            "finish_reason": None,
        }],
    }
    yield f"data: {json.dumps(initial_chunk)}\n\n"

    for update in run_agent(query, history, stream=True):
        if update["type"] == "tool_call":
            # Send tool call as a content chunk (optional visibility)
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": f"*Søger: {update['name']}...*\n\n"},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        elif update["type"] == "answer_chunk":
            # Stream answer token by token
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": update["content"]},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        elif update["type"] == "answer_done":
            # Send final chunk with finish_reason
            final_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            return
        elif update["type"] == "answer":
            # Fallback for non-streamed answer (shouldn't happen with stream=True)
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": update["content"]},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    # Fallback final chunk if answer_done wasn't received
    final_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [{
            "id": "byraadsgpt",
            "object": "model",
            "created": 1700000000,
            "owned_by": "local",
        }],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
