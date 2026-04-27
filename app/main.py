import asyncio
import json
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.auth import AuthError, decode_token, is_user_token_expired
from app.config import get_settings
from app.models import AuthMessage, UserMessage

app = FastAPI(title="Mini RAG Chatbot Backend")
logger = logging.getLogger(__name__)
settings = get_settings()

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    await websocket.accept()

    try:
        raw_auth = await websocket.receive_text()
        auth_payload = json.loads(raw_auth)
        if not isinstance(auth_payload, dict) or auth_payload.get("type") != "auth":
            await websocket.send_json({"type": "auth_failed", "message": "First message must be auth"})
            await websocket.close()
            return
        auth = AuthMessage.model_validate(auth_payload)
        user = decode_token(auth.token)
        await websocket.send_json(
            {
                "type": "auth_success",
                "user_id": user.user_id,
                "department": user.department,
                "level": user.level,
            }
        )
    except (json.JSONDecodeError, ValidationError):
        await websocket.send_json({"type": "auth_failed", "message": "Malformed auth payload"})
        await websocket.close()
        return
    except AuthError as exc:
        await websocket.send_json({"type": "auth_failed", "message": str(exc)})
        await websocket.close()
        return

    while True:
        try:
            raw_msg = await websocket.receive_text()
            msg = UserMessage.model_validate(json.loads(raw_msg))
            if msg.type != "message":
                await websocket.send_json({"type": "error", "message": "Unsupported message type"})
                continue
            if is_user_token_expired(user):
                await websocket.send_json(
                    {"type": "auth_failed", "message": "Token expired during session"}
                )
                await websocket.close(code=1008)
                break

            await websocket.send_json({"type": "stream", "text": "Streaming successfully..."})
            await websocket.send_json({"type": "done"})
        except WebSocketDisconnect:
            break
        except (json.JSONDecodeError, ValidationError):
            await websocket.send_json({"type": "error", "message": "Malformed message payload"})
        except Exception:
            logger.exception("WebSocket message handling failed for user_id=%s", user.user_id)
            await websocket.send_json(
                {"type": "error", "message": "Request failed due to an internal error."}
            )

