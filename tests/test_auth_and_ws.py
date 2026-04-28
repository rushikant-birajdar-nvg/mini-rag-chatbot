import time
from collections.abc import AsyncIterator

import jwt
from fastapi.testclient import TestClient

import app.main as main_module
from app.auth import AuthError, decode_token
from app.config import get_settings
from app.models import AuthenticatedUser


def _mint_token(secret: str, exp_delta_seconds: int = 600) -> str:
    now = int(time.time())
    payload = {
        "sub": "emp-001",
        "email": "emp@test.com",
        "department": "hr",
        "level": 1,
        "iat": now,
        "exp": now + exp_delta_seconds,
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def test_decode_token_valid(monkeypatch) -> None:
    monkeypatch.setenv("JWT_SECRET", "required-test-secret-key-for-all-test-cases")
    get_settings.cache_clear()
    token = _mint_token("required-test-secret-key-for-all-test-cases", exp_delta_seconds=600)

    user = decode_token(token)
    assert user.user_id == "emp-001"
    assert user.department == "hr"
    assert user.level == 1
    assert user.exp > int(time.time())


def test_decode_token_expired(monkeypatch) -> None:
    monkeypatch.setenv("JWT_SECRET", "required-test-secret-key-for-all-test-cases")
    get_settings.cache_clear()
    token = _mint_token("required-test-secret-key-for-all-test-cases", exp_delta_seconds=-5)

    try:
        decode_token(token)
        assert False, "Expected AuthError for expired token"
    except AuthError as exc:
        assert "expired" in str(exc).lower()


def test_health_smoke() -> None:
    client = TestClient(main_module.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ws_rejects_non_auth_as_first_message() -> None:
    client = TestClient(main_module.app)
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"type": "message", "text": "hello"})
        data = ws.receive_json()
        assert data["type"] == "auth_failed"
        assert "First message must be auth" in data["message"]


def test_ws_auth_then_stream_protocol(monkeypatch) -> None:
    async def fake_stream_response(user: AuthenticatedUser, text: str) -> AsyncIterator[str]:
        assert user.user_id == "emp-001"
        assert text == "Hello?"
        yield "Part 1 "
        yield "Part 2"

    class FakeChatService:
        stream_response = staticmethod(fake_stream_response)

    def fake_decode_token(_: str) -> AuthenticatedUser:
        return AuthenticatedUser(
            user_id="emp-001",
            email="emp@test.com",
            department="hr",
            level=1,
            exp=int(time.time()) + 600,
        )

    monkeypatch.setattr(main_module, "chat_service", FakeChatService())
    monkeypatch.setattr(main_module, "decode_token", fake_decode_token)

    client = TestClient(main_module.app)
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"type": "auth", "token": "dummy"})
        auth_resp = ws.receive_json()
        assert auth_resp["type"] == "auth_success"
        assert auth_resp["user_id"] == "emp-001"

        ws.send_json({"type": "message", "text": "Hello?"})
        assert ws.receive_json() == {"type": "stream", "text": "Part 1 "}
        assert ws.receive_json() == {"type": "stream", "text": "Part 2"}
        assert ws.receive_json() == {"type": "done"}


def test_ws_rejects_message_when_token_expired(monkeypatch) -> None:
    class FakeChatService:
        @staticmethod
        async def stream_response(_: AuthenticatedUser, __: str) -> AsyncIterator[str]:
            yield "never reached"

    def fake_decode_token(_: str) -> AuthenticatedUser:
        return AuthenticatedUser(
            user_id="emp-001",
            email="emp@test.com",
            department="hr",
            level=1,
            exp=int(time.time()) - 1,
        )

    monkeypatch.setattr(main_module, "chat_service", FakeChatService())
    monkeypatch.setattr(main_module, "decode_token", fake_decode_token)

    client = TestClient(main_module.app)
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"type": "auth", "token": "dummy"})
        _ = ws.receive_json()  # auth_success
        ws.send_json({"type": "message", "text": "Hello?"})
        expired_resp = ws.receive_json()
        assert expired_resp["type"] == "auth_failed"
        assert "expired" in expired_resp["message"].lower()
