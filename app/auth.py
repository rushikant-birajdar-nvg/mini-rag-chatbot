"""JWT authentication helpers for websocket session setup."""

import jwt
from jwt import ExpiredSignatureError, InvalidTokenError
from time import time

from app.config import get_settings
from app.models import AuthenticatedUser


class AuthError(Exception):
    """Raised when token validation fails or claims are incomplete."""

    pass


def decode_token(token: str) -> AuthenticatedUser:
    """Decode and validate JWT, returning normalized user claims."""
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except ExpiredSignatureError as exc:
        raise AuthError("Token expired") from exc
    except InvalidTokenError as exc:
        raise AuthError("Invalid token") from exc

    user_id = payload.get("sub")
    email = payload.get("email")
    department = payload.get("department")
    level = payload.get("level")
    exp = payload.get("exp")
    if not all([user_id, email, department]) or level is None or exp is None:
        raise AuthError("Token missing required claims")

    return AuthenticatedUser(
        user_id=str(user_id),
        email=str(email),
        department=str(department),
        level=int(level),
        exp=int(exp),
    )


def is_user_token_expired(user: AuthenticatedUser) -> bool:
    """Return True when the authenticated user's token is expired."""
    return int(time()) >= user.exp

