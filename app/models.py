from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


class AuthMessage(BaseModel):
    type: str
    token: str


class UserMessage(BaseModel):
    type: str
    text: str


@dataclass
class AuthenticatedUser:
    user_id: str
    email: str
    department: str
    level: int
    exp: int

