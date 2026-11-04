"""Request and response shapes. Input is validated here, before any logic runs."""

import re
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, field_validator

USERNAME_RE = re.compile(r"[a-z0-9_]{3,32}")


class RegisterIn(BaseModel):
    """Usernames are stored lowercase, so 'Alice' and 'alice' are the same account."""

    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("username", mode="before")
    @classmethod
    def _normalise(cls, value):  # noqa: ANN001
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("username")
    @classmethod
    def _check_characters(cls, value: str) -> str:
        if not USERNAME_RE.fullmatch(value):
            raise ValueError("username may only contain letters, numbers and underscores")
        if value.startswith("bot_"):
            raise ValueError("usernames starting with 'bot_' are reserved")
        return value


class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=32)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("username", mode="before")
    @classmethod
    def _normalise(cls, value):  # noqa: ANN001
        return value.strip().lower() if isinstance(value, str) else value


class UserOut(BaseModel):
    id: int
    username: str


class MeOut(UserOut):
    balance: int


class TransactionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    amount: int
    balance_after: int
    kind: str
    game_id: int | None
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def _as_utc(cls, value: datetime) -> datetime:
        """SQLite returns naive datetimes; label them UTC so browsers parse them correctly."""
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
