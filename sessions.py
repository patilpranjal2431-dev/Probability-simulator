"""Server-side sessions. The browser holds a random token; the database holds only its hash.

Because sessions live in the database, logging out really revokes them (unlike a
stateless token that stays valid until it expires).
"""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import User, UserSession, utcnow


def hash_token(token: str) -> str:
    """SHA-256 of the token. (A fast hash is fine here: the token is 256 random bits.)"""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def create_session(db: Session, user_id: int) -> str:
    """Create a session and return the RAW token (shown to the browser once, never stored)."""
    token = secrets.token_urlsafe(32)
    db.add(
        UserSession(
            token_hash=hash_token(token),
            user_id=user_id,
            expires_at=utcnow() + timedelta(hours=settings.session_ttl_hours),
        )
    )
    db.flush()
    return token


def get_user_for_token(db: Session, token: str) -> User | None:
    """The user this token belongs to, or None if unknown or expired."""
    row = db.execute(
        select(UserSession).where(UserSession.token_hash == hash_token(token))
    ).scalar_one_or_none()
    if row is None or _as_utc(row.expires_at) <= utcnow():
        return None
    return db.get(User, row.user_id)


def delete_session(db: Session, token: str) -> None:
    """Revoke one session (logout). Unknown tokens are ignored."""
    db.execute(
        delete(UserSession).where(UserSession.token_hash == hash_token(token)),
        execution_options={"synchronize_session": False},
    )


def delete_expired_sessions(db: Session) -> int:
    """Housekeeping: remove expired sessions and return how many were removed."""
    result = db.execute(
        delete(UserSession).where(UserSession.expires_at <= utcnow()),
        execution_options={"synchronize_session": False},
    )
    return result.rowcount
