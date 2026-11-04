"""Bot accounts: each bot is a normal user (with a wallet) whose name starts with 'bot_'.

Humans cannot register such names (see schemas.RegisterIn), so a bot name can never be
impersonated. Bots cannot log in: their password hash is not a valid hash.
"""

from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.games.bots import BOT_NAMES
from app.models import User
from app.services.wallet import create_user
from app.tables.common import UserRef


def bot_username(bot_name: str) -> str:
    return f"bot_{bot_name}"


def ensure_bot_users(db: Session) -> None:
    """Create any missing bot accounts (each starts with the standard coin balance)."""
    for name in BOT_NAMES:
        existing = db.execute(select(User).where(User.username == bot_username(name))).scalar_one_or_none()
        if existing is None:
            create_user(db, bot_username(name), "!", is_bot=True)
        elif not existing.is_bot:
            raise RuntimeError(f"username {existing.username!r} is taken by a non-bot account")
    db.commit()


def make_bot_directory(session_factory: Callable[[], Session]) -> Callable[[str], UserRef]:
    """A function name -> UserRef for the lobby (it runs in a worker thread)."""

    def lookup(name: str) -> UserRef:
        with session_factory() as db:
            user = db.execute(select(User).where(User.username == bot_username(name))).scalar_one()
            return UserRef(user.id, user.username, is_bot=True)

    return lookup


def make_directory(session_factory: Callable[[], Session]) -> Callable[[], dict[int, tuple[str, bool]]]:
    """A function returning {user_id: (username, is_bot)} for every user."""

    def lookup() -> dict[int, tuple[str, bool]]:
        with session_factory() as db:
            return {u.id: (u.username, u.is_bot) for u in db.execute(select(User)).scalars()}

    return lookup
