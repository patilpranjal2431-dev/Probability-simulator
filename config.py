"""Application settings, read once from environment variables at import time."""

import os
from dataclasses import dataclass


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Override with environment variables."""

    database_url: str = os.environ.get("DATABASE_URL", "sqlite:///./edge_lab.db")
    # Every new wallet starts with this many VIRTUAL coins (no real money).
    starting_coins: int = 1000
    # Sessions
    session_cookie_name: str = "edge_session"
    session_ttl_hours: int = 24 * 7
    # Set COOKIE_SECURE=1 in production (HTTPS). Browsers will not send Secure
    # cookies over plain http, so it is off by default for local development.
    cookie_secure: bool = _env_flag("COOKIE_SECURE")


settings = Settings()
