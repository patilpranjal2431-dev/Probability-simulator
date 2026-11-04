"""Shared small types."""

from dataclasses import dataclass


@dataclass(frozen=True)
class UserRef:
    """Who is sitting at a table."""

    id: int
    name: str
    is_bot: bool = False


@dataclass(frozen=True)
class TableConfig:
    """Time limits and caps. Defaults suit real play; tests use tiny values."""

    turn_timeout: float = 30.0  # seconds to act before an automatic check/fold
    reconnect_grace: float = 30.0  # seconds a disconnected player has to come back
    max_consecutive_timeouts: int = 3  # this many timeouts in a row ends the match
    max_hands: int = 20
    max_strikes: int = 5  # malformed or illegal messages tolerated before disconnecting a client
    waiting_timeout: float = 120.0  # an unfilled table is removed after this long
    settle_retries: int = 3


class LobbyError(Exception):
    """A request to the lobby cannot be honoured (carries a machine-readable code)."""

    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code
        self.message = message or code


class TableError(Exception):
    """A move is rejected by the table (not your turn, illegal action, ...)."""

    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code
        self.message = message or code
