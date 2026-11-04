"""Build the production Lobby (database ledger + bot accounts)."""

import random
from collections.abc import Callable

from sqlalchemy.orm import Session

from app.games.base import secure_rng
from app.services.bots import make_bot_directory

from .common import TableConfig
from .lobby import Lobby
from .sql_ledger import SqlLedger


def build_lobby(
    session_factory: Callable[[], Session],
    config: TableConfig | None = None,
    rng_factory: Callable[[], random.Random] = secure_rng,
    ledger: SqlLedger | None = None,
    grade_store=None,
) -> Lobby:
    return Lobby(
        ledger or SqlLedger(session_factory),
        make_bot_directory(session_factory),
        config,
        rng_factory,
        grade_store=grade_store,
    )
