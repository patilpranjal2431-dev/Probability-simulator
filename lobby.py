"""Lobby: creates tables, lists the open ones, and enforces ONE table per user.

Flow:  create_table()  ->  host connects  ->  (human table) a second user connects, or
       (bot table) the match starts as soon as the host connects.
"""

import asyncio
import random
import secrets
from collections.abc import Callable

from app.games.base import secure_rng
from app.games.bots import BOT_NAMES
from app.games.registry import buy_in_limits, make_game

from .common import LobbyError, TableConfig, UserRef
from .ledger import Ledger_like
from .table import Connection, Table


class Lobby:
    def __init__(
        self,
        ledger: Ledger_like,
        bot_directory: Callable[[str], UserRef],
        config: TableConfig | None = None,
        rng_factory: Callable[[], random.Random] = secure_rng,
        grade_store=None,
    ) -> None:
        self.ledger = ledger
        self.grade_store = grade_store
        self.bot_directory = bot_directory  # bot name -> the bot's user account
        self.config = config or TableConfig()
        self.rng_factory = rng_factory  # secure_rng for real play; a seeded one in tests
        self._tables: dict[str, Table] = {}
        self._user_table: dict[int, str] = {}

    # ------------------------------------------------------------------ queries

    def get(self, table_id: str) -> Table | None:
        return self._tables.get(table_id)

    def table_of(self, user_id: int) -> Table | None:
        table_id = self._user_table.get(user_id)
        return self._tables.get(table_id) if table_id else None

    def list_open(self) -> list[dict]:
        """Human tables waiting for an opponent (their host is connected)."""
        return [table.info() for table in self._tables.values() if table.joinable]

    # ----------------------------------------------------------------- commands

    async def create_table(
        self, user: UserRef, game_name: str, buy_in: int | None = None, bot: str | None = None
    ) -> Table:
        """Open a table. Nothing is escrowed yet: coins move only when the match starts."""
        if user.id in self._user_table:
            raise LobbyError("already_seated", "you are already at a table")
        try:
            game = make_game(game_name)
        except ValueError as error:
            raise LobbyError("unknown_game", str(error)) from None
        low, high = buy_in_limits(game)
        buy_in = high if buy_in is None else buy_in
        if isinstance(buy_in, bool) or not isinstance(buy_in, int) or not low <= buy_in <= high:
            raise LobbyError("bad_buy_in", f"buy-in must be a whole number from {low} to {high}")
        bot_user = None
        if bot is not None:
            if bot not in BOT_NAMES:
                raise LobbyError("unknown_bot", f"choose a bot from {', '.join(BOT_NAMES)}")
            bot_user = await asyncio.to_thread(self.bot_directory, bot)
        if await asyncio.to_thread(self.ledger.balance, user.id) < buy_in:
            raise LobbyError("insufficient_funds", "your balance is below the buy-in")

        table_id = secrets.token_urlsafe(6)
        table = Table(
            table_id, game_name, game, buy_in, user, bot, bot_user,
            self.ledger, self.config, self.rng_factory, self._on_closed, self.grade_store,
        )
        self._tables[table_id] = table
        self._user_table[user.id] = table_id
        return table

    async def connect(self, user: UserRef, table_id: str, conn: Connection) -> Table:
        """Attach a user's connection to a table (as host, as the second player, or reconnecting)."""
        table = self._tables.get(table_id)
        if table is None:
            raise LobbyError("no_such_table", "no such table")
        registered_here = False
        if not table.has_user(user.id):
            if user.id in self._user_table:
                raise LobbyError("already_seated", "you are already at another table")
            self._user_table[user.id] = table_id  # reserve before awaiting, so it cannot be raced
            registered_here = True
        try:
            await table.connect(user, conn)
        except LobbyError:
            if registered_here:
                self._user_table.pop(user.id, None)
            raise
        return table

    async def close_all(self) -> None:
        for table in list(self._tables.values()):
            await table.close()
        self._tables.clear()
        self._user_table.clear()

    # ----------------------------------------------------------------- internals

    def _on_closed(self, table: Table) -> None:
        self._tables.pop(table.id, None)
        for user in (table.host, table.guest):
            if user is not None and self._user_table.get(user.id) == table.id:
                del self._user_table[user.id]
