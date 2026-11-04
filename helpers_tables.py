"""Helpers for table tests: simulated clients and a ready-made lobby."""

import asyncio
import copy
import itertools
import random

from app.tables.common import TableConfig, UserRef
from app.tables.ledger import MemoryLedger
from app.tables.lobby import Lobby

ALICE, BOB, CAROL = UserRef(1, "alice"), UserRef(2, "bob"), UserRef(3, "carol")
BOT_USER_ID = 99
START = {1: 1000, 2: 1000, 3: 1000, BOT_USER_ID: 1000}


class SimClient:
    """A simulated player connection: records everything the server sends it."""

    def __init__(self, name: str = "client") -> None:
        self.name = name
        self.inbox: list[dict] = []
        self.closed: int | None = None
        self.fail_sends = False  # pretend the network died

    async def send(self, message: dict) -> None:
        if self.fail_sends or self.closed is not None:
            raise ConnectionError("connection is dead")
        self.inbox.append(copy.deepcopy(message))

    async def close(self, code: int = 1000) -> None:
        self.closed = code

    def of_type(self, kind: str) -> list[dict]:
        return [m for m in self.inbox if m["type"] == kind]

    def events(self, name: str) -> list[dict]:
        return [m for m in self.inbox if m["type"] == "event" and m["event"] == name]

    def errors(self) -> list[str]:
        return [m["code"] for m in self.of_type("error")]

    @property
    def state(self) -> dict | None:
        states = self.of_type("state")
        return states[-1] if states else None

    @property
    def match_over(self) -> dict | None:
        over = self.of_type("match_over")
        return over[0] if over else None

    def my_turn(self) -> bool:
        return bool(self.state and self.state["decision"])


def run(coroutine):
    """Run an async test body from a plain (synchronous) test function."""
    return asyncio.run(coroutine)


async def wait_for(predicate, timeout: float = 5.0, what: str = "condition") -> None:
    """Poll until `predicate()` is true. Generous timeout, so slow machines do not flake."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.005)


def make_lobby(balances=None, ledger=None, grade_store=None, **config):
    """A lobby with an in-memory ledger, one bot account, tiny time limits and seeded dice."""
    ledger = ledger or MemoryLedger(dict(START if balances is None else balances))
    defaults = dict(turn_timeout=30.0, reconnect_grace=30.0, max_hands=3, max_strikes=3, waiting_timeout=30.0)
    seeds = itertools.count(1)
    lobby = Lobby(
        ledger,
        lambda name: UserRef(BOT_USER_ID, f"bot_{name}", True),
        TableConfig(**{**defaults, **config}),
        lambda: random.Random(next(seeds)),
        grade_store=grade_store,
    )
    return lobby, ledger


def passive_action(client: SimClient) -> str:
    """The simplest move: check when you can, call when you must."""
    return "check" if client.state["decision"]["kind"] == "bet" else "call"


async def act(table, user, client: SimClient, action: str, amount: int | None = None) -> None:
    import json

    message = {"type": "action", "action": action}
    if amount is not None:
        message["amount"] = amount
    await table.handle(user.id, json.dumps(message))


async def play_passively(table, humans: list[tuple], limit: int = 500) -> None:
    """Both (or the one) human(s) check/call until the match ends."""
    for _ in range(limit):
        if table.state in ("closed", "settling"):
            return
        for user, client in humans:
            if table.state == "playing" and client.my_turn():
                await act(table, user, client, passive_action(client))
        await asyncio.sleep(0)
    raise AssertionError("match did not finish")
