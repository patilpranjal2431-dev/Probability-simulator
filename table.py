"""Table: the async runtime of one table.

Phases:  waiting  ->  playing  ->  closed          (or  waiting -> closed  if cancelled)

Concurrency: every public method takes ONE lock, so a message, a timer firing and a
disconnect can never run at the same time on the same table.

Time rules (see TableConfig):
  * a connected player who does not act within `turn_timeout` is auto-played (fold if facing
    a bet, otherwise check). `max_consecutive_timeouts` in a row ends the match.
  * a player who disconnects while the other human is waiting is auto-played immediately, so
    nobody is stuck waiting. They have `reconnect_grace` seconds to come back; after that the
    match is abandoned (they lose the chips in the current hand, and keep the rest).
  * a client that keeps sending garbage (`max_strikes`) is disconnected.

Money rules: both buy-ins are escrowed atomically when the match starts, and settled in one
atomic step when it ends (see ledger.py). A match that cannot settle leaves its escrow in
place for `refund_unfinished` to return on the next server start.
"""

import asyncio
import logging
import random
from collections.abc import Callable
from typing import Protocol

from app.games.base import Game
from app.games.bots import make_bot
from app.grading.grade import grade_match
from app.grading.stats import aggregate

from .common import LobbyError, TableConfig, TableError, UserRef
from .core import Seat, TableCore
from .ledger import InsufficientFunds, Ledger_like
from .protocol import ProtocolError, parse_action, parse_client_message

log = logging.getLogger("edge_lab.tables")


class Connection(Protocol):
    """What the table needs from a client connection (a WebSocket, or a simulated client)."""

    async def send(self, message: dict) -> None: ...

    async def close(self, code: int = 1000) -> None: ...


class Table:
    def __init__(
        self,
        table_id: str,
        game_name: str,
        game: Game,
        buy_in: int,
        host: UserRef,
        bot_name: str | None,
        bot_user: UserRef | None,
        ledger: Ledger_like,
        config: TableConfig,
        rng_factory: Callable[[], random.Random],
        on_closed: Callable[["Table"], None],
        grade_store=None,
    ) -> None:
        self.id = table_id
        self.game_name = game_name
        self.game = game
        self.buy_in = buy_in
        self.host = host
        self.bot_name = bot_name
        self.guest: UserRef | None = bot_user  # a bot table is "full" from the start
        self.ledger = ledger
        self.config = config
        self._rng_factory = rng_factory
        self._on_closed = on_closed
        self.grade_store = grade_store  # optional: where graded decisions are saved

        self.state = "waiting"  # waiting | playing | settling | closed
        self.core: TableCore | None = None
        self.users: tuple[UserRef, UserRef] | None = None
        self.seat_of: dict[int, int] = {}
        self.conns: dict[int, Connection] = {}
        self.escrow_ref: int | None = None
        self.close_reason: str | None = None

        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()
        self._turn_token = 0
        self._turn_task: asyncio.Task | None = None
        self._armed_for: tuple | None = None
        self._deadline_key: tuple | None = None  # which turn the absolute deadline belongs to
        self._deadline_value: float = 0.0
        self._turn_deadline: float | None = None
        self._grace_tasks: dict[int, asyncio.Task] = {}
        self._dead: set[int] = set()
        self._strikes: dict[int, int] = {}
        self._timeouts: dict[int, int] = {}
        self._seq = 0
        self._spawn(self._expire_waiting(config.waiting_timeout))

    # ------------------------------------------------------------------ queries

    def info(self) -> dict:
        return {
            "id": self.id,
            "game": self.game_name,
            "buy_in": self.buy_in,
            "host": self.host.name,
            "vs_bot": self.bot_name,
            "state": self.state,
        }

    def has_user(self, user_id: int) -> bool:
        return user_id == self.host.id or (self.guest is not None and user_id == self.guest.id)

    @property
    def joinable(self) -> bool:
        """A human table whose host is connected and which still has a free seat."""
        return (
            self.state == "waiting"
            and self.bot_name is None
            and self.guest is None
            and self.host.id in self.conns
        )

    # --------------------------------------------------------------- public API

    async def connect(self, user: UserRef, conn: Connection) -> None:
        """A user's WebSocket arrives. Raises LobbyError if they cannot take part."""
        async with self._lock:
            if self.state in ("settling", "closed"):
                raise LobbyError("table_closed", "this table has finished")

            if self.state == "waiting":
                if user.id == self.host.id:
                    self.conns[user.id] = conn
                    if self.bot_name is not None:
                        await self._start_locked(user)
                    else:
                        await self._send(user.id, {"type": "waiting", "table": self.id})
                    await self._drop_dead_locked()
                    return
                if self.bot_name is not None or self.guest is not None:
                    raise LobbyError("table_full", "this table is full")
                if self.host.id not in self.conns:
                    raise LobbyError("host_not_connected", "the host has not connected yet")
                self.guest = user
                self.conns[user.id] = conn
                await self._start_locked(user)
                await self._drop_dead_locked()
                return

            # playing: only the two seated users may (re)connect
            if user.id not in self.seat_of:
                raise LobbyError("table_full", "this table is full")
            old = self.conns.get(user.id)
            self.conns[user.id] = conn
            self._cancel(self._grace_tasks.pop(user.id, None))
            if old is not None and old is not conn:
                await self._close_conn(old, 4000)  # replaced by the newer connection
            seat = self.seat_of[user.id]
            self._arm_turn_timer()  # keeps the existing deadline if this turn's clock is already running
            await self._send(user.id, self._state_message(seat))
            other = self.users[1 - seat]
            await self._send(other.id, {"type": "event", "event": "opponent_reconnected"})
            await self._drop_dead_locked()

    async def disconnect(self, user_id: int, conn: Connection | None = None) -> None:
        """A user's connection went away. Pass `conn` so a replaced old socket is ignored."""
        async with self._lock:
            current = self.conns.get(user_id)
            if current is None or (conn is not None and current is not conn):
                return
            self.conns.pop(user_id, None)
            await self._on_disconnected_locked(user_id)
            await self._drop_dead_locked()

    async def handle(self, user_id: int, raw: str | bytes) -> None:
        """A message from a connected user."""
        async with self._lock:
            if user_id not in self.conns:
                return
            if self.state != "playing":
                await self._send(user_id, {"type": "error", "code": "not_started", "message": "no match in progress"})
                return
            try:
                data = parse_client_message(raw)
            except ProtocolError as error:
                await self._reject_locked(user_id, error.code, error.message)
                return

            kind = data["type"]
            if kind == "ping":
                await self._send(user_id, {"type": "pong"})
            elif kind == "leave":
                await self._publish_locked(self.core.concede(self.seat_of[user_id], "left"))
            else:  # "action"
                try:
                    action = parse_action(data)
                    events = self.core.submit(self.seat_of[user_id], action)
                except (ProtocolError, TableError) as error:
                    await self._reject_locked(user_id, error.code, error.message)
                    return
                self._timeouts[user_id] = 0
                await self._publish_locked(events)
            await self._drop_dead_locked()

    async def close(self) -> None:
        """Stop all timers (server shutdown / test cleanup). Does not settle: any escrow is
        returned by `refund_unfinished` on the next start."""
        async with self._lock:
            for task in list(self._tasks):
                self._cancel(task)
            self.state = "closed"

    # --------------------------------------------------------------- start / end

    async def _start_locked(self, caller: UserRef) -> None:
        host, guest = self.host, self.guest
        try:
            self.escrow_ref = await asyncio.to_thread(
                self.ledger.escrow, self.game_name, {host.id: self.buy_in, guest.id: self.buy_in}
            )
        except InsufficientFunds as error:
            await self._start_failed_locked(error.user_id, caller)
            return

        bot = make_bot(self.bot_name) if self.bot_name else None
        self.users = (host, guest)
        self.seat_of = {host.id: 0, guest.id: 1}
        self.core = TableCore(
            self.game,
            (Seat(host.id, host.name), Seat(guest.id, guest.name, bot=bot)),
            self.buy_in,
            rng=self._rng_factory(),
            bot_rngs=(self._rng_factory(), self._rng_factory()),
            max_hands=self.config.max_hands,
        )
        self.state = "playing"
        await self._publish_locked(self.core.opening_events)  # also sends everyone their first state

    async def _start_failed_locked(self, broke_user_id: int, caller: UserRef) -> None:
        """Escrow failed. Nothing was debited (it is all-or-nothing). Tell each side the truth."""
        short = {"type": "error", "code": "insufficient_funds", "message": "you cannot cover the buy-in"}
        if broke_user_id == self.host.id:
            await self._send(self.host.id, short)
            await self._close_locked("host_insufficient_funds")
            if caller.id == self.host.id:
                raise LobbyError("insufficient_funds", "you cannot cover the buy-in")
            raise LobbyError("host_unavailable", "the host can no longer cover the buy-in")
        if self.bot_name is not None:
            await self._send(self.host.id, {"type": "error", "code": "bot_unavailable",
                                            "message": "that bot cannot cover the buy-in"})
            await self._close_locked("bot_unavailable")
            raise LobbyError("bot_unavailable", "that bot cannot cover the buy-in")
        # the guest is short: drop them, keep the table open for someone else
        guest = self.guest
        self.guest = None
        conn = self.conns.pop(guest.id, None)
        if conn is not None:
            await self._send_raw(conn, short)
        await self._send(self.host.id, {"type": "event", "event": "guest_could_not_join"})
        raise LobbyError("insufficient_funds", "you cannot cover the buy-in")

    async def _finish_locked(self, over: dict) -> None:
        """Settle the money, tell the players, close the table."""
        self.state = "settling"
        self._cancel_timers()
        payouts = {user.id: over["final_stacks"][seat] for seat, user in enumerate(self.users)}
        balances = None
        for attempt in range(1, self.config.settle_retries + 1):
            try:
                balances = await asyncio.to_thread(self.ledger.settle, self.escrow_ref, payouts)
                break
            except Exception:  # noqa: BLE001
                log.exception("settlement attempt %d failed for table %s", attempt, self.id)
                await asyncio.sleep(0.05 * attempt)

        grading = await self._grade_locked() if balances is not None else {}

        for seat, user in enumerate(self.users):
            if user.id not in self.conns:
                continue
            if balances is None:
                await self._send(user.id, {"type": "error", "code": "settlement_failed",
                                           "message": "your coins are held safely and will be returned"})
            else:
                await self._send(user.id, {
                    "type": "match_over",
                    "reason": over["reason"],
                    "hands": over["hands"],
                    "final_stacks": over["final_stacks"],
                    "conceded_by": over["conceded_by"],
                    "your_payout": payouts[user.id],
                    "net_change": payouts[user.id] - self.buy_in,
                    "your_balance": balances[user.id],
                    "grading": grading.get(user.id),
                })
        await self._close_locked(over["reason"] if balances is not None else "settlement_failed")

    async def _grade_locked(self) -> dict[int, dict]:
        """Grade and store every human's decisions. Runs AFTER the money is settled, and a
        failure here only means no grades: it can never touch anyone's coins."""
        try:
            return await asyncio.to_thread(self._grade_and_store)
        except Exception:  # noqa: BLE001
            log.exception("grading failed for table %s", self.id)
            return {}

    def _grade_and_store(self) -> dict[int, dict]:
        """(runs in a worker thread) -> {user_id: summary}"""
        graded = grade_match(self.game_name, self.core.decision_log, self.core.results)
        summaries: dict[int, dict] = {}
        for seat, user in enumerate(self.users):
            if self.core.seats[seat].is_bot:
                continue
            rows = graded[seat]
            if self.grade_store is not None:
                self.grade_store.save(self.escrow_ref, user.id, rows)
            totals = aggregate((user.id, d) for d in rows).get(user.id)
            summaries[user.id] = {
                "decisions": len(rows),
                "hands_counted": totals.hands if totals else 0,
                "ev_earned": round(totals.ev_earned, 2) if totals else 0.0,
                "regret": round(totals.regret, 2) if totals else 0.0,
                "luck": round(totals.luck, 2) if totals else 0.0,
            }
        return summaries

    async def _close_locked(self, reason: str) -> None:
        self.state = "closed"
        self.close_reason = reason
        self._cancel_timers()
        for task in list(self._tasks):
            if task is not asyncio.current_task():
                self._cancel(task)
        for conn in list(self.conns.values()):
            await self._close_conn(conn, 1000)
        self.conns.clear()
        self._on_closed(self)

    # -------------------------------------------------------- publishing events

    async def _publish_locked(self, events: list[dict]) -> None:
        """Send events and fresh state to everyone, finish the match if it ended."""
        await self._emit_locked(events)
        if self.state == "playing":
            await self._auto_loop_locked()
            self._arm_turn_timer()

    async def _emit_locked(self, events: list[dict]) -> None:
        over = None
        for event in events:
            if event["type"] == "match_over":
                over = event
            else:
                await self._broadcast(event)
        if over is not None:
            await self._finish_locked(over)
            return
        self._arm_turn_timer()  # before building the states, so they carry the deadline
        for seat, user in enumerate(self.users):
            await self._send(user.id, self._state_message(seat))

    async def _auto_loop_locked(self) -> None:
        """Keep a disconnected player's turns moving while the other human waits."""
        while self.state == "playing":
            seat = self.core.to_act
            if seat is None:
                return
            me, other = self.users[seat], self.users[1 - seat]
            other_waiting = not self.core.seats[1 - seat].is_bot and other.id in self.conns
            if me.id in self.conns or not other_waiting:
                return
            await self._emit_locked(self.core.auto_play(seat, "disconnected"))

    # ------------------------------------------------------------ disconnections

    async def _on_disconnected_locked(self, user_id: int) -> None:
        if self.state == "waiting":
            if user_id == self.host.id:
                await self._close_locked("host_left")
            return
        if self.state != "playing":
            return
        seat = self.seat_of[user_id]
        other = self.users[1 - seat]
        await self._send(other.id, {"type": "event", "event": "opponent_disconnected",
                                    "grace_s": self.config.reconnect_grace})
        self._cancel(self._grace_tasks.pop(user_id, None))
        self._grace_tasks[user_id] = self._spawn(self._grace_expired(user_id, self.config.reconnect_grace))
        await self._auto_loop_locked()
        if self.state == "playing":
            self._arm_turn_timer()

    async def _drop_dead_locked(self) -> None:
        """Handle connections that failed while we were sending to them."""
        while self._dead:
            user_id = self._dead.pop()
            self.conns.pop(user_id, None)
            await self._on_disconnected_locked(user_id)

    async def _reject_locked(self, user_id: int, code: str, message: str) -> None:
        """Tell the client its message was refused. Too many refusals disconnect it."""
        self._strikes[user_id] = self._strikes.get(user_id, 0) + 1
        await self._send(user_id, {"type": "error", "code": code, "message": message})
        if self._strikes[user_id] >= self.config.max_strikes:
            conn = self.conns.pop(user_id, None)
            if conn is not None:
                await self._close_conn(conn, 1008)
            await self._on_disconnected_locked(user_id)
        await self._drop_dead_locked()

    # ------------------------------------------------------------------- timers

    def _arm_turn_timer(self) -> None:
        """Start the clock for the player to act.

        The deadline is fixed once per turn, in absolute time. It keeps running while the player
        is disconnected and is resumed (with whatever time is left) when they come back, so
        neither reconnecting nor any other event can buy extra time."""
        if self.state != "playing" or self.core.to_act is None:
            self._cancel_turn_timer()
            return
        seat = self.core.to_act
        user = self.users[seat]
        if self.core.seats[seat].is_bot or user.id not in self.conns:
            self._cancel_turn_timer()
            return
        key = (self.core.hands_played, len(self.core.hand.history), seat)
        loop = asyncio.get_running_loop()
        if key != self._deadline_key:  # a new turn: its deadline is fixed ONCE, in absolute time
            self._deadline_key = key
            self._deadline_value = loop.time() + self.config.turn_timeout
        if key == self._armed_for and self._turn_task is not None and not self._turn_task.done():
            return
        self._cancel_turn_timer()
        self._armed_for = key
        self._turn_token += 1
        self._turn_deadline = self._deadline_value
        remaining = max(0.0, self._deadline_value - loop.time())
        self._turn_task = self._spawn(self._turn_timer(self._turn_token, remaining))

    def _cancel_turn_timer(self) -> None:
        self._turn_token += 1  # any timer already waiting becomes stale
        self._turn_deadline = None
        self._armed_for = None
        if self._turn_task is not None and self._turn_task is not asyncio.current_task():
            self._cancel(self._turn_task)
        self._turn_task = None

    def _cancel_timers(self) -> None:
        self._cancel_turn_timer()
        for task in list(self._grace_tasks.values()):
            if task is not asyncio.current_task():
                self._cancel(task)
        self._grace_tasks.clear()

    async def _turn_timer(self, token: int, delay: float) -> None:
        await asyncio.sleep(delay)
        async with self._lock:
            if token != self._turn_token or self.state != "playing":
                return
            seat = self.core.to_act
            if seat is None:
                return
            user = self.users[seat]
            self._timeouts[user.id] = self._timeouts.get(user.id, 0) + 1
            events = self.core.auto_play(seat, "timeout")
            if self._timeouts[user.id] >= self.config.max_consecutive_timeouts:
                events += self.core.concede(seat, "timeouts")
            await self._publish_locked(events)
            await self._drop_dead_locked()

    async def _grace_expired(self, user_id: int, delay: float) -> None:
        await asyncio.sleep(delay)
        async with self._lock:
            if self.state != "playing" or user_id in self.conns:
                return
            await self._publish_locked(self.core.concede(self.seat_of[user_id], "disconnected"))
            await self._drop_dead_locked()

    async def _expire_waiting(self, delay: float) -> None:
        await asyncio.sleep(delay)
        async with self._lock:
            if self.state != "waiting":
                return
            await self._broadcast({"type": "error", "code": "table_expired", "message": "nobody joined in time"})
            await self._close_locked("expired")

    # ----------------------------------------------------------------- messaging

    def _state_message(self, seat: int) -> dict:
        self._seq += 1
        message = self.core.view_for(seat)
        opponent_seat = 1 - seat
        opponent = self.users[opponent_seat]
        message["seq"] = self._seq
        message["opponent"] = {
            "name": opponent.name,
            "is_bot": self.core.seats[opponent_seat].is_bot,
            "connected": self.core.seats[opponent_seat].is_bot or opponent.id in self.conns,
        }
        remaining = None
        if self._turn_deadline is not None and self.core.to_act == seat:
            remaining = max(0.0, self._turn_deadline - asyncio.get_running_loop().time())
        message["deadline_s"] = remaining
        return message

    async def _broadcast(self, message: dict) -> None:
        for user_id in list(self.conns):
            await self._send(user_id, message)

    async def _send(self, user_id: int, message: dict) -> None:
        conn = self.conns.get(user_id)
        if conn is not None:
            await self._send_raw(conn, message, user_id)

    async def _send_raw(self, conn: Connection, message: dict, user_id: int | None = None) -> None:
        try:
            await conn.send(message)
        except Exception:  # noqa: BLE001  (a dead socket must never break the table)
            if user_id is not None:
                self._dead.add(user_id)

    @staticmethod
    async def _close_conn(conn: Connection, code: int) -> None:
        try:
            await conn.close(code)
        except Exception:  # noqa: BLE001
            pass

    def _spawn(self, coroutine) -> asyncio.Task:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    @staticmethod
    def _cancel(task: asyncio.Task | None) -> None:
        if task is not None and not task.done():
            task.cancel()
