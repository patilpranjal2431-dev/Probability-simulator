"""Money movement for tables: escrow, settle, refund.

Rules (identical for both implementations, and enforced by one shared test-suite):
  * escrow: debit EVERY player's buy-in or none of them (all-or-nothing).
  * settle: pay out exactly what was escrowed, to exactly the players who escrowed, ONCE.
            The payouts must add up to the escrow, so coins are never created or destroyed.
  * refund_unfinished: after a server restart, give back the escrow of any match that never settled.

`MemoryLedger` (this file, standard library only) is for tests; `SqlLedger` (sql_ledger.py)
uses the database and the Phase 1 wallet service.
"""

import threading
from typing import Protocol


class LedgerError(Exception):
    """Base class for ledger problems."""


class InsufficientFunds(LedgerError):
    def __init__(self, user_id: int):
        super().__init__(f"user {user_id} cannot cover the buy-in")
        self.user_id = user_id


class UnknownUser(LedgerError):
    def __init__(self, user_id: int):
        super().__init__(f"user {user_id} has no wallet")
        self.user_id = user_id


class AlreadySettled(LedgerError):
    """This match was already settled (or refunded); paying again would create coins."""


class SettlementMismatch(LedgerError):
    """The payouts do not match what was escrowed."""


class Ledger_like(Protocol):
    """What the Table needs from a ledger (MemoryLedger and SqlLedger both provide it)."""

    def balance(self, user_id: int) -> int: ...

    def escrow(self, game_name: str, buy_ins: dict[int, int]) -> int: ...

    def settle(self, game_ref: int, payouts: dict[int, int]) -> dict[int, int]: ...

    def refund_unfinished(self) -> int: ...

    def net_coin_rows(self) -> dict[int, int]: ...


def _check_amounts(amounts: dict[int, int], what: str, allow_zero: bool) -> None:
    if not amounts:
        raise LedgerError(f"{what} must not be empty")
    for user_id, amount in amounts.items():
        if isinstance(amount, bool) or not isinstance(amount, int):
            raise LedgerError(f"{what} for user {user_id} must be a whole number")
        if amount < 0 or (amount == 0 and not allow_zero):
            raise LedgerError(f"{what} for user {user_id} must be {'non-negative' if allow_zero else 'positive'}")


# ---------------------------------------------------------------------- in memory


class MemoryLedger:
    """Thread-safe in-memory ledger with the same behaviour as SqlLedger. For tests."""

    def __init__(self, balances: dict[int, int]) -> None:
        self._balances = dict(balances)
        self._games: dict[int, dict] = {}
        self._next_ref = 1
        self._lock = threading.Lock()
        self.transactions: list[tuple[int, int, str, int, int]] = []  # (user, amount, kind, ref, balance_after)

    def balance(self, user_id: int) -> int:
        with self._lock:
            if user_id not in self._balances:
                raise UnknownUser(user_id)
            return self._balances[user_id]

    def escrow(self, game_name: str, buy_ins: dict[int, int]) -> int:
        _check_amounts(buy_ins, "buy-in", allow_zero=False)
        with self._lock:
            for user_id, amount in sorted(buy_ins.items()):  # check everything BEFORE changing anything
                if user_id not in self._balances:
                    raise UnknownUser(user_id)
                if self._balances[user_id] < amount:
                    raise InsufficientFunds(user_id)
            ref = self._next_ref
            self._next_ref += 1
            for user_id, amount in sorted(buy_ins.items()):
                self._balances[user_id] -= amount
                self.transactions.append((user_id, -amount, "bet", ref, self._balances[user_id]))
            self._games[ref] = {"game": game_name, "buy_ins": dict(buy_ins), "status": "in_progress"}
            return ref

    def settle(self, game_ref: int, payouts: dict[int, int]) -> dict[int, int]:
        _check_amounts(payouts, "payout", allow_zero=True)
        with self._lock:
            game = self._games.get(game_ref)
            if game is None:
                raise LedgerError(f"unknown game {game_ref}")
            if game["status"] != "in_progress":
                raise AlreadySettled(f"game {game_ref} is already {game['status']}")
            if set(payouts) != set(game["buy_ins"]) or sum(payouts.values()) != sum(game["buy_ins"].values()):
                raise SettlementMismatch("payouts must go to the escrowing players and equal the escrow")
            for user_id, amount in sorted(payouts.items()):
                if amount > 0:
                    self._balances[user_id] += amount
                    self.transactions.append((user_id, amount, "payout", game_ref, self._balances[user_id]))
            game["status"] = "finished"
            return {user_id: self._balances[user_id] for user_id in payouts}

    def refund_unfinished(self) -> int:
        refunded = 0
        with self._lock:
            for ref, game in self._games.items():
                if game["status"] != "in_progress":
                    continue
                for user_id, amount in sorted(game["buy_ins"].items()):
                    self._balances[user_id] += amount
                    self.transactions.append((user_id, amount, "refund", ref, self._balances[user_id]))
                game["status"] = "aborted"
                refunded += 1
        return refunded

    def game_status(self, game_ref: int) -> str:
        with self._lock:
            return self._games[game_ref]["status"]

    def net_coin_rows(self) -> dict[int, int]:
        """Coins won (+) or lost (-) at tables, per user, counting only matches that are over.

        A match still in progress is ignored: its buy-in is escrowed, not lost."""
        net: dict[int, int] = {}
        with self._lock:
            for user_id, amount, kind, ref, _after in self.transactions:
                if kind in ("bet", "payout", "refund") and self._games[ref]["status"] != "in_progress":
                    net[user_id] = net.get(user_id, 0) + amount
        return net
