"""Game engine core: actions, what a player is allowed to see, and the hand state machine.

Fairness rules built into the design
* A bot only ever receives a `Decision`: its OWN odds, the stacks, the pot and the legal bet
  range. It never receives the opponent's hidden value.
* `Hand.public_state()` is safe to send to either player. Hidden values are only exposed
  through `private_view(seat)` (your own) or `HandResult.private_values` (server-side, after
  the hand).
* Randomness is injected (`rng`). Simulations pass a seeded `random.Random` (reproducible);
  live games should pass `secure_rng()` (not seedable, not predictable).
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

ACTION_KINDS = ("check", "bet", "call", "fold")


class IllegalActionError(ValueError):
    """The action is not allowed right now."""


def secure_rng() -> random.Random:
    """Cryptographically secure randomness (uses the OS entropy source). For live games."""
    return random.SystemRandom()


# ---------------------------------------------------------------------------- actions


@dataclass(frozen=True)
class Action:
    """What a player does: check, bet <amount>, call or fold."""

    kind: str
    amount: int = 0

    def __post_init__(self) -> None:
        if self.kind not in ACTION_KINDS:
            raise ValueError(f"unknown action kind: {self.kind!r}")
        if isinstance(self.amount, bool) or not isinstance(self.amount, int):
            raise TypeError("amount must be an integer")
        if self.kind == "bet" and self.amount < 1:
            raise ValueError("a bet must be at least 1 coin")
        if self.kind != "bet" and self.amount != 0:
            raise ValueError(f"{self.kind} takes no amount")


CHECK = Action("check")
CALL = Action("call")
FOLD = Action("fold")


def bet(amount: int) -> Action:
    return Action("bet", amount)


# -------------------------------------------------------------------------- decisions


@dataclass(frozen=True)
class Decision:
    """Everything a player is allowed to know when it is their turn.

    kind = "bet":      you may check, or bet an integer from min_bet to max_bet
                       (max_bet == 0 means you are short-stacked and can only check)
    kind = "respond":  you face a bet; call it or fold

    win/tie/lose_prob are your chances if the hand reaches showdown, assuming the
    opponent's hidden value is uniformly random given what you can see. They deliberately
    ignore what the opponent's ACTIONS reveal (a simple "naive equity" model).
    """

    kind: str
    stack: int  # chips behind (not yet in the pot)
    my_invested: int  # your chips already in the pot
    opp_invested: int  # opponent's chips already in the pot
    min_bet: int
    max_bet: int
    win_prob: float
    tie_prob: float
    lose_prob: float

    @property
    def to_call(self) -> int:
        return self.opp_invested - self.my_invested

    @property
    def wealth(self) -> int:
        """Chips you own right now, counting the ones you already put in the pot."""
        return self.stack + self.my_invested

    def can_bet(self) -> bool:
        return self.kind == "bet" and self.max_bet >= self.min_bet >= 1

    def is_legal(self, action: Action) -> bool:
        if self.kind == "bet":
            if action.kind == "check":
                return True
            return (
                action.kind == "bet"
                and self.can_bet()
                and self.min_bet <= action.amount <= self.max_bet
            )
        if action.kind == "fold":
            return True
        return action.kind == "call" and self.to_call <= self.stack


@dataclass(frozen=True)
class HandResult:
    net: tuple[int, int]  # chips won (+) or lost (-) by each seat; always sums to zero
    ended_by: str  # "fold" or "showdown"
    folded: int | None  # seat that folded, if any
    showdown_values: tuple[int, int] | None  # public: revealed only if there was a showdown
    private_values: tuple[int, int]  # SERVER-SIDE ONLY: never send these to a player mid-hand


# ------------------------------------------------------------------------------- hand


class Hand(ABC):
    """One hand of a heads-up game: a small state machine.

    Stages: "open" (first player may check/bet) -> maybe "second_open" -> "respond" -> done.
    """

    name = "hand"

    def __init__(self, ante: int, stacks: tuple[int, int], first: int, private: tuple[int, int]):
        if first not in (0, 1):
            raise ValueError("first must be 0 or 1")
        if min(stacks) < ante:
            raise ValueError("both players need at least the ante")
        self.ante = ante
        self.stack = [stacks[0] - ante, stacks[1] - ante]  # chips behind, after the ante
        self.invested = [ante, ante]
        self.first = first
        self.to_act = first
        self.stage = "open"
        self.history: list[tuple[int, Action]] = []
        self.result: HandResult | None = None
        self._private = private

    # ---- rules that differ between games
    @abstractmethod
    def _equity(self, seat: int) -> tuple[float, float, float]:
        """(win, tie, lose) chances for `seat` at showdown vs a uniformly random opponent value."""

    @abstractmethod
    def _bet_limits(self, cap: int) -> tuple[int, int]:
        """Allowed (min_bet, max_bet) when the smaller stack behind is `cap`."""

    @abstractmethod
    def _compare(self) -> int:
        """+1 if seat 0 wins at showdown, -1 if seat 1 wins, 0 for a tie."""

    @property
    @abstractmethod
    def _second_player_may_bet_after_check(self) -> bool: ...

    # ---- state
    @property
    def done(self) -> bool:
        return self.result is not None

    def decision(self) -> Decision:
        """The current player's view. Contains no information about the opponent's hidden value."""
        if self.done:
            raise IllegalActionError("the hand is over")
        seat, opp = self.to_act, 1 - self.to_act
        win, tie, lose = self._equity(seat)
        if self.stage in ("open", "second_open"):
            kind = "bet"
            low, high = self._bet_limits(min(self.stack))
        else:
            kind, low, high = "respond", 0, 0
        return Decision(
            kind, self.stack[seat], self.invested[seat], self.invested[opp], low, high, win, tie, lose
        )

    def act(self, action: Action) -> None:
        """Apply the current player's action, or raise IllegalActionError."""
        decision = self.decision()
        if not decision.is_legal(action):
            raise IllegalActionError(f"{action} is not legal for seat {self.to_act} ({decision.kind})")
        seat = self.to_act
        self.history.append((seat, action))
        if action.kind == "check":
            self._on_check()
        elif action.kind == "bet":
            self._pay(seat, action.amount)
            self.stage = "respond"
            self.to_act = 1 - seat
        elif action.kind == "call":
            self._pay(seat, decision.to_call)
            self._showdown()
        else:
            self._fold(seat)

    def concede(self, seat: int) -> None:
        """`seat` abandons the hand (leaves, or is removed): they lose what they have put in.

        Works at any point, even when it is not their turn, so a player who disappears
        cannot freeze the table.
        """
        if self.done:
            raise IllegalActionError("the hand is over")
        self._fold(seat)

    # ---- views
    def public_state(self) -> dict:
        """Safe to show to BOTH players: no hidden values (except revealed ones after a showdown)."""
        state = {
            "game": self.name,
            "stage": self.stage,
            "done": self.done,
            "to_act": None if self.done else self.to_act,
            "pot": sum(self.invested),
            "invested": tuple(self.invested),
            "stacks": tuple(self.stack),
            "history": [(seat, a.kind, a.amount) for seat, a in self.history],
        }
        if self.result is not None:
            state["ended_by"] = self.result.ended_by
            state["net"] = self.result.net
            state["showdown_values"] = self.result.showdown_values
        return state

    def private_view(self, seat: int) -> int:
        """Only the given seat's own hidden value."""
        return self._private[seat]

    # ---- internals
    def _pay(self, seat: int, amount: int) -> None:
        self.stack[seat] -= amount
        self.invested[seat] += amount

    def _on_check(self) -> None:
        if self.stage == "open" and self._second_player_may_bet_after_check:
            self.stage = "second_open"
            self.to_act = 1 - self.first
        else:
            self._showdown()

    def _showdown(self) -> None:
        assert self.invested[0] == self.invested[1], "showdown needs equal contributions"
        stake = self.invested[0]
        outcome = self._compare()
        self.result = HandResult(
            net=(stake * outcome, -stake * outcome),
            ended_by="showdown",
            folded=None,
            showdown_values=self._private,
            private_values=self._private,
        )

    def _fold(self, seat: int) -> None:
        lost = self.invested[seat]
        net = [0, 0]
        net[seat], net[1 - seat] = -lost, lost
        self.result = HandResult(
            net=(net[0], net[1]), ended_by="fold", folded=seat,
            showdown_values=None, private_values=self._private,
        )


class Game(ABC):
    """A game: knows its stakes and how to start a hand."""

    name = "game"
    ante = 1
    default_stack = 100

    @abstractmethod
    def all_deals(self) -> list[tuple[int, int]]:
        """Every equally likely deal (for exact analysis)."""

    @abstractmethod
    def _random_deal(self, rng: random.Random) -> tuple[int, int]: ...

    @abstractmethod
    def _make_hand(
        self, stacks: tuple[int, int], first: int, deal: tuple[int, int]
    ) -> Hand: ...

    def new_hand(
        self,
        rng: random.Random,
        stacks: tuple[int, int],
        first: int,
        deal: tuple[int, int] | None = None,
    ) -> Hand:
        """Start a hand. `deal` is for tests and analysis; normally leave it to `rng`."""
        if deal is None:
            deal = self._random_deal(rng)
        elif deal not in self.all_deals():
            raise ValueError("not a valid deal for this game")
        return self._make_hand(stacks, first, deal)


class Bot(Protocol):
    """A bot is a pure function of the Decision it is shown (plus its own randomness)."""

    name: str

    def act(self, decision: Decision, rng: random.Random) -> Action: ...
