"""TableCore: one match between two seats, as a plain synchronous state machine.

No sockets, timers or database in here, so every rule can be tested deterministically.
The async `Table` wraps this and turns time, connections and money into method calls.

Methods that change the match return a list of EVENT dicts for the caller to publish:
    {"type": "hand_result", ...}   public result of a finished hand
    {"type": "auto_action", ...}   the table acted for a player (timeout / disconnect)
    {"type": "match_over", ...}    the match has ended
"""

import random
from dataclasses import dataclass

from app.games.base import CHECK, FOLD, Action, Bot, Game, Hand, HandResult, IllegalActionError

from app.grading.types import DecisionContext

from .common import TableError


@dataclass
class Seat:
    user_id: int
    name: str
    bot: Bot | None = None

    @property
    def is_bot(self) -> bool:
        return self.bot is not None


class TableCore:
    def __init__(
        self,
        game: Game,
        seats: tuple[Seat, Seat],
        buy_in: int,
        rng: random.Random,
        bot_rngs: tuple[random.Random, random.Random],
        max_hands: int = 20,
    ) -> None:
        if buy_in < game.ante:
            raise ValueError("buy-in must cover the ante")
        if max_hands < 1:
            raise ValueError("max_hands must be at least 1")
        self.game = game
        self.seats = seats
        self.buy_in = buy_in
        self.max_hands = max_hands
        self.stacks = [buy_in, buy_in]  # chips each seat holds BETWEEN hands
        self.hands_played = 0
        self.status = "playing"  # "playing" or "finished"
        self.end_reason: str | None = None
        self.results: list[HandResult] = []  # server-side record of every finished hand
        self.decision_log: list[DecisionContext] = []  # every human decision, for grading (server-side)
        self._rng = rng
        self._bot_rngs = bot_rngs
        self.hand: Hand = self._new_hand()
        self.opening_events: list[dict] = []
        self._advance(self.opening_events)  # a bot may have to act first

    # ------------------------------------------------------------------ queries

    @property
    def to_act(self) -> int | None:
        return self.hand.to_act if self.status == "playing" and not self.hand.done else None

    def view_for(self, seat: int) -> dict:
        """Everything `seat` may know right now. Contains ONLY that seat's own hidden value."""
        hand = self.hand
        view = {
            "type": "state",
            "game": self.game.name,
            "you": seat,
            "hand_number": self.hands_played + 1,
            "max_hands": self.max_hands,
            "match_stacks": list(self.stacks),  # chips at the start of this hand
            "hand": hand.public_state(),
            "your_value": hand.private_view(seat),
            "decision": None,
        }
        if self.status == "playing" and not hand.done and hand.to_act == seat:
            d = hand.decision()
            actions = ["call", "fold"] if d.kind == "respond" else (["check", "bet"] if d.can_bet() else ["check"])
            view["decision"] = {
                "kind": d.kind,
                "actions": actions,
                "to_call": d.to_call,
                "min_bet": d.min_bet,
                "max_bet": d.max_bet,
            }
        return view

    # ----------------------------------------------------------------- commands

    def submit(self, seat: int, action: Action) -> list[dict]:
        """A player's move. Raises TableError if it is not allowed (state is unchanged)."""
        self._require_playing()
        if seat != self.hand.to_act:
            raise TableError("not_your_turn", "it is not your turn")
        context = self._context(seat, action, auto=False)
        try:
            self.hand.act(action)
        except IllegalActionError as error:
            raise TableError("illegal_action", str(error)) from None
        self._log(context)
        events: list[dict] = []
        self._advance(events)
        return events

    def auto_play(self, seat: int, reason: str) -> list[dict]:
        """Act for a player who timed out or disconnected: fold if facing a bet, else check."""
        if self.status != "playing" or self.hand.to_act != seat:
            return []
        action = FOLD if self.hand.decision().kind == "respond" else CHECK
        context = self._context(seat, action, auto=True)
        self.hand.act(action)
        self._log(context)
        events: list[dict] = [{"type": "auto_action", "seat": seat, "action": action.kind, "reason": reason}]
        self._advance(events)
        return events

    def concede(self, seat: int, reason: str) -> list[dict]:
        """`seat` abandons the match: they lose the current hand's chips, and the match ends.
        They keep whatever is left of their stack."""
        if self.status != "playing":
            return []
        events: list[dict] = []
        if not self.hand.done:
            self.hand.concede(seat)
            self._record_hand(events)
        self._end(reason, events, conceded_by=seat)
        return events

    # ---------------------------------------------------------------- internals

    def _context(self, seat: int, action: Action, auto: bool) -> DecisionContext | None:
        """Snapshot of the decision about to be made (None for a bot: bots are not graded)."""
        if self.seats[seat].is_bot:
            return None
        hand = self.hand
        return DecisionContext(
            hand_number=self.hands_played + 1,
            decision_index=sum(1 for who, _ in hand.history if who == seat),
            seat=seat,
            own_value=hand.private_view(seat),
            stacks=(self.stacks[0], self.stacks[1]),
            first=hand.first,
            history=tuple((who, a.kind, a.amount) for who, a in hand.history),
            action_kind=action.kind,
            action_amount=action.amount,
            auto=auto,
        )

    def _log(self, context: DecisionContext | None) -> None:
        if context is not None:
            self.decision_log.append(context)

    def _require_playing(self) -> None:
        if self.status != "playing":
            raise TableError("match_over", "the match is over")

    def _new_hand(self) -> Hand:
        return self.game.new_hand(self._rng, (self.stacks[0], self.stacks[1]), first=self.hands_played % 2)

    def _advance(self, events: list[dict]) -> None:
        """Run forward until a human must act, or the match ends."""
        while self.status == "playing":
            if self.hand.done:
                self._record_hand(events)
                reason = self._over_reason()
                if reason:
                    self._end(reason, events)
                    return
                self.hand = self._new_hand()
                continue
            seat = self.hand.to_act
            bot = self.seats[seat].bot
            if bot is None:
                return
            self.hand.act(bot.act(self.hand.decision(), self._bot_rngs[seat]))

    def _record_hand(self, events: list[dict]) -> None:
        result = self.hand.result
        for seat in (0, 1):
            self.stacks[seat] += result.net[seat]
        self.hands_played += 1
        self.results.append(result)
        events.append(
            {
                "type": "hand_result",
                "hand": self.hands_played,
                "net": list(result.net),
                "ended_by": result.ended_by,
                "folded": result.folded,
                # hidden values are revealed ONLY if there was a showdown
                "showdown_values": list(result.showdown_values) if result.showdown_values else None,
                "stacks": list(self.stacks),
            }
        )

    def _over_reason(self) -> str | None:
        if self.hands_played >= self.max_hands:
            return "max_hands"
        if min(self.stacks) < self.game.ante:
            return "bust"
        return None

    def _end(self, reason: str, events: list[dict], conceded_by: int | None = None) -> None:
        self.status = "finished"
        self.end_reason = reason
        events.append(
            {
                "type": "match_over",
                "reason": reason,
                "final_stacks": list(self.stacks),
                "hands": self.hands_played,
                "conceded_by": conceded_by,
            }
        )
