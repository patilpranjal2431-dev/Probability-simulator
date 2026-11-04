"""The fixed reference opponents. Every player is graded against the SAME one, which is what
makes grades comparable. They are models, not claims about optimal play:

* Dice Duel: the EV bot of Phase 4. It bets (the maximum) with a better-than-average roll and
  checks otherwise; facing a bet it calls exactly when calling beats folding.
* Kuhn poker: the published Nash equilibrium (bluff a Jack 1/3 of the time, bet every King, ...).

A reference answers one question: given the hand as the opponent sees it, what will it do?
"""

from math import ceil

from app.games.base import CALL, CHECK, FOLD, Action, Hand, bet
from app.games.bots import EVBot

KUHN_CARDS = ("J", "Q", "K")


def history_key(history: list[tuple[int, Action]]) -> str:
    """Kuhn information-set history: 'p' = check/fold, 'b' = bet/call."""
    return "".join("b" if action.kind in ("bet", "call") else "p" for _seat, action in history)


class EVReference:
    """Dice Duel reference: exactly the EVBot's decision rules."""

    name = "ev-bot"
    _bot = EVBot()

    def action_probs(self, hand: Hand) -> dict[Action, float]:
        return {self._bot.act(hand.decision(), None): 1.0}

    def bet_breakpoints(self, hand: Hand) -> list[int]:
        """Bet sizes at which the reference caller's response changes.

        Expected value is a straight line in the bet size between these points, so the best bet
        is always the minimum, the maximum, or one of these. That lets us check a handful of sizes
        instead of every size (and a test confirms it finds the same optimum as checking all).
        """
        d = hand.decision()
        sides = round(1 / d.tie_prob) if d.tie_prob else 6
        ante = d.my_invested  # both players have put in the ante and nothing else
        points: list[int] = []
        for value in range(1, sides + 1):
            edge = ((value - 1) - (sides - value)) / sides  # the caller's win minus lose chance
            if edge < 0:
                # the caller calls while edge * (ante + b) > -ante, i.e. while b < ante/(-edge) - ante
                limit = ante / (-edge) - ante
                last_call = ceil(limit) - 1
                points += [last_call, last_call + 1]
        return points


class KuhnNashReference:
    """Kuhn poker reference: the Nash equilibrium with bluffing parameter alpha = 1/3.

    Probabilities are of the AGGRESSIVE action (bet when betting, call when responding),
    keyed by (card, history).
    """

    name = "kuhn-nash"
    ALPHA = 1 / 3
    TABLE = {
        # first player
        ("J", ""): ALPHA, ("Q", ""): 0.0, ("K", ""): 3 * ALPHA,
        ("J", "pb"): 0.0, ("Q", "pb"): ALPHA + 1 / 3, ("K", "pb"): 1.0,
        # second player
        ("J", "b"): 0.0, ("Q", "b"): 1 / 3, ("K", "b"): 1.0,
        ("J", "p"): 1 / 3, ("Q", "p"): 0.0, ("K", "p"): 1.0,
    }

    def action_probs(self, hand: Hand) -> dict[Action, float]:
        seat = hand.to_act
        card = KUHN_CARDS[hand.private_view(seat)]
        p = self.TABLE[(card, history_key(hand.history))]
        d = hand.decision()
        if d.kind == "respond":
            aggressive, passive = CALL, FOLD
        else:
            aggressive, passive = (bet(1), CHECK) if d.can_bet() else (CHECK, CHECK)
        probs: dict[Action, float] = {}
        for action, weight in ((aggressive, p), (passive, 1 - p)):
            if weight > 0:
                probs[action] = probs.get(action, 0.0) + weight
        return probs

    def bet_breakpoints(self, hand: Hand) -> list[int]:
        return []  # only one bet size exists


def reference_for(game_name: str):
    """The reference opponent used to grade a game."""
    if game_name == "dice_duel":
        return EVReference()
    if game_name == "kuhn":
        return KuhnNashReference()
    raise ValueError(f"no reference opponent for game {game_name!r}")
