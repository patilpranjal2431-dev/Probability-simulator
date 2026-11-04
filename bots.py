"""Four bots. Each is a pure function of the Decision it is shown (plus its own randomness),
so the same bots play both Dice Duel and Kuhn poker.

All of them use the "naive equity" in the Decision (win/tie/lose chance against a uniformly
random opponent value). None of them read what the opponent's bet reveals, so none of them
bluff or detect bluffs. That is a deliberate simplification, not a claim of optimal play.

    random    picks a legal action at random
    cautious  only bets or calls with strong hands, and bets small
    ev        risk-neutral: maximises expected chips, ignoring risk of ruin
    kelly     maximises expected LOG wealth (the Kelly criterion), so it sizes bets to its edge
"""

import math
import random

from .base import CALL, CHECK, FOLD, Action, Decision, bet

STRONG = 0.66  # "strong" = at least about a 2-in-3 chance to win (a 5 or 6 on a d6, a King in Kuhn)


# --------------------------------------------------------------------------- helpers


def edge(d: Decision) -> float:
    """win chance minus lose chance (ties cancel out)."""
    return d.win_prob - d.lose_prob


def expected_log_wealth(d: Decision, final_invested: int) -> float:
    """E[ln(final wealth)] if the hand reaches showdown with `final_invested` chips each.

    Wealth now is stack + my_invested. Win: +final_invested. Lose: -final_invested. Tie: 0.
    Losing everything gives ln(0) = -infinity, which correctly ranks as the worst outcome.
    """
    wealth = d.wealth
    total = 0.0
    for probability, outcome in (
        (d.win_prob, wealth + final_invested),
        (d.tie_prob, wealth),
        (d.lose_prob, wealth - final_invested),
    ):
        if probability > 0:
            total += probability * (math.log(outcome) if outcome > 0 else -math.inf)
    return total


# ------------------------------------------------------------------------------ bots


class RandomBot:
    """Uniformly random legal action: a bet size uniform on 0..max_bet (0 = check), call or fold 50/50."""

    name = "random"

    def act(self, d: Decision, rng: random.Random) -> Action:
        if d.kind == "respond":
            return CALL if rng.random() < 0.5 else FOLD
        if not d.can_bet():
            return CHECK
        size = rng.randint(0, d.max_bet)
        if size == 0:
            return CHECK
        return bet(max(size, d.min_bet))


class CautiousBot:
    """Bets a small amount (5% of its stack, at least the minimum) with a strong hand,
    checks otherwise, and calls only with a strong hand."""

    name = "cautious"

    def act(self, d: Decision, rng: random.Random) -> Action:
        strong = d.win_prob >= STRONG
        if d.kind == "respond":
            return CALL if strong else FOLD
        if strong and d.can_bet():
            return bet(min(d.max_bet, max(d.min_bet, d.stack // 20)))
        return CHECK


class EVBot:
    """Risk-neutral expected-value maximiser, assuming the opponent always calls.

    If a bet is called, a bet of b has expected value  edge * (my_invested + b),
    which grows with b when the edge is positive, so it bets the maximum; with no edge it checks.
    Facing a bet it calls exactly when  edge * opp_invested  beats the loss from folding.
    """

    name = "ev"

    def act(self, d: Decision, rng: random.Random) -> Action:
        if d.kind == "respond":
            return CALL if edge(d) * d.opp_invested > -d.my_invested else FOLD
        if edge(d) > 0 and d.can_bet():
            return bet(d.max_bet)
        return CHECK


class KellyBot:
    """Kelly criterion: maximise the expected LOG of its bankroll.

    For an even-money bet with win chance w, lose chance l and the rest ties, expected log
    wealth is maximised by risking the fraction  f* = (w - l) / (w + l)  of the bankroll.
    `fraction` scales that down (half-Kelly by default, the usual defence against errors in
    the probability estimate). It bets nothing without a positive edge.

    Facing a bet it compares expected log wealth after calling with the certain result of folding.
    """

    def __init__(self, fraction: float = 0.5):
        if not 0 < fraction <= 1:
            raise ValueError("fraction must be in (0, 1]")
        self.fraction = fraction
        self.name = "kelly"

    def act(self, d: Decision, rng: random.Random) -> Action:
        if d.kind == "respond":
            call_value = expected_log_wealth(d, d.opp_invested)
            fold_value = math.log(d.stack) if d.stack > 0 else -math.inf
            return CALL if call_value >= fold_value else FOLD

        win, lose = d.win_prob, d.lose_prob
        if not d.can_bet() or win <= lose:
            return CHECK
        f_star = (win - lose) / (win + lose)
        target_total = self.fraction * f_star * d.wealth  # total chips to have in the pot
        size = math.floor(target_total - d.my_invested)
        if size < d.min_bet:
            return CHECK
        return bet(min(size, d.max_bet))


# ---------------------------------------------------------------------------- registry

BOT_NAMES = ("random", "cautious", "ev", "kelly")


def make_bot(name: str):
    """Create a bot by name."""
    bots = {"random": RandomBot, "cautious": CautiousBot, "ev": EVBot, "kelly": KellyBot}
    if name not in bots:
        raise ValueError(f"unknown bot {name!r}; choose from {', '.join(BOT_NAMES)}")
    return bots[name]()
