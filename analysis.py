"""Exact expected value of one hand, by enumerating every possible deal."""

import random
from fractions import Fraction

from .base import Bot, Game


class NoRandomness(random.Random):
    """An rng that refuses to produce numbers: proves a bot is deterministic."""

    def random(self):  # noqa: D102
        raise RuntimeError("this bot uses randomness, so its exact EV is not defined")

    def getrandbits(self, k):  # noqa: D102
        raise RuntimeError("this bot uses randomness, so its exact EV is not defined")


def exact_hand_ev(
    game: Game,
    bots: tuple[Bot, Bot],
    first: int = 0,
    stack: int | None = None,
) -> Fraction:
    """Seat 0's exact expected net chips for ONE hand, averaged over all equally likely deals.

    Works only for deterministic bots (cautious, ev, kelly); raises for random.
    """
    stacks = game.default_stack if stack is None else stack
    rng = NoRandomness()
    deals = game.all_deals()
    total = Fraction(0)
    for deal in deals:
        hand = game.new_hand(rng, (stacks, stacks), first, deal=deal)
        while not hand.done:
            seat = hand.to_act
            hand.act(bots[seat].act(hand.decision(), rng))
        total += hand.result.net[0]
    return total / len(deals)
