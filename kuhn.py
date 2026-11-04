"""Kuhn poker: the second game, a "simplified card game".

A 3-card deck (Jack < Queen < King), one card each, one ante each, a single fixed bet size.
First player checks or bets; if they check, the second player may check (showdown) or bet,
and then the first player must call or fold. Higher card wins at showdown.

Kuhn poker was solved by hand in 1950. With optimal play the first player loses exactly
1/18 of an ante per hand, which makes it a good test of this engine's rules.
"""

import random
from itertools import permutations

from .base import Game, Hand

CARD_NAMES = ("J", "Q", "K")


class KuhnHand(Hand):
    name = "kuhn"
    _second_player_may_bet_after_check = True

    def _equity(self, seat):
        card = self._private[seat]  # 0, 1 or 2; the opponent holds one of the other two
        win = card / 2
        return win, 0.0, 1 - win

    def _bet_limits(self, cap):
        return (1, 1) if cap >= 1 else (0, 0)

    def _compare(self):
        a, b = self._private
        return (a > b) - (a < b)


class Kuhn(Game):
    name = "kuhn"
    ante = 1
    default_stack = 100

    def all_deals(self):
        return list(permutations(range(3), 2))  # 6 equally likely deals

    def _random_deal(self, rng: random.Random):
        first, second = rng.sample(range(3), 2)
        return first, second

    def _make_hand(self, stacks, first, deal):
        return KuhnHand(self.ante, stacks, first, deal)
