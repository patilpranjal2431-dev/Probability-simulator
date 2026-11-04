"""Dice Duel: the first game.

Each hand: both players ante, then each secretly rolls one die. The first player may check
(go to showdown for the antes) or bet any whole number of coins up to the smaller stack.
Facing a bet, the other player calls or folds. At showdown the higher die takes the pot;
equal dice return everyone's chips.
"""

import random

from .base import Game, Hand


class DiceDuelHand(Hand):
    name = "dice_duel"
    _second_player_may_bet_after_check = False  # a check goes straight to showdown

    def __init__(self, ante, stacks, first, private, sides):
        super().__init__(ante, stacks, first, private)
        self.sides = sides

    def _equity(self, seat):
        roll = self._private[seat]
        return (roll - 1) / self.sides, 1 / self.sides, (self.sides - roll) / self.sides

    def _bet_limits(self, cap):
        return (1, cap) if cap >= 1 else (0, 0)

    def _compare(self):
        a, b = self._private
        return (a > b) - (a < b)


class DiceDuel(Game):
    name = "dice_duel"
    ante = 10
    default_stack = 1000

    def __init__(self, sides: int = 6):
        if sides < 2:
            raise ValueError("a die needs at least 2 sides")
        self.sides = sides

    def all_deals(self):
        faces = range(1, self.sides + 1)
        return [(a, b) for a in faces for b in faces]

    def _random_deal(self, rng: random.Random):
        return rng.randint(1, self.sides), rng.randint(1, self.sides)

    def _make_hand(self, stacks, first, deal):
        return DiceDuelHand(self.ante, stacks, first, deal, self.sides)
