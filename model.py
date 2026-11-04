"""A repeated bet, the Kelly fraction, and the long-run growth rate.

A BetSpec describes ONE bet that is repeated many times:
    win  with probability `win`   -> you gain  odds * stake
    lose with probability `lose`  -> you lose  stake
    push with the remaining probability -> nothing happens (like a tie in Dice Duel)

Staking a fraction f of your bankroll each time, one bet multiplies your bankroll by
    (1 + odds*f)  on a win,   (1 - f)  on a loss,   1  on a push.
The long-run growth rate per bet is the average of the logarithm of that multiplier:
    g(f) = win*ln(1 + odds*f) + lose*ln(1 - f)
The Kelly fraction is the f that maximises g(f).
"""

from dataclasses import dataclass
from math import isfinite, log

TOLERANCE = 1e-9


@dataclass(frozen=True)
class BetSpec:
    win: float
    lose: float
    odds: float = 1.0  # net payout per 1 staked when you win (1.0 = even money)

    def __post_init__(self) -> None:
        for name, value in (("win", self.win), ("lose", self.lose), ("odds", self.odds)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.win < 0 or self.lose < 0 or self.win + self.lose > 1 + TOLERANCE:
            raise ValueError("win and lose must be non-negative and add up to at most 1")
        if self.win + self.lose <= 0:
            raise ValueError("a bet that can neither win nor lose is not a bet")
        if not 0 < self.odds <= 1000:
            raise ValueError("odds must be greater than 0 and at most 1000")

    @property
    def push(self) -> float:
        return max(0.0, 1.0 - self.win - self.lose)

    @property
    def edge(self) -> float:
        """Expected net gain per 1 staked: odds*win - lose. Positive means the bet favours you."""
        return self.odds * self.win - self.lose

    def kelly_fraction(self) -> float:
        """The stake fraction that maximises long-run growth. Zero if there is no edge.

        Setting g'(f) = 0:   win*odds/(1+odds*f) = lose/(1-f)   gives
            f* = (odds*win - lose) / (odds*(win + lose))
        With no pushes (win+lose = 1) this is the textbook (b*p - q)/b."""
        return max(0.0, self.edge / (self.odds * (self.win + self.lose)))

    def growth_rate(self, fraction: float) -> float:
        """Expected log growth of the bankroll per bet when staking `fraction` of it each time."""
        if not 0 <= fraction <= 1:
            raise ValueError("fraction must be between 0 and 1")
        if fraction == 1 and self.lose > 0:
            return float("-inf")  # staking everything: one loss wipes you out
        total = 0.0
        if self.win:
            total += self.win * log(1 + self.odds * fraction)
        if self.lose:
            total += self.lose * log(1 - fraction)
        return total

    def breakeven_fraction(self) -> float | None:
        """The stake fraction ABOVE the Kelly fraction where growth falls back to zero.

        Staking more than this loses money in the long run even though every bet is favourable.
        Returns None if there is no edge (growth is never positive) or the bet cannot lose."""
        star = self.kelly_fraction()
        if star <= 0 or self.lose == 0:
            return None
        low, high = star, 1.0 - 1e-12
        if self.growth_rate(high) >= 0:
            return None
        for _ in range(200):  # bisection: g is decreasing beyond the Kelly fraction
            mid = (low + high) / 2
            if self.growth_rate(mid) > 0:
                low = mid
            else:
                high = mid
        return (low + high) / 2
