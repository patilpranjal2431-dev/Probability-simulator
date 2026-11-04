"""Monte Carlo estimation of probabilities, with Wilson 95% confidence intervals.

Randomness: simulations use a SEEDED `random.Random`, so a result can be reproduced
exactly from its seed. (Live games, in a later phase, use the `secrets` module instead,
which cannot be seeded or predicted.)

An "event" is a function `trial(rng) -> bool`: run the experiment once and say whether
the event happened. The estimate is  p_hat = hits / n.
"""

import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import sqrt

from .poker import DECK_SIZE, classify_hand

Trial = Callable[[random.Random], bool]

Z_95 = 1.959963984540054  # 97.5th percentile of the standard normal distribution


# ----------------------------------------------------------------- statistics


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion (95% by default).

    Better than the simple "p +/- z*sqrt(p(1-p)/n)" interval: it stays inside [0, 1] and
    behaves sensibly for rare events and small n (where the simple one collapses to
    zero width when 0 successes are seen).
    """
    for value in (successes, n):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("successes and n must be integers")
    if n < 0 or not 0 <= successes <= n:
        raise ValueError("need 0 <= successes <= n")
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    z2 = z * z
    denominator = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denominator
    half_width = z * sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    # At the extremes the exact bounds are 0 and 1; floating-point rounding could
    # otherwise leave 8.7e-19 instead of 0, putting the estimate outside its own interval.
    low = 0.0 if successes == 0 else max(0.0, centre - half_width)
    high = 1.0 if successes == n else min(1.0, centre + half_width)
    return low, high


@dataclass(frozen=True)
class Estimate:
    """A Monte Carlo estimate after `n` trials."""

    n: int
    successes: int
    p_hat: float
    std_error: float  # sqrt(p_hat * (1 - p_hat) / n)
    ci_low: float
    ci_high: float

    def contains(self, value: float) -> bool:
        """Is `value` inside the 95% confidence interval?"""
        return self.ci_low <= value <= self.ci_high


def make_estimate(successes: int, n: int) -> Estimate:
    """Build an Estimate from raw counts."""
    low, high = wilson_interval(successes, n)
    p_hat = successes / n if n else 0.0
    std_error = sqrt(p_hat * (1 - p_hat) / n) if n else 0.0
    return Estimate(n, successes, p_hat, std_error, low, high)


# ------------------------------------------------------------------- sampling


def log_spaced_sizes(max_n: int, points_per_decade: int = 4, min_n: int = 10) -> list[int]:
    """Sample sizes spread evenly on a log scale, e.g. 10, 18, 32, 56, 100, ... max_n."""
    if min_n < 1 or max_n < min_n or points_per_decade < 1:
        raise ValueError("need 1 <= min_n <= max_n and points_per_decade >= 1")
    sizes: set[int] = {max_n}
    k = 0
    while True:
        value = round(min_n * 10 ** (k / points_per_decade))
        if value >= max_n:
            break
        sizes.add(value)
        k += 1
    return sorted(sizes)


def simulate(trial: Trial, sizes: Iterable[int], seed: int) -> list[Estimate]:
    """Run ONE long stream of trials and record the running estimate at each size.

    Using one stream means the estimate at n=1000 is built from the same trials as
    the estimate at n=100, so the points on a convergence chart are consistent.
    """
    checkpoints = sorted(set(sizes))
    if not checkpoints or checkpoints[0] < 1:
        raise ValueError("sizes must be positive integers")
    rng = random.Random(seed)
    results: list[Estimate] = []
    hits = 0
    done = 0
    for target in checkpoints:
        while done < target:
            hits += bool(trial(rng))
            done += 1
        results.append(make_estimate(hits, done))
    return results


# --------------------------------------------------------------- shuffling


class CardDrawer:
    """Draw k distinct cards from a 52-card deck with a partial Fisher-Yates shuffle.

    Fisher-Yates: for i = 0..k-1, swap position i with a uniformly random position in
    [i, 51]. Every ordered draw is then equally likely. (Sorting by random keys, by
    contrast, is biased whenever keys can tie, and is slower.)

    The deck is NOT reset between draws: applying a uniform shuffle step to ANY
    arrangement still gives a uniform result, so reuse is both correct and fast.
    """

    def __init__(self, deck_size: int = DECK_SIZE) -> None:
        self._deck = list(range(deck_size))

    def draw(self, rng: random.Random, k: int) -> list[int]:
        deck = self._deck
        size = len(deck)
        if not 0 <= k <= size:
            raise ValueError("cannot draw that many cards")
        for i in range(k):
            j = i + rng.randrange(size - i)
            deck[i], deck[j] = deck[j], deck[i]
        return deck[:k]


# ----------------------------------------------------------------- trials


def dice_sum_trial(num_dice: int, target: int, sides: int = 6) -> Trial:
    """Event: `num_dice` dice total exactly `target`."""
    if num_dice < 1 or sides < 1:
        raise ValueError("num_dice and sides must be at least 1")

    def trial(rng: random.Random) -> bool:
        return sum(rng.randint(1, sides) for _ in range(num_dice)) == target

    return trial


def poker_trial(category: str) -> Trial:
    """Event: a random 5-card hand falls in `category` (see poker.classify_hand)."""
    drawer = CardDrawer()

    def trial(rng: random.Random) -> bool:
        return classify_hand(drawer.draw(rng, 5)) == category

    return trial


def hypergeometric_trial(population: int, successes: int, draws: int, k: int) -> Trial:
    """Event: exactly k 'success' items appear when drawing `draws` from `population`.

    Items 0..successes-1 count as successes.
    """
    if not 0 <= successes <= population or not 0 <= draws <= population:
        raise ValueError("invalid hypergeometric parameters")
    drawer = CardDrawer(population)

    def trial(rng: random.Random) -> bool:
        return sum(1 for item in drawer.draw(rng, draws) if item < successes) == k

    return trial


def birthday_trial(people: int, days: int = 365) -> Trial:
    """Event: at least two of `people` share a birthday (each uniform over `days`)."""
    if people < 0 or days < 1:
        raise ValueError("people must be >= 0 and days must be >= 1")

    def trial(rng: random.Random) -> bool:
        seen = set()
        for _ in range(people):
            day = rng.randrange(days)
            if day in seen:
                return True
            seen.add(day)
        return False

    return trial
