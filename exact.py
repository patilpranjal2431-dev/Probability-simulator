"""Exact probabilities, computed with integers and Fractions (no floating-point error).

Contents
* dice_sum_*        - distribution of the sum of k dice by dynamic programming
* poker_hand_*      - 5-card poker hand counts by combinatorial counting
* hypergeometric_*  - drawing without replacement
* birthday_*        - the birthday problem
"""

from fractions import Fraction
from math import comb

# --------------------------------------------------------------------------- dice


def _check_dice_args(num_dice: int, sides: int) -> None:
    for name, value in (("num_dice", num_dice), ("sides", sides)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if num_dice < 1:
        raise ValueError("num_dice must be at least 1")
    if sides < 1:
        raise ValueError("sides must be at least 1")


def dice_sum_counts(num_dice: int, sides: int = 6) -> dict[int, int]:
    """Number of ways each total can occur when rolling `num_dice` fair dice.

    Dynamic programming: if `ways[s]` is the number of ways the first i dice can
    total s, then after one more die, `new[s + face] += ways[s]` for each face.
    The counts always add up to sides ** num_dice.
    """
    _check_dice_args(num_dice, sides)
    ways: dict[int, int] = {0: 1}
    for _ in range(num_dice):
        nxt: dict[int, int] = {}
        for total, count in ways.items():
            for face in range(1, sides + 1):
                nxt[total + face] = nxt.get(total + face, 0) + count
        ways = nxt
    return dict(sorted(ways.items()))


def dice_sum_probabilities(num_dice: int, sides: int = 6) -> dict[int, Fraction]:
    """Exact probability of each total."""
    counts = dice_sum_counts(num_dice, sides)  # validates the arguments
    outcomes = sides**num_dice
    return {total: Fraction(count, outcomes) for total, count in counts.items()}


def dice_sum_probability(num_dice: int, target: int, sides: int = 6) -> Fraction:
    """Exact probability that `num_dice` dice total exactly `target` (0 if impossible)."""
    counts = dice_sum_counts(num_dice, sides)
    return Fraction(counts.get(target, 0), sides**num_dice)


# -------------------------------------------------------------------------- poker

POKER_CATEGORIES: tuple[str, ...] = (
    "royal_flush",
    "straight_flush",
    "four_of_a_kind",
    "full_house",
    "flush",
    "straight",
    "three_of_a_kind",
    "two_pair",
    "one_pair",
    "high_card",
)

TOTAL_POKER_HANDS = comb(52, 5)  # 2,598,960


def poker_hand_counts() -> dict[str, int]:
    """How many of the 2,598,960 five-card hands fall in each category.

    Each category is counted with a formula. Categories are exclusive: "flush" means a
    flush that is NOT a straight flush, and so on.

    Building blocks:
      * 13 ranks, 4 suits.
      * 10 possible straights (A-2-3-4-5 up to 10-J-Q-K-A, since the ace counts low or high).
      * C(13,5) = 1287 ways to pick 5 distinct ranks; 10 of those are straights,
        so 1277 are not.
      * 4**5 = 1024 ways to assign suits to 5 cards; 4 of those make all the same suit,
        so 1020 are not a flush.
    """
    rank_sets_non_straight = comb(13, 5) - 10  # 1277 sets of 5 ranks that are not a straight
    suit_patterns_non_flush = 4**5 - 4  # 1020 suit assignments that are not a flush

    counts = {
        "royal_flush": 4,  # one per suit
        "straight_flush": 10 * 4 - 4,  # 10 straights x 4 suits, minus the 4 royal flushes
        "four_of_a_kind": 13 * 48,  # rank of the four, then any other card
        "full_house": 13 * comb(4, 3) * 12 * comb(4, 2),  # trip rank+suits, pair rank+suits
        "flush": 4 * rank_sets_non_straight,  # 4 suits x non-straight rank sets
        "straight": 10 * suit_patterns_non_flush,  # 10 straights x non-flush suit patterns
        "three_of_a_kind": 13 * comb(4, 3) * comb(12, 2) * 4**2,
        "two_pair": comb(13, 2) * comb(4, 2) ** 2 * 44,  # two pair ranks, suits, kicker (44 cards)
        "one_pair": 13 * comb(4, 2) * comb(12, 3) * 4**3,
        "high_card": rank_sets_non_straight * suit_patterns_non_flush,
    }
    assert sum(counts.values()) == TOTAL_POKER_HANDS  # internal consistency check
    return counts


def poker_hand_probabilities() -> dict[str, Fraction]:
    """Exact probability of each 5-card poker category."""
    return {k: Fraction(v, TOTAL_POKER_HANDS) for k, v in poker_hand_counts().items()}


# ------------------------------------------------------------------ hypergeometric


def _check_hypergeometric(population: int, successes: int, draws: int) -> None:
    for name, value in (("population", population), ("successes", successes), ("draws", draws)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if population < 1:
        raise ValueError("population must be at least 1")
    if not 0 <= successes <= population:
        raise ValueError("successes must be between 0 and population")
    if not 0 <= draws <= population:
        raise ValueError("draws must be between 0 and population")


def hypergeometric_pmf(population: int, successes: int, draws: int, k: int) -> Fraction:
    """P(exactly k successes) when drawing `draws` items WITHOUT replacement from a
    population holding `successes` successes.

        C(successes, k) * C(population - successes, draws - k) / C(population, draws)
    """
    _check_hypergeometric(population, successes, draws)
    failures = population - successes
    if k < 0 or k > successes or draws - k < 0 or draws - k > failures:
        return Fraction(0)
    return Fraction(comb(successes, k) * comb(failures, draws - k), comb(population, draws))


def hypergeometric_cdf(population: int, successes: int, draws: int, k: int) -> Fraction:
    """P(at most k successes)."""
    _check_hypergeometric(population, successes, draws)
    return sum(
        (hypergeometric_pmf(population, successes, draws, i) for i in range(0, k + 1)),
        Fraction(0),
    )


# --------------------------------------------------------------------- birthday


def birthday_no_match_probability(people: int, days: int = 365) -> Fraction:
    """P(everyone has a different birthday) = prod_{i=0}^{people-1} (days - i) / days."""
    for name, value in (("people", people), ("days", days)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if people < 0 or days < 1:
        raise ValueError("people must be >= 0 and days must be >= 1")
    if people > days:
        return Fraction(0)  # pigeonhole principle: a repeat is certain
    result = Fraction(1)
    for i in range(people):
        result *= Fraction(days - i, days)
    return result


def birthday_match_probability(people: int, days: int = 365) -> Fraction:
    """P(at least two people share a birthday)."""
    return 1 - birthday_no_match_probability(people, days)


def birthday_smallest_group(target: float = 0.5, days: int = 365) -> int:
    """Smallest group size whose match probability is at least `target` (0 < target <= 1)."""
    if not 0 < target <= 1:
        raise ValueError("target must be in (0, 1]")
    people = 1
    while birthday_match_probability(people, days) < Fraction(target):
        people += 1
    return people


# ---------------------------------------------------------------------- coin runs


def coin_run_probability(flips: int, run: int) -> Fraction:
    """P(at least one run of `run` heads in a row) in `flips` fair coin flips.

    Dynamic programming over the length of the current run of heads: a tail resets it to 0,
    a head extends it, and reaching `run` heads means the event has happened. We track the
    probability of NEVER reaching it, and subtract from 1."""
    for name, value in (("flips", flips), ("run", run)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if flips < 1 or run < 1:
        raise ValueError("flips and run must be at least 1")
    if run > flips:
        return Fraction(0)
    distribution = [Fraction(0)] * run  # distribution[r] = P(current run is r and no run of `run` yet)
    distribution[0] = Fraction(1)
    for _ in range(flips):
        nxt = [Fraction(0)] * run
        for length, probability in enumerate(distribution):
            if probability:
                nxt[0] += probability / 2  # a tail
                if length + 1 < run:
                    nxt[length + 1] += probability / 2  # a head (reaching `run` leaves the tracked states)
        distribution = nxt
    return 1 - sum(distribution)
