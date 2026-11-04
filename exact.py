"""Exact reference results, used to check the simulator. None of this uses randomness."""

from itertools import product

from .model import BetSpec


def gamblers_ruin_probability(win: float, start_units: int, target_units: int | None = None) -> float:
    """Chance of going broke, betting ONE unit at even money (win with probability `win`,
    otherwise lose), starting from `start_units`.

    With a `target_units`: the chance of hitting 0 BEFORE reaching the target. With
    r = (1-p)/p and p != 1/2:     P(ruin) = (r^i - r^N) / (1 - r^N)
    For a fair game (p = 1/2):    P(ruin) = 1 - i/N
    Without a target (play forever): P(ruin) = r^i if p > 1/2, and 1 otherwise.
    """
    if not 0 < win < 1:
        raise ValueError("win must be strictly between 0 and 1")
    if start_units < 1 or (target_units is not None and target_units <= start_units):
        raise ValueError("need 1 <= start < target")
    ratio = (1 - win) / win
    if target_units is None:
        return 1.0 if win <= 0.5 else ratio**start_units
    if abs(win - 0.5) < 1e-12:
        return 1 - start_units / target_units
    return (ratio**start_units - ratio**target_units) / (1 - ratio**target_units)


def enumerate_fixed_fraction(spec: BetSpec, fraction: float, bets: int, ruin_fraction: float) -> dict:
    """EXACT results for staking `fraction` of the bankroll on each of `bets` bets, by trying every
    possible sequence of win/lose/push (3**bets of them, so keep `bets` small).

    Returns the exact expected final bankroll (starting from 1), the exact probability that the bankroll
    ever touches `ruin_fraction`, and the exact probability of finishing below the start.
    """
    outcomes = [(spec.win, 1 + spec.odds * fraction), (spec.lose, 1 - fraction), (spec.push, 1.0)]
    outcomes = [(p, m) for p, m in outcomes if p > 0]
    mean_final = 0.0
    ruin = 0.0
    below_start = 0.0
    for sequence in product(outcomes, repeat=bets):
        probability, bankroll, touched = 1.0, 1.0, 1.0 <= ruin_fraction + 1e-12
        for p, multiplier in sequence:
            probability *= p
            bankroll *= multiplier
            touched = touched or bankroll <= ruin_fraction + 1e-12
        mean_final += probability * bankroll
        ruin += probability if touched else 0.0
        below_start += probability if bankroll < 1 - 1e-12 else 0.0
    return {"mean_final": mean_final, "ruin": ruin, "below_start": below_start}


def expected_final_fixed_fraction(spec: BetSpec, fraction: float, bets: int) -> float:
    """Exact expected final bankroll (from 1): each bet multiplies it by 1 + fraction*edge on average,
    and the bets are independent, so the expectation is (1 + fraction*edge)^bets."""
    return (1 + fraction * spec.edge) ** bets
