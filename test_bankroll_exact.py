"""The exact reference results, verified independently of the simulator."""

import pytest

from app.bankroll.exact import enumerate_fixed_fraction, expected_final_fixed_fraction, gamblers_ruin_probability
from app.bankroll.model import BetSpec


def ruin_by_solving_the_markov_chain(p, start, target):
    """An INDEPENDENT calculation: solve  r[i] = p*r[i+1] + (1-p)*r[i-1]  with r[0]=1, r[N]=0 by
    repeated sweeps (Gauss-Seidel) until it stops changing. Shares no code with the closed formula."""
    r = [1.0] + [0.5] * (target - 1) + [0.0]
    for _ in range(200_000):
        change = 0.0
        for i in range(1, target):
            new = p * r[i + 1] + (1 - p) * r[i - 1]
            change = max(change, abs(new - r[i]))
            r[i] = new
        if change < 1e-14:
            break
    return r[start]


@pytest.mark.parametrize(("p", "start", "target"), [(0.55, 10, 20), (0.45, 10, 20), (0.6, 3, 12), (0.52, 25, 30), (0.3, 4, 9), (0.7, 1, 6)])
def test_gamblers_ruin_formula_agrees_with_an_independent_markov_chain_solver(p, start, target):
    assert gamblers_ruin_probability(p, start, target) == pytest.approx(ruin_by_solving_the_markov_chain(p, start, target), abs=1e-9)


def test_a_fair_game_ruins_in_proportion_to_the_distance():
    assert gamblers_ruin_probability(0.5, 10, 20) == pytest.approx(0.5)
    assert gamblers_ruin_probability(0.5, 5, 20) == pytest.approx(0.75)
    assert ruin_by_solving_the_markov_chain(0.5, 5, 20) == pytest.approx(0.75, abs=1e-9)


def test_play_forever_ruin_probability():
    assert gamblers_ruin_probability(0.55, 10) == pytest.approx((0.45 / 0.55) ** 10)
    assert gamblers_ruin_probability(0.5, 10) == 1.0 and gamblers_ruin_probability(0.4, 3) == 1.0
    # more room to fall means a smaller chance of ever getting there
    assert gamblers_ruin_probability(0.55, 20) < gamblers_ruin_probability(0.55, 10)


def test_gamblers_ruin_rejects_bad_arguments():
    for args in ((0, 5, 10), (1, 5, 10), (0.5, 0, 10), (0.5, 10, 10), (0.5, 12, 10)):
        with pytest.raises(ValueError):
            gamblers_ruin_probability(*args)


@pytest.mark.parametrize(
    ("spec", "fraction", "bets"),
    [(BetSpec(0.55, 0.45, 1), 0.1, 8), (BetSpec(0.4, 0.4, 2), 0.2, 7), (BetSpec(0.2, 0.8, 5), 0.05, 8), (BetSpec(0.7, 0.3, 1), 0.5, 8)],
)
def test_expected_final_bankroll_formula_matches_full_enumeration(spec, fraction, bets):
    """The formula (1 + f*edge)^T against trying every one of the 2^T or 3^T possible sequences."""
    exact = enumerate_fixed_fraction(spec, fraction, bets, ruin_fraction=0.5)
    assert exact["mean_final"] == pytest.approx(expected_final_fixed_fraction(spec, fraction, bets), rel=1e-12)


def test_enumeration_probabilities_are_complete_and_sensible():
    spec = BetSpec(0.5, 0.5, 1)
    out = enumerate_fixed_fraction(spec, 0.5, 4, ruin_fraction=0.2)
    assert 0 <= out["ruin"] <= 1 and 0 <= out["below_start"] <= 1
    # Hand check: 4 fair bets of half the bankroll multiply it by 1.5 (win) or 0.5 (lose).
    # Three losses leave 1.5 * 0.5^3 = 0.1875 <= 0.2 whatever the order, and four leave 0.0625, so every
    # sequence with 3 or more losses touches the ruin level: C(4,3) + C(4,4) = 5 of the 16.
    assert out["ruin"] == pytest.approx(5 / 16)


def test_expected_value_formula_for_zero_fraction_and_no_edge():
    assert expected_final_fixed_fraction(BetSpec(0.55, 0.45), 0.0, 100) == 1.0
    assert expected_final_fixed_fraction(BetSpec(0.5, 0.5), 0.3, 100) == 1.0  # a fair bet keeps the same average
