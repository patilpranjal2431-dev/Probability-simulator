"""Bets, the Kelly fraction, and the growth rate: checked against textbook values and by brute force."""

from math import log

import pytest

from app.bankroll.model import BetSpec


def test_kelly_matches_textbook_values():
    assert BetSpec(0.55, 0.45, 1).kelly_fraction() == pytest.approx(0.10)  # (bp - q)/b = (0.55 - 0.45)/1
    assert BetSpec(0.4, 0.6, 2).kelly_fraction() == pytest.approx((2 * 0.4 - 0.6) / 2)  # 0.1
    assert BetSpec(0.2, 0.8, 5).kelly_fraction() == pytest.approx(0.04)  # a 5-to-1 long shot
    assert BetSpec(0.6, 0.4, 1).kelly_fraction() == pytest.approx(0.2)


def test_kelly_with_pushes_matches_the_dice_duel_formula_used_by_the_kelly_bot():
    """Phase 4's bot staked (win - lose) / (win + lose) of its bankroll; this must agree."""
    for roll in range(4, 7):
        win, lose = (roll - 1) / 6, (6 - roll) / 6
        assert BetSpec(win, lose, 1).kelly_fraction() == pytest.approx((win - lose) / (win + lose))
    assert BetSpec(4 / 6, 1 / 6, 1).kelly_fraction() == pytest.approx(0.6)


def test_no_edge_or_a_negative_edge_means_stake_nothing():
    assert BetSpec(0.5, 0.5, 1).kelly_fraction() == 0
    assert BetSpec(0.4, 0.6, 1).kelly_fraction() == 0
    assert BetSpec(2 / 6, 3 / 6, 1).kelly_fraction() == 0  # a Dice Duel 3


def test_a_bet_that_cannot_lose_says_stake_everything():
    assert BetSpec(0.7, 0.0, 1).kelly_fraction() == pytest.approx(1.0)


def test_kelly_is_exactly_where_growth_peaks():
    """Brute force: the best of 100,000 candidate fractions is the Kelly fraction."""
    for spec in (BetSpec(0.55, 0.45, 1), BetSpec(0.3, 0.7, 4), BetSpec(0.45, 0.35, 1.5)):
        grid = [i / 100_000 for i in range(0, 99_000)]
        best = max(grid, key=spec.growth_rate)
        assert best == pytest.approx(spec.kelly_fraction(), abs=2e-5)


def test_growth_rate_matches_the_formula_by_hand():
    spec = BetSpec(0.55, 0.45, 1)
    assert spec.growth_rate(0.1) == pytest.approx(0.55 * log(1.1) + 0.45 * log(0.9))
    assert spec.growth_rate(0) == 0
    assert BetSpec(0.4, 0.4, 2).growth_rate(0.1) == pytest.approx(0.4 * log(1.2) + 0.4 * log(0.9))  # a push adds nothing


def test_staking_everything_on_a_bet_that_can_lose_has_no_growth():
    assert BetSpec(0.55, 0.45, 1).growth_rate(1) == float("-inf")
    assert BetSpec(0.7, 0.0, 1).growth_rate(1) == pytest.approx(0.7 * log(2))


def test_the_break_even_fraction_is_about_twice_kelly_for_a_small_edge():
    spec = BetSpec(0.505, 0.495, 1)
    star, even = spec.kelly_fraction(), spec.breakeven_fraction()
    assert spec.growth_rate(even) == pytest.approx(0, abs=1e-9)
    assert even / star == pytest.approx(2.0, rel=0.01)
    bigger = BetSpec(0.55, 0.45, 1)
    assert 1.9 < bigger.breakeven_fraction() / bigger.kelly_fraction() < 2.0


def test_betting_more_than_the_break_even_fraction_loses_money_even_though_every_bet_is_good():
    spec = BetSpec(0.55, 0.45, 1)
    star, even = spec.kelly_fraction(), spec.breakeven_fraction()
    assert spec.growth_rate(star) > 0 and spec.growth_rate(even * 0.99) > 0
    assert spec.growth_rate(even * 1.05) < 0 and spec.growth_rate(3 * star) < 0
    assert spec.edge > 0  # yet the bet itself is favourable


def test_half_kelly_keeps_about_three_quarters_of_the_growth():
    spec = BetSpec(0.55, 0.45, 1)
    ratio = spec.growth_rate(spec.kelly_fraction() / 2) / spec.growth_rate(spec.kelly_fraction())
    assert ratio == pytest.approx(0.75, abs=0.01)


def test_break_even_is_none_when_there_is_no_edge_or_no_risk():
    assert BetSpec(0.5, 0.5).breakeven_fraction() is None
    assert BetSpec(0.7, 0.0).breakeven_fraction() is None


def test_probabilities_push_and_edge():
    spec = BetSpec(0.4, 0.4, 2)
    assert spec.push == pytest.approx(0.2) and spec.edge == pytest.approx(2 * 0.4 - 0.4)
    assert BetSpec(0.5, 0.5).push == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(win=-0.1, lose=0.5), dict(win=0.5, lose=-0.1), dict(win=0.7, lose=0.5), dict(win=0.0, lose=0.0),
        dict(win=0.5, lose=0.5, odds=0), dict(win=0.5, lose=0.5, odds=-1), dict(win=0.5, lose=0.5, odds=1001),
        dict(win=float("nan"), lose=0.5), dict(win=float("inf"), lose=0.0), dict(win="0.5", lose=0.5),
        dict(win=True, lose=0.0), dict(win=None, lose=0.5),
    ],
)
def test_nonsense_bets_are_rejected(kwargs):
    with pytest.raises(ValueError):
        BetSpec(**kwargs)


def test_growth_rate_rejects_a_fraction_outside_zero_to_one():
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            BetSpec(0.55, 0.45).growth_rate(bad)
