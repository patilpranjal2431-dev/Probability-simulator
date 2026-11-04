"""Seeded Monte Carlo, Wilson intervals, shuffling and agreement with exact values."""

import random
from math import sqrt

import pytest

from app.probability import exact
from app.probability.montecarlo import (
    CardDrawer,
    birthday_trial,
    dice_sum_trial,
    hypergeometric_trial,
    log_spaced_sizes,
    make_estimate,
    poker_trial,
    simulate,
    wilson_interval,
)

# ---------------------------------------------------------------------- Wilson


def test_wilson_known_values():
    low, high = wilson_interval(0, 10)
    assert low == 0.0
    assert high == pytest.approx(0.27753, abs=1e-4)
    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(0.23659, abs=1e-4)
    assert high == pytest.approx(0.76341, abs=1e-4)


def test_wilson_with_no_data_is_the_whole_range():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_stays_inside_zero_and_one_and_contains_the_estimate():
    for n in (1, 2, 5, 30, 400):
        for successes in range(0, n + 1, max(1, n // 7)):
            low, high = wilson_interval(successes, n)
            assert 0.0 <= low <= successes / n <= high <= 1.0


def test_wilson_is_not_degenerate_when_nothing_was_seen():
    low, high = wilson_interval(0, 50)
    assert high > 0.05  # the simple +/- interval would wrongly give zero width here


def test_wilson_narrows_as_n_grows():
    widths = []
    for n in (100, 1_000, 10_000):
        low, high = wilson_interval(n // 4, n)
        widths.append(high - low)
    assert widths[0] > widths[1] > widths[2]


def test_wilson_rejects_bad_input():
    with pytest.raises(ValueError):
        wilson_interval(-1, 10)
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
    with pytest.raises(ValueError):
        wilson_interval(1, -5)
    with pytest.raises(TypeError):
        wilson_interval(1.5, 10)


def test_make_estimate_fields():
    est = make_estimate(30, 100)
    assert (est.n, est.successes) == (100, 30)
    assert est.p_hat == 0.3
    assert est.std_error == pytest.approx(sqrt(0.3 * 0.7 / 100))
    assert est.contains(0.3)
    assert not est.contains(0.9)


# ----------------------------------------------------------------- sample sizes


def test_log_spaced_sizes_known_output():
    assert log_spaced_sizes(10_000) == [10, 18, 32, 56, 100, 178, 316, 562, 1000, 1778, 3162, 5623, 10000]


def test_log_spaced_sizes_always_ends_at_max_and_is_sorted_and_unique():
    for max_n in (10, 11, 99, 12_345):
        sizes = log_spaced_sizes(max_n)
        assert sizes[0] == 10 and sizes[-1] == max_n
        assert sizes == sorted(set(sizes))


def test_log_spaced_sizes_rejects_bad_input():
    with pytest.raises(ValueError):
        log_spaced_sizes(5)  # max below the default min of 10
    with pytest.raises(ValueError):
        log_spaced_sizes(100, points_per_decade=0)


# ------------------------------------------------------------------- simulate


def test_same_seed_gives_identical_results():
    trial = dice_sum_trial(2, 7)
    assert simulate(trial, [10, 100, 1000], seed=5) == simulate(trial, [10, 100, 1000], seed=5)


def test_different_seeds_give_different_results():
    trial = dice_sum_trial(2, 7)
    assert simulate(trial, [1000], seed=1) != simulate(trial, [1000], seed=2)


def test_results_do_not_depend_on_the_global_random_state():
    trial = birthday_trial(23)
    random.seed(111)
    first = simulate(trial, [500], seed=9)
    random.seed(222)
    assert simulate(trial, [500], seed=9) == first


def test_checkpoints_are_cumulative_and_deduplicated():
    estimates = simulate(dice_sum_trial(2, 7), [100, 10, 100, 1000], seed=3)
    assert [e.n for e in estimates] == [10, 100, 1000]
    hits = [e.successes for e in estimates]
    assert hits == sorted(hits)  # running totals never decrease


def test_simulate_rejects_bad_sizes():
    with pytest.raises(ValueError):
        simulate(dice_sum_trial(2, 7), [], seed=1)
    with pytest.raises(ValueError):
        simulate(dice_sum_trial(2, 7), [0, 10], seed=1)


# ------------------------------------------------------------------ shuffling


def test_drawn_cards_are_distinct_and_in_range():
    drawer, rng = CardDrawer(), random.Random(1)
    for _ in range(500):
        cards = drawer.draw(rng, 5)
        assert len(set(cards)) == 5
        assert all(0 <= c < 52 for c in cards)


def test_drawing_zero_or_all_cards():
    drawer, rng = CardDrawer(), random.Random(1)
    assert drawer.draw(rng, 0) == []
    assert sorted(drawer.draw(rng, 52)) == list(range(52))


def test_cannot_draw_more_cards_than_the_deck_holds():
    with pytest.raises(ValueError):
        CardDrawer().draw(random.Random(1), 53)
    with pytest.raises(ValueError):
        CardDrawer().draw(random.Random(1), -1)


def test_each_card_is_equally_likely_to_come_first():
    drawer, rng = CardDrawer(4), random.Random(42)
    tally = [0] * 4
    for _ in range(40_000):
        tally[drawer.draw(rng, 1)[0]] += 1
    assert all(abs(count - 10_000) < 400 for count in tally)  # about 4.6 standard errors


def test_a_fresh_shuffle_makes_all_orderings_equally_likely():
    """Uses a NEW deck for every draw, which isolates the algorithm itself.

    This is what catches the classic 'naive shuffle' bug (swap each card with ANY random
    position): it makes some of the 6 orderings 5/27 likely and others only 4/27, instead
    of 1/6 each. Reusing one deck would hide that, because repeated shuffling drifts
    toward uniform.
    """
    rng = random.Random(42)
    tally: dict[tuple[int, ...], int] = {}
    for _ in range(36_000):
        order = tuple(CardDrawer(3).draw(rng, 3))
        tally[order] = tally.get(order, 0) + 1
    assert len(tally) == 6
    assert all(abs(count - 6_000) < 400 for count in tally.values())  # naive shuffle is off by ~667


def test_a_reused_deck_still_gives_all_orderings_equally_likely():
    """The deck is reused between draws in the simulations, so check that usage too."""
    drawer, rng = CardDrawer(3), random.Random(42)
    tally: dict[tuple[int, ...], int] = {}
    for _ in range(36_000):
        order = tuple(drawer.draw(rng, 3))
        tally[order] = tally.get(order, 0) + 1
    assert len(tally) == 6
    assert all(abs(count - 6_000) < 400 for count in tally.values())


# ------------------------------------------------- Monte Carlo agrees with exact


@pytest.mark.parametrize(
    ("name", "trial", "truth", "n"),
    [
        ("two dice sum 7", dice_sum_trial(2, 7), exact.dice_sum_probability(2, 7), 20_000),
        ("three dice sum 10", dice_sum_trial(3, 10), exact.dice_sum_probability(3, 10), 20_000),
        ("birthday 23", birthday_trial(23), exact.birthday_match_probability(23), 20_000),
        ("two aces", hypergeometric_trial(52, 4, 5, 2), exact.hypergeometric_pmf(52, 4, 5, 2), 20_000),
        ("one pair", poker_trial("one_pair"), exact.poker_hand_probabilities()["one_pair"], 20_000),
        ("flush", poker_trial("flush"), exact.poker_hand_probabilities()["flush"], 50_000),
    ],
)
def test_estimate_is_within_four_standard_errors_of_the_exact_value(name, trial, truth, n):
    truth = float(truth)
    estimate = simulate(trial, [n], seed=2026)[0]
    standard_error = sqrt(truth * (1 - truth) / n)  # uses the TRUE p, not the estimate
    assert abs(estimate.p_hat - truth) < 4 * standard_error, name


def test_trials_reject_bad_arguments():
    with pytest.raises(ValueError):
        dice_sum_trial(0, 5)
    with pytest.raises(ValueError):
        birthday_trial(-1)
    with pytest.raises(ValueError):
        hypergeometric_trial(52, 60, 5, 1)
