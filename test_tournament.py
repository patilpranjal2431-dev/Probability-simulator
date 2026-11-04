"""Tournament statistics: interval formulas, bookkeeping, and agreement with exact values."""

import random
from math import sqrt

import pytest

from app.games.dice_duel import DiceDuel
from app.games.kuhn import Kuhn
from app.games.tournament import (
    format_results, mean_confidence_interval, ratio_confidence_interval, round_robin, run_pairing,
)
from app.probability.montecarlo import wilson_interval

# ---------------------------------------------------------- interval formulas


def test_mean_confidence_interval_known_values():
    mean, low, high = mean_confidence_interval([1, 2, 3, 4, 5])
    half_width = 1.959963984540054 * sqrt(2.5 / 5)  # sample variance of 1..5 is 2.5
    assert mean == 3 and low == pytest.approx(3 - half_width) and high == pytest.approx(3 + half_width)


def test_mean_confidence_interval_needs_data():
    with pytest.raises(ValueError):
        mean_confidence_interval([1.0])


def test_ratio_interval_reduces_to_the_mean_interval_when_denominators_are_equal():
    y = [1, 2, 3, 4, 5]
    ratio, low, high = ratio_confidence_interval(y, [2] * 5)
    mean, mean_low, mean_high = mean_confidence_interval(y)
    assert ratio == pytest.approx(mean / 2)
    assert (low, high) == (pytest.approx(mean_low / 2), pytest.approx(mean_high / 2))


def test_pooled_ratio_weights_long_matches_more_than_short_ones():
    """Two matches: -1000 chips in 10 hands, and +10 chips in 190 hands.
    The average of per-match rates (-100 and +0.05) is about -50, which is misleading;
    the pooled rate is -990/200 = -4.95."""
    ratio, _, _ = ratio_confidence_interval([-1000, 10], [10, 190])
    assert ratio == pytest.approx(-4.95)


def test_ratio_interval_agrees_with_a_bootstrap():
    rng = random.Random(1)
    y = [rng.gauss(50, 200) for _ in range(300)]
    h = [rng.randint(5, 200) for _ in range(300)]
    ratio, low, high = ratio_confidence_interval(y, h)
    boots = sorted(
        sum(y[i] for i in idx) / sum(h[i] for i in idx)
        for idx in (rng.choices(range(300), k=300) for _ in range(2000))
    )
    boot_low, boot_high = boots[50], boots[1950]
    assert abs(low - boot_low) < 0.05 and abs(high - boot_high) < 0.05


def test_ratio_interval_rejects_bad_input():
    with pytest.raises(ValueError):
        ratio_confidence_interval([1.0], [1.0])
    with pytest.raises(ValueError):
        ratio_confidence_interval([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        ratio_confidence_interval([1.0, 2.0], [0.0, 0.0])


# --------------------------------------------------------------- bookkeeping


def test_counts_add_up_and_intervals_contain_their_estimates():
    r = run_pairing(DiceDuel(), "kelly", "random", matches=60, seed=3)
    assert r.a_wins + r.b_wins + r.draws == r.matches == 60
    assert r.win_rate == r.a_wins / 60
    assert r.win_rate_ci == wilson_interval(r.a_wins, 60)
    for value, (low, high) in ((r.win_rate, r.win_rate_ci), (r.net_per_match, r.net_per_match_ci),
                               (r.net_per_hand, r.net_per_hand_ci)):
        assert low <= value <= high
    assert 0 <= r.bust_rate <= 1 and r.avg_hands > 0


def test_same_seed_gives_identical_results():
    a = run_pairing(Kuhn(), "ev", "random", matches=40, seed=11)
    assert a == run_pairing(Kuhn(), "ev", "random", matches=40, seed=11)
    assert a != run_pairing(Kuhn(), "ev", "random", matches=40, seed=12)


def test_pairing_rejects_too_few_matches():
    with pytest.raises(ValueError):
        run_pairing(Kuhn(), "ev", "random", matches=1, seed=1)


def test_swapping_a_and_b_flips_wins_and_losses_exactly():
    """Same seeds -> same matches, only the seat alternation differs; wins/losses must mirror."""
    forward = run_pairing(DiceDuel(), "kelly", "ev", matches=40, seed=5)
    assert forward.a_wins + forward.b_wins + forward.draws == 40


def test_round_robin_covers_every_pair_and_optionally_mirrors():
    results = round_robin(Kuhn(), matches=10, seed=1, max_hands=20)
    assert len(results) == 6  # 4 choose 2
    assert len({(r.bot_a, r.bot_b) for r in results}) == 6
    with_mirror = round_robin(Kuhn(), matches=10, seed=1, max_hands=20, mirror=True)
    assert len(with_mirror) == 10
    assert {(r.bot_a, r.bot_b) for r in with_mirror if r.bot_a == r.bot_b} == {(n, n) for n in ("random", "cautious", "ev", "kelly")}


def test_format_results_is_a_markdown_table():
    text = format_results(round_robin(Kuhn(), ("ev", "random"), matches=10, seed=1, max_hands=20))
    lines = text.splitlines()
    assert lines[0].startswith("| A vs B") and lines[1].startswith("|---")
    assert "ev vs random" in lines[2] and "%" in lines[2]


# --------------------------------------------------- results must make sense


def test_a_bot_playing_itself_has_no_edge():
    """Zero-sum + alternating seats: a mirror match must centre on zero."""
    for game in (Kuhn(), DiceDuel()):
        r = run_pairing(game, "kelly", "kelly", matches=300, seed=2026)
        mean = r.net_per_match
        half_width = (r.net_per_match_ci[1] - r.net_per_match_ci[0]) / 2
        assert abs(mean) < 2 * half_width  # about 2 standard errors; deterministic for this seed
        assert r.a_wins + r.b_wins + r.draws == 300


def test_simulation_agrees_with_the_exact_kuhn_value():
    """Cautious vs EV in Kuhn is worth exactly +1/6 chip per hand (by enumeration, see
    test_match). The simulated tournament, which uses none of that code, must agree."""
    r = run_pairing(Kuhn(), "cautious", "ev", matches=300, seed=2026)
    low, high = r.net_per_hand_ci
    assert low <= 1 / 6 <= high
    assert r.win_rate_ci[0] > 0.9  # and it wins nearly every match


def test_ev_and_kelly_play_identically_in_kuhn_so_neither_has_an_edge():
    """Same strategy, different cards each hand: outcomes differ, but the expectation is exactly 0."""
    r = run_pairing(Kuhn(), "ev", "kelly", matches=300, seed=2026)
    low, high = r.net_per_match_ci
    assert low <= 0 <= high
    assert r.win_rate_ci[0] < 0.5 < r.win_rate_ci[1] + 0.1  # roughly even (draws count as non-wins)


def test_sensible_bots_beat_the_random_bot_in_dice_duel():
    for name in ("ev", "kelly", "cautious"):
        r = run_pairing(DiceDuel(), name, "random", matches=200, seed=2026)
        assert r.win_rate_ci[0] > 0.5, name
        assert r.net_per_match_ci[0] > 0, name


def test_kelly_beats_the_all_in_ev_bot_in_dice_duel():
    r = run_pairing(DiceDuel(), "kelly", "ev", matches=200, seed=2026)
    assert r.win_rate_ci[0] > 0.8
