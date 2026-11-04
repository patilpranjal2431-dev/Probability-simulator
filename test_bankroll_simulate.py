"""The Monte Carlo simulator, checked against exact answers and against Kelly's theory."""

import json
from math import log, sqrt

import pytest

from app.bankroll.exact import enumerate_fixed_fraction, expected_final_fixed_fraction, gamblers_ruin_probability
from app.bankroll.model import BetSpec
from app.bankroll.simulate import MAX_STEPS, Strategy, simulate_bankroll

SPEC = BetSpec(0.55, 0.45, 1)


def standard_error(p, n):
    return sqrt(p * (1 - p) / n)


# ============================================================ validation


def test_strategy_validation():
    for kind, value in (("kelly", 0), ("kelly", 6), ("fraction", 0), ("fraction", 1.5), ("flat", -1), ("flat", True), ("martingale", 1), ("kelly", "1")):
        with pytest.raises(ValueError):
            Strategy(kind, value)
    assert Strategy("kelly", 0.5).stake_fraction(SPEC) == pytest.approx(0.05)
    assert Strategy("fraction", 0.2).stake_fraction(SPEC) == 0.2
    assert Strategy("flat", 5).stake_fraction(SPEC) is None
    assert Strategy("kelly", 5).stake_fraction(BetSpec(0.9, 0.1)) == 1.0  # capped at everything


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(bets=0, paths=10), dict(bets=10, paths=0), dict(bets=10.5, paths=10), dict(bets=True, paths=10),
        dict(bets=5001, paths=10), dict(bets=10, paths=20_001), dict(bets=2000, paths=3000),
        dict(bets=10, paths=10, start=0), dict(bets=10, paths=10, ruin_fraction=1.0),
        dict(bets=10, paths=10, ruin_fraction=-0.1), dict(bets=10, paths=10, target=500, start=1000),
    ],
)
def test_bad_simulation_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        simulate_bankroll(SPEC, Strategy("kelly", 1), **kwargs)


def test_the_total_amount_of_work_is_capped():
    assert 2000 * 3000 > MAX_STEPS


def test_same_seed_gives_identical_results_and_a_different_seed_does_not():
    a = simulate_bankroll(SPEC, Strategy("kelly", 1), 50, 200, seed=4)
    assert a == simulate_bankroll(SPEC, Strategy("kelly", 1), 50, 200, seed=4)
    assert a.finals != simulate_bankroll(SPEC, Strategy("kelly", 1), 50, 200, seed=5).finals


# ================================================ against the exact gambler's ruin


@pytest.mark.parametrize("p", [0.55, 0.5, 0.45])
def test_flat_betting_matches_the_gamblers_ruin_formula(p):
    """Bet 1 coin at even money from 5 coins; stop at 10 or at 0. Compare with the closed formula.
    (A narrow range, so that essentially every path has finished by 300 bets: a fair game absorbs slowly.)"""
    spec = BetSpec(p, 1 - p, 1)
    result = simulate_bankroll(spec, Strategy("flat", 1), bets=300, paths=10_000, start=5, ruin_fraction=0,
                               target=10, seed=11)
    exact = gamblers_ruin_probability(p, 5, 10)
    assert abs(result.ruin_probability - exact) < 4 * standard_error(exact, 10_000)
    assert abs(result.target_probability - (1 - exact)) < 4 * standard_error(exact, 10_000)
    assert result.ruin_probability + result.target_probability == pytest.approx(1.0, abs=0.001)  # every path finished


def test_flat_betting_with_no_profit_goal_approaches_the_play_forever_formula():
    spec = BetSpec(0.55, 0.45, 1)
    result = simulate_bankroll(spec, Strategy("flat", 1), bets=1200, paths=3000, start=5, ruin_fraction=0, seed=2)
    exact = gamblers_ruin_probability(0.55, 5)
    assert abs(result.ruin_probability - exact) < 4 * standard_error(exact, 3000) + 0.01  # a little left to happen after 1200 bets


def test_a_flat_bettor_who_hits_zero_stays_at_zero():
    spec = BetSpec(0.45, 0.55, 1)
    result = simulate_bankroll(spec, Strategy("flat", 1), bets=200, paths=500, start=3, ruin_fraction=0, seed=1)
    assert result.fraction_wiped_out == pytest.approx(result.ruin_probability)  # ruin level 0 = nothing left
    assert min(result.finals) == 0.0 and result.mean_log_growth is None  # undefined when a path hit zero


# ======================================================= against full enumeration


@pytest.mark.parametrize(
    ("spec", "fraction", "bets", "ruin_fraction"),
    [(BetSpec(0.55, 0.45, 1), 0.2, 8, 0.5), (BetSpec(0.4, 0.4, 2), 0.25, 7, 0.4), (BetSpec(0.6, 0.4, 1), 0.5, 8, 0.3)],
)
def test_every_statistic_matches_exact_enumeration_of_all_sequences(spec, fraction, bets, ruin_fraction):
    n = 20_000
    exact = enumerate_fixed_fraction(spec, fraction, bets, ruin_fraction)
    result = simulate_bankroll(spec, Strategy("fraction", fraction), bets, n, start=1.0, ruin_fraction=ruin_fraction, seed=7)
    assert abs(result.ruin_probability - exact["ruin"]) < 4 * standard_error(exact["ruin"], n)
    assert abs(result.prob_below_start - exact["below_start"]) < 4 * standard_error(exact["below_start"], n)
    mean = sum(result.finals) / n
    sd = sqrt(sum((v - mean) ** 2 for v in result.finals) / (n - 1))
    assert abs(result.mean_final - exact["mean_final"]) < 4 * sd / sqrt(n)


def test_all_in_on_every_bet_matches_the_exact_survival_chance():
    """Stake everything on 5 bets at 55%: you survive only by winning all 5."""
    result = simulate_bankroll(SPEC, Strategy("fraction", 1.0), 5, 20_000, start=100, ruin_fraction=0, seed=3)
    survive = 0.55**5
    assert abs((1 - result.ruin_probability) - survive) < 4 * standard_error(survive, 20_000)
    assert result.fraction_wiped_out == pytest.approx(result.ruin_probability)


# ===================================================== against Kelly's theory


@pytest.mark.parametrize("multiple", [0.25, 0.5, 1.0, 2.0, 3.0])
def test_measured_growth_rate_matches_the_theoretical_growth_rate(multiple):
    result = simulate_bankroll(SPEC, Strategy("kelly", multiple), bets=150, paths=4000, seed=5)
    logs = [log(v / result.start) / 150 for v in result.finals]
    mean = sum(logs) / len(logs)
    sd = sqrt(sum((x - mean) ** 2 for x in logs) / (len(logs) - 1))
    assert result.theory_growth_rate == pytest.approx(SPEC.growth_rate(result.stake_fraction))
    assert abs(result.mean_log_growth - result.theory_growth_rate) < 4 * sd / sqrt(len(logs))


def test_kelly_gives_the_best_median_outcome_and_overbetting_destroys_it():
    medians = {m: simulate_bankroll(SPEC, Strategy("kelly", m), 200, 3000, seed=1).median_final for m in (0.5, 1.0, 2.0, 3.0)}
    assert medians[1.0] > medians[0.5] > medians[2.0] > medians[3.0]
    assert medians[3.0] < 100  # three times Kelly: the typical bankroll collapses despite a 55% edge


def test_overbetting_looks_great_on_average_while_the_typical_path_is_ruined():
    over = simulate_bankroll(SPEC, Strategy("kelly", 3.0), 200, 3000, seed=1)
    assert over.mean_final > 100 * over.median_final  # the mean is dragged up by a few lucky paths


def test_ruin_and_drawdown_rise_with_the_stake():
    results = [simulate_bankroll(SPEC, Strategy("kelly", m), 200, 3000, seed=9) for m in (0.25, 0.5, 1.0, 2.0, 3.0)]
    ruin = [r.ruin_probability for r in results]
    drawdown = [r.median_max_drawdown for r in results]
    assert ruin == sorted(ruin) and ruin[-1] - ruin[2] > 0.3
    assert drawdown == sorted(drawdown) and drawdown[2] > 0.5  # even full Kelly means painful drawdowns


def test_mean_final_matches_theory_when_the_stake_is_small_enough_to_measure():
    result = simulate_bankroll(SPEC, Strategy("fraction", 0.02), 100, 5000, start=1000, seed=6)
    mean = result.mean_final
    sd = sqrt(sum((v - mean) ** 2 for v in result.finals) / (len(result.finals) - 1))
    assert result.theory_mean_final == pytest.approx(1000 * expected_final_fixed_fraction(SPEC, 0.02, 100))
    assert abs(mean - result.theory_mean_final) < 4 * sd / sqrt(5000)


def test_with_no_edge_the_kelly_stake_is_zero_and_nothing_moves():
    result = simulate_bankroll(BetSpec(0.5, 0.5, 1), Strategy("kelly", 1), 100, 50, start=500, seed=1)
    assert result.stake_fraction == 0 and set(result.finals) == {500.0}
    assert result.ruin_probability == 0 and result.median_max_drawdown == 0


def test_a_losing_bet_staked_anyway_loses_money_and_theory_agrees():
    spec = BetSpec(0.45, 0.55, 1)
    result = simulate_bankroll(spec, Strategy("fraction", 0.1), 100, 2000, seed=2)
    assert result.theory_growth_rate < 0 and result.mean_log_growth < 0
    assert result.median_final < result.start


# ================================================================ the outputs


def test_the_quantile_series_and_sample_paths_are_well_formed():
    result = simulate_bankroll(SPEC, Strategy("kelly", 1), 120, 500, start=1000, seed=3, sample_paths=5, grid_points=30)
    assert result.times[0] == 0 and result.times[-1] == 120 and result.times == sorted(set(result.times))
    for q, values in result.series.items():
        assert len(values) == len(result.times) and values[0] == 1000
    for i in range(len(result.times)):
        row = [result.series[q][i] for q in (5, 25, 50, 75, 95)]
        assert row == sorted(row)  # percentiles never cross
    assert len(result.sample_paths) == 5 and all(len(p) == len(result.times) for p in result.sample_paths)
    assert result.series[50][-1] == pytest.approx(result.median_final)


def test_final_quantiles_and_basic_probabilities_are_consistent():
    result = simulate_bankroll(SPEC, Strategy("kelly", 1), 100, 1000, seed=3)
    q = result.final_quantiles
    assert q[5] <= q[25] <= q[50] <= q[75] <= q[95] and result.median_final == q[50]
    assert result.finals == sorted(result.finals)
    assert result.prob_below_start == pytest.approx(sum(v < 1000 for v in result.finals) / 1000)
    assert result.prob_double == pytest.approx(sum(v >= 2000 for v in result.finals) / 1000)
    low, high = result.ruin_ci
    assert low <= result.ruin_probability <= high


def test_the_profit_goal_stops_paths_and_is_reported():
    result = simulate_bankroll(SPEC, Strategy("kelly", 1), 300, 1000, start=1000, target=1500, ruin_fraction=0.5, seed=4)
    assert 0 < result.target_probability <= 1
    assert max(result.finals) < 1500 * 1.11  # a path stops as soon as it reaches the goal
    assert simulate_bankroll(SPEC, Strategy("kelly", 1), 20, 20).target_probability is None


def test_time_to_ruin_is_reported_only_when_there_is_ruin():
    ruined = simulate_bankroll(SPEC, Strategy("kelly", 3.0), 200, 1000, seed=1)
    assert 1 <= ruined.median_time_to_ruin <= 200
    safe = simulate_bankroll(SPEC, Strategy("kelly", 0.25), 50, 200, seed=1)
    assert safe.ruin_probability == 0 and safe.median_time_to_ruin is None


def test_the_summary_is_json_safe_even_when_values_are_undefined():
    wiped = simulate_bankroll(SPEC, Strategy("fraction", 1.0), 20, 100, start=10, ruin_fraction=0, seed=1)
    assert wiped.theory_growth_rate == float("-inf")  # staking everything: one loss ends it
    text = json.dumps(wiped.summary(), allow_nan=False)  # raises if any inf/NaN slipped through
    assert "NaN" not in text and "Infinity" not in text
    summary = wiped.summary()
    assert summary["theory_growth_rate"] is None and summary["mean_log_growth"] is None and summary["strategy"] == {"kind": "fraction", "value": 1.0}
    assert summary["bet"]["edge"] == pytest.approx(0.1) and summary["final_quantiles"]["50"] == wiped.median_final


def test_a_flat_bettor_can_never_stake_more_than_they_have():
    """A 7-coin stake from a 10-coin bankroll: after a loss only 3 remain, so the next stake is 3, not 7.
    Without that cap a bankroll could go negative."""
    spec = BetSpec(0.4, 0.6, 1)
    result = simulate_bankroll(spec, Strategy("flat", 7), bets=40, paths=2000, start=10, ruin_fraction=0, seed=8)
    assert min(result.finals) == 0.0 and all(v >= 0 for v in result.finals)
    assert all(v >= 0 for path in result.sample_paths for v in path)
    assert all(v >= 0 for q in result.series.values() for v in q)


def test_drawdown_is_measured_from_the_previous_peak_not_from_the_start():
    """A bet that only ever wins can have no drawdown, however large the bankroll grows."""
    sure_thing = simulate_bankroll(BetSpec(1.0, 0.0, 1), Strategy("fraction", 0.5), 30, 20, start=100, seed=1)
    assert sure_thing.median_max_drawdown == 0 and sure_thing.p90_max_drawdown == 0
    assert sure_thing.median_final == pytest.approx(100 * 1.5**30)
    # and a single guaranteed 50% loss from the start is a drawdown of exactly 50%
    always_lose = simulate_bankroll(BetSpec(0.0, 1.0, 1), Strategy("fraction", 0.5), 1, 20, start=100, seed=1)
    assert always_lose.median_max_drawdown == pytest.approx(0.5) and always_lose.median_final == 50


def test_each_percentile_series_ends_at_its_own_final_quantile_and_the_bands_have_real_width():
    """Catches a bug where every band collapses onto the median: 'sorted' alone is satisfied by equal values."""
    result = simulate_bankroll(SPEC, Strategy("kelly", 1), 150, 1500, start=1000, seed=12)
    for q in (5, 25, 50, 75, 95):
        assert result.series[q][-1] == pytest.approx(result.final_quantiles[q])  # the last column IS the finals
    assert result.series[5][-1] < result.series[25][-1] < result.series[50][-1] < result.series[75][-1] < result.series[95][-1]
    middle = len(result.times) // 2
    assert result.series[5][middle] < result.series[95][middle]  # and the fan widens over time:
    assert (result.series[95][-1] / result.series[5][-1]) > (result.series[95][middle] / result.series[5][middle])
