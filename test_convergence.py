"""Convergence analysis: slope, error band, and the 95%-interval coverage test."""

from math import pi, sqrt

import pytest

from app.probability import exact
from app.probability.convergence import (
    convergence_study,
    loglog_slope,
    percentile,
)
from app.probability.montecarlo import birthday_trial, dice_sum_trial, log_spaced_sizes

# ------------------------------------------------------------------- helpers


def test_percentile_known_values():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 0) == 1.0
    assert percentile(values, 25) == 2.0
    assert percentile(values, 50) == 3.0
    assert percentile(values, 100) == 5.0
    assert percentile([0.0, 10.0], 50) == 5.0  # linear interpolation
    assert percentile([7.0], 95) == 7.0


def test_percentile_rejects_bad_input():
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0, 2.0], 101)


def test_loglog_slope_recovers_known_power_laws():
    xs = [10.0, 100.0, 1_000.0, 10_000.0]
    assert loglog_slope(xs, [x**-0.5 for x in xs]) == pytest.approx(-0.5)
    assert loglog_slope(xs, [3 * x**-1 for x in xs]) == pytest.approx(-1.0)
    assert loglog_slope(xs, [2 * x for x in xs]) == pytest.approx(1.0)


def test_loglog_slope_skips_zero_values_and_needs_two_points():
    assert loglog_slope([1, 10, 100], [0.0, 0.1, 0.01]) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        loglog_slope([10], [0.5])
    with pytest.raises(ValueError):
        loglog_slope([10, 100], [0.0, 0.0])


# --------------------------------------------------------------------- study

SIZES = log_spaced_sizes(5_000)
DICE_TRUTH = float(exact.dice_sum_probability(2, 7))
BIRTHDAY_TRUTH = float(exact.birthday_match_probability(23))


def test_study_structure_and_reproducibility():
    a = convergence_study(dice_sum_trial(2, 7), DICE_TRUTH, [10, 100, 1000], 30, seed=1, label="x")
    b = convergence_study(dice_sum_trial(2, 7), DICE_TRUTH, [10, 100, 1000], 30, seed=1, label="x")
    assert a == b
    assert [s.n for s in a.stats] == [10, 100, 1000]
    assert a.replicates == 30 and a.exact == DICE_TRUTH
    for s in a.stats:
        assert 0 <= s.error_p05 <= s.error_p95
        assert s.mean_abs_error >= 0 and s.rmse >= s.mean_abs_error - 1e-12
        assert 0 <= s.coverage <= 1


def test_error_shrinks_as_n_grows():
    result = convergence_study(dice_sum_trial(2, 7), DICE_TRUTH, SIZES, 100, seed=3)
    assert result.stats[-1].mean_abs_error < result.stats[0].mean_abs_error / 5


def test_reference_line_formula():
    result = convergence_study(dice_sum_trial(2, 7), DICE_TRUTH, [10, 100], 5, seed=1)
    expected = sqrt(2 / pi) * sqrt(DICE_TRUTH * (1 - DICE_TRUTH) / 100)
    assert result.reference_error(100) == pytest.approx(expected)
    assert result.reference_error(400) == pytest.approx(result.reference_error(100) / 2)  # 4x n -> half error


@pytest.mark.parametrize(
    ("trial", "truth"),
    [(dice_sum_trial(2, 7), DICE_TRUTH), (birthday_trial(23), BIRTHDAY_TRUTH)],
)
def test_error_falls_like_one_over_sqrt_n(trial, truth):
    """On a log-log plot the slope should be about -1/2."""
    result = convergence_study(trial, truth, SIZES, 200, seed=11)
    assert -0.56 < result.slope < -0.44


def test_mean_error_tracks_the_theory_line():
    result = convergence_study(birthday_trial(23), BIRTHDAY_TRUTH, SIZES, 200, seed=11)
    for s in result.stats:
        if s.n >= 100:  # at tiny n the normal approximation is crude
            assert 0.7 < s.mean_abs_error / result.reference_error(s.n) < 1.3


# ------------------------------------------------------------------- coverage


@pytest.mark.parametrize(
    ("trial", "truth"),
    [(dice_sum_trial(2, 7), DICE_TRUTH), (birthday_trial(23), BIRTHDAY_TRUTH)],
)
def test_ninety_five_percent_intervals_cover_the_truth_about_95_percent_of_the_time(trial, truth):
    result = convergence_study(trial, truth, SIZES, 200, seed=21)
    assert 0.92 < result.overall_coverage < 0.98
    assert all(s.coverage > 0.85 for s in result.stats)  # no single size is badly off


def test_coverage_test_can_detect_a_wrong_reference_answer():
    """The check has teeth: against a WRONG 'exact' value (0.2 instead of 1/6) it fails."""
    wrong = convergence_study(dice_sum_trial(2, 7), 0.2, SIZES, 200, seed=21)
    assert wrong.stats[-1].coverage < 0.5
    assert wrong.overall_coverage < 0.8


def test_study_rejects_bad_arguments():
    with pytest.raises(ValueError):
        convergence_study(dice_sum_trial(2, 7), 0.0, [10], 10, seed=1)
    with pytest.raises(ValueError):
        convergence_study(dice_sum_trial(2, 7), 1.0, [10], 10, seed=1)
    with pytest.raises(ValueError):
        convergence_study(dice_sum_trial(2, 7), 0.5, [10], 1, seed=1)
