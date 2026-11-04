"""Brier scores and calibration statistics: definitions, identities and proper-scoring-rule theory."""

import random

import pytest

from app.calibration.scoring import (
    BASELINE_BRIER, Answer, brier, calibration_bins, excess_brier, expected_brier, series, summarize, uncertainty,
    verdict,
)


def ans(f, p, outcome, kind="duel_win"):
    return Answer(kind, f, p, outcome)


# ---------------------------------------------------------------- definitions


def test_brier_known_values():
    assert brier(1.0, True) == 0 and brier(0.0, False) == 0  # perfect
    assert brier(1.0, False) == 1 and brier(0.0, True) == 1  # confidently wrong
    assert brier(0.7, True) == pytest.approx(0.09) and brier(0.7, False) == pytest.approx(0.49)


def test_always_saying_fifty_percent_scores_exactly_the_baseline_whatever_happens():
    assert brier(0.5, True) == brier(0.5, False) == BASELINE_BRIER == 0.25


def test_expected_brier_is_the_average_over_the_two_outcomes():
    for f in (0.0, 0.2, 0.5, 0.9):
        for p in (0.1, 0.5, 0.8):
            assert expected_brier(f, p) == pytest.approx(p * brier(f, True) + (1 - p) * brier(f, False))


def test_the_decomposition_expected_equals_uncertainty_plus_excess():
    rng = random.Random(1)
    for _ in range(500):
        f, p = rng.random(), rng.random()
        assert expected_brier(f, p) == pytest.approx(uncertainty(p) + excess_brier(f, p))


def test_brier_is_a_proper_scoring_rule_so_honesty_is_the_best_policy():
    """If the true probability is p, the expected score is lowest when you say exactly p."""
    for p in (0.07, 0.3, 0.5, 0.83):
        grid = [i / 10_000 for i in range(10_001)]
        best = min(grid, key=lambda f: expected_brier(f, p))
        assert best == pytest.approx(p, abs=1e-4)
        assert expected_brier(p, p) == pytest.approx(uncertainty(p))  # the best anyone can expect


def test_nobody_can_beat_the_uncertainty_floor():
    rng = random.Random(2)
    for _ in range(500):
        f, p = rng.random(), rng.random()
        assert expected_brier(f, p) >= uncertainty(p) - 1e-12


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5", None])
def test_nonsense_forecasts_are_rejected(bad):
    for function in (lambda f: brier(f, True), lambda f: expected_brier(f, 0.5), lambda f: excess_brier(f, 0.5)):
        with pytest.raises(ValueError):
            function(bad)


@pytest.mark.parametrize(
    ("forecast", "truth", "expected"),
    [(0.50, 0.50, "Spot on"), (0.51, 0.50, "Spot on"), (0.54, 0.50, "Very close: too high"), (0.40, 0.45, "Very close: too low"),
     (0.58, 0.50, "Close: too high"), (0.65, 0.50, "Off: too high"), (0.10, 0.50, "Far off: too low")],
)
def test_the_verdict_reads_the_gap(forecast, truth, expected):
    assert verdict(forecast, truth) == expected


# -------------------------------------------------------------------- summary


def test_summary_of_a_small_hand_checked_set():
    answers = [ans(0.8, 0.75, True), ans(0.2, 0.25, False), ans(0.5, 0.9, True, "poker")]
    s = summarize(answers)
    assert s["n"] == 3 and s["events_that_happened"] == 2
    assert s["mean_brier"] == pytest.approx((0.04 + 0.04 + 0.25) / 3)
    assert s["mean_excess"] == pytest.approx((0.05**2 + 0.05**2 + 0.4**2) / 3)
    assert s["mean_uncertainty"] == pytest.approx((0.75 * 0.25 + 0.25 * 0.75 + 0.9 * 0.1) / 3)
    assert s["mean_expected_brier"] == pytest.approx(s["mean_uncertainty"] + s["mean_excess"])
    assert s["mean_signed_error"] == pytest.approx((0.05 - 0.05 - 0.4) / 3)
    assert s["skill_vs_always_50_realized"] == pytest.approx(1 - s["mean_brier"] / 0.25)
    assert s["baseline_brier"] == 0.25
    kinds = {k["kind"]: k for k in s["by_kind"]}
    assert kinds["poker"]["n"] == 1 and kinds["poker"]["mean_signed_error"] == pytest.approx(-0.4)
    assert kinds["duel_win"]["n"] == 2 and kinds["duel_win"]["mean_excess"] == pytest.approx(0.0025)


def test_summary_edge_cases():
    assert summarize([]) == {"n": 0}
    one = summarize([ans(0.3, 0.3, False)])
    assert one["brier_standard_error"] is None and one["mean_excess"] == 0
    perfect_knowledge = summarize([ans(0.3, 0.3, False), ans(0.8, 0.8, True)])
    assert perfect_knowledge["mean_excess"] == 0 and perfect_knowledge["mean_signed_error"] == 0


def test_a_standard_error_is_reported_for_the_noisy_realized_score():
    answers = [ans(0.5, 0.5, i % 2 == 0) for i in range(10)]
    assert summarize(answers)["brier_standard_error"] == 0  # 0.5 always scores 0.25: no noise at all
    noisy = [ans(0.9, 0.5, i % 2 == 0) for i in range(10)]
    assert summarize(noisy)["brier_standard_error"] > 0


# ---------------------------------------------------------------- calibration


def test_bins_put_boundaries_in_the_right_place_and_skip_empty_bins():
    answers = [ans(0.0, 0.1, False), ans(0.099, 0.1, False), ans(0.1, 0.2, True), ans(0.95, 0.9, True), ans(1.0, 0.9, True)]
    bins = calibration_bins(answers)
    assert [(b["lower"], b["n"]) for b in bins] == [(0.0, 2), (0.1, 1), (0.9, 2)]
    last = bins[-1]
    assert last["upper"] == 1.0 and last["mean_forecast"] == pytest.approx(0.975)
    assert last["mean_probability"] == pytest.approx(0.9) and last["observed_frequency"] == 1.0


def test_a_perfectly_calibrated_forecaster_has_every_bin_on_the_diagonal():
    rng = random.Random(5)
    answers = []
    for _ in range(400):
        p = rng.random()
        answers.append(ans(p, p, rng.random() < p))
    for b in calibration_bins(answers):
        assert b["mean_forecast"] == pytest.approx(b["mean_probability"])  # forecasting p IS the true probability
        if b["n"] >= 30:
            se = (b["mean_probability"] * (1 - b["mean_probability"]) / b["n"]) ** 0.5
            assert abs(b["observed_frequency"] - b["mean_probability"]) < 4.5 * se


def test_an_overconfident_forecaster_sits_off_the_diagonal_in_the_right_direction():
    rng = random.Random(6)
    answers = []
    for _ in range(600):
        p = rng.uniform(0.2, 0.8)
        f = min(1.0, max(0.0, 0.5 + 2 * (p - 0.5)))  # pushes every guess away from 50%
        answers.append(ans(f, p, rng.random() < p))
    high = [b for b in calibration_bins(answers) if b["mean_forecast"] > 0.7]
    low = [b for b in calibration_bins(answers) if b["mean_forecast"] < 0.3]
    assert all(b["mean_forecast"] > b["mean_probability"] for b in high)  # says 85%, truth is 65%
    assert all(b["mean_forecast"] < b["mean_probability"] for b in low)


# ----------------------------------------------------------------------- series


def test_series_cumulative_and_rolling_averages():
    answers = [ans(0.5, 0.5, True), ans(1.0, 0.5, False), ans(0.0, 0.5, False), ans(0.2, 0.5, True)]
    s = series(answers, window=2)
    assert s["index"] == [1, 2, 3, 4]
    assert s["brier"] == [pytest.approx(0.25), pytest.approx(1.0), pytest.approx(0.0), pytest.approx(0.64)]
    assert s["cumulative_brier"] == [pytest.approx(0.25), pytest.approx(0.625), pytest.approx(1.25 / 3), pytest.approx(1.89 / 4)]
    assert s["rolling_brier"] == [pytest.approx(0.25), pytest.approx(0.625), pytest.approx(0.5), pytest.approx(0.32)]
    assert s["cumulative_uncertainty"] == [pytest.approx(0.25)] * 4
    assert set(map(len, s.values())) == {4}


def test_series_of_nothing_is_empty():
    assert all(v == [] for v in series([]).values())
