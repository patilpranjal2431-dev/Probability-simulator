"""The calibration game end to end, with simulated players of different quality."""

import itertools
import random
import xml.etree.ElementTree as ET

import pytest

from app.calibration.questions import question_from_spec
from app.calibration.scoring import brier, expected_brier, uncertainty
from app.calibration.service import CalibrationService
from app.calibration.store import AlreadyAnswered, MemoryCalibrationStore, NoSuchQuestion


def make_service(seed_start=1, store=None):
    seeds = itertools.count(seed_start)
    return CalibrationService(store or MemoryCalibrationStore(), lambda: random.Random(next(seeds)))


def true_probability(service, user, question_id):
    """Test-only peek at the answer, to build forecasters who know it."""
    spec = service.store.get(user, question_id)["spec"]
    return float(question_from_spec(spec).probability)


def clip(x):
    return min(1.0, max(0.0, x))


POLICIES = {
    "oracle": lambda p: p,
    "fifty": lambda p: 0.5,
    "overconfident": lambda p: clip(0.5 + 2.5 * (p - 0.5)),
    "too_high": lambda p: clip(p + 0.2),
    "too_low": lambda p: clip(p - 0.2),
}


def play(service, user, policy, n):
    results = []
    for _ in range(n):
        q = service.next_question(user)
        results.append(service.answer(user, q["question_id"], POLICIES[policy](true_probability(service, user, q["question_id"]))))
    return results


# ===================================================================== flow


def test_you_keep_the_same_question_until_you_answer_it():
    service = make_service()
    first = service.next_question(1)
    assert service.next_question(1)["question_id"] == first["question_id"]  # no rerolling
    assert first["answered_so_far"] == 0 and first["text"].endswith("?")
    service.answer(1, first["question_id"], 0.5)
    second = service.next_question(1)
    assert second["question_id"] != first["question_id"] and second["answered_so_far"] == 1


def test_the_question_does_not_reveal_the_answer():
    service = make_service()
    q = service.next_question(1)
    assert set(q) == {"question_id", "kind", "text", "answered_so_far"}  # no probability, no outcome


def test_each_user_gets_their_own_questions():
    service = make_service()
    a, b = service.next_question(1), service.next_question(2)
    assert a["question_id"] != b["question_id"]
    service.answer(1, a["question_id"], 0.5)
    assert service.next_question(2)["question_id"] == b["question_id"]  # unaffected by user 1


def test_the_result_contains_every_number_and_they_agree_with_the_formulas():
    service = make_service()
    q = service.next_question(1)
    p = true_probability(service, 1, q["question_id"])
    r = service.answer(1, q["question_id"], 0.8)
    assert r["true_probability"] == pytest.approx(p) and r["forecast"] == 0.8
    assert r["brier"] == pytest.approx(brier(0.8, r["event_happened"]))
    assert r["expected_brier"] == pytest.approx(expected_brier(0.8, p))
    assert r["excess_brier"] == pytest.approx((0.8 - p) ** 2) and r["uncertainty"] == pytest.approx(uncertainty(p))
    assert r["expected_brier"] == pytest.approx(r["uncertainty"] + r["excess_brier"])
    assert r["what_happened"] and r["verdict"]


def test_short_exact_fractions_are_shown_and_huge_ones_are_not():
    service = make_service()
    shown = set()
    for _ in range(80):
        q = service.next_question(1)
        r = service.answer(1, q["question_id"], 0.5)
        fraction = r["true_probability_fraction"]
        shown.add(q["kind"] if fraction is None else "short")
        if fraction is not None:
            assert len(fraction) < 15 and "/" in fraction or fraction in ("0", "1")
    assert "birthday" in shown and "short" in shown  # birthday answers have enormous denominators


# ======================================================= refusals and safety


def test_a_question_can_only_be_answered_once():
    service = make_service()
    q = service.next_question(1)
    service.answer(1, q["question_id"], 0.3)
    with pytest.raises(AlreadyAnswered):
        service.answer(1, q["question_id"], 0.9)  # trying again with a better guess
    assert len(service.store.answers(1)) == 1


def test_you_cannot_answer_somebody_elses_question_or_an_invented_one():
    service = make_service()
    q = service.next_question(1)
    with pytest.raises(NoSuchQuestion):
        service.answer(2, q["question_id"], 0.5)
    with pytest.raises(NoSuchQuestion):
        service.answer(1, "made-up-id", 0.5)
    assert service.store.answers(1) == [] and service.store.pending(1)["id"] == q["question_id"]


@pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan"), float("inf"), float("-inf"), True, "0.5", None, [0.5]])
def test_a_bad_forecast_is_refused_and_does_not_use_up_the_question(bad):
    service = make_service()
    q = service.next_question(1)
    with pytest.raises(ValueError):
        service.answer(1, q["question_id"], bad)
    assert service.store.answers(1) == []  # nothing recorded
    assert service.answer(1, q["question_id"], 0.5)["forecast"] == 0.5  # the player can simply try again


def test_the_outcome_cannot_depend_on_the_forecast():
    """Same randomness, different forecasts: the SAME thing happens. (The experiment runs after the
    forecast is locked in and never looks at it.)"""
    outcomes = []
    for forecast in (0.0, 0.35, 0.9, 1.0):
        service = make_service(seed_start=77)
        q = service.next_question(1)
        r = service.answer(1, q["question_id"], forecast)
        outcomes.append((q["text"], r["event_happened"], r["what_happened"]))
    assert len(set(outcomes)) == 1


def test_the_extreme_forecasts_zero_and_one_are_allowed_and_scored():
    service = make_service()
    for forecast in (0.0, 1.0):
        q = service.next_question(1)
        r = service.answer(1, q["question_id"], forecast)
        assert r["brier"] in (0.0, 1.0)


def test_the_default_randomness_is_secure_and_the_game_still_works():
    service = CalibrationService(MemoryCalibrationStore())
    q = service.next_question(1)
    assert service.answer(1, q["question_id"], 0.5)["brier"] == 0.25


# ============================================ skilled and unskilled forecasters


def test_a_player_who_always_says_fifty_percent_scores_exactly_the_baseline():
    service = make_service()
    results = play(service, 1, "fifty", 100)
    assert all(r["brier"] == 0.25 for r in results)
    summary = service.summary(1)["summary"]
    assert summary["mean_brier"] == 0.25 and summary["skill_vs_always_50_realized"] == 0


def test_a_player_who_knows_the_true_probabilities_has_no_excess_and_beats_the_baseline():
    service = make_service()
    play(service, 1, "oracle", 400)
    s = service.summary(1)["summary"]
    assert s["mean_excess"] < 1e-12
    assert s["mean_expected_brier"] == pytest.approx(s["mean_uncertainty"])  # the best anyone can do
    assert s["mean_uncertainty"] < 0.25 and s["skill_vs_always_50_expected"] > 0.2
    # the realized score is noisy but must land near its expectation
    assert abs(s["mean_brier"] - s["mean_uncertainty"]) < 4 * s["brier_standard_error"]


def test_the_ranking_of_forecasters_by_their_noise_free_score():
    expected = {}
    for name in POLICIES:
        service = make_service(seed_start=500)
        play(service, 1, name, 300)
        s = service.summary(1)["summary"]
        expected[name] = (s["mean_expected_brier"], s["mean_excess"], s["mean_signed_error"])
    assert expected["oracle"][0] < expected["fifty"][0] < 0.25 + 1e-9
    assert expected["oracle"][0] < expected["too_high"][0] and expected["oracle"][0] < expected["too_low"][0]
    # The policy exaggerates every distance from 50% by 2.5x. Unclipped that would cost 0.23, but forecasts are
    # clipped to 0..1, so any true probability beyond 30%/70% becomes a certainty (about two thirds of the
    # questions), which is only mildly wrong. Over this question mix the population value is 0.0233.
    assert expected["overconfident"][1] == pytest.approx(0.0233, abs=0.009) and expected["oracle"][1] < 1e-12
    assert expected["too_high"][2] > 0.1 and expected["too_low"][2] < -0.1  # bias is detected, with the right sign


def test_end_to_end_the_events_happen_as_often_as_the_questions_claim():
    """Through the whole service: the average true probability must equal how often events happened."""
    service = make_service(seed_start=900)
    results = play(service, 1, "oracle", 2500)
    mean_p = sum(r["true_probability"] for r in results) / len(results)
    happened = sum(r["event_happened"] for r in results) / len(results)
    se = (sum(r["uncertainty"] for r in results)) ** 0.5 / len(results)  # sd of the mean of Bernoulli(p_i)
    assert abs(happened - mean_p) < 4.5 * se


def test_a_calibrated_player_sits_on_the_diagonal_of_the_reliability_chart():
    service = make_service(seed_start=40)
    play(service, 1, "oracle", 600)
    bins = service.summary(1)["bins"]
    assert len(bins) >= 6
    for b in bins:
        assert b["mean_forecast"] == pytest.approx(b["mean_probability"])
        if b["n"] >= 40:
            se = (b["mean_probability"] * (1 - b["mean_probability"]) / b["n"]) ** 0.5
            assert abs(b["observed_frequency"] - b["mean_probability"]) < 4.5 * se


def test_an_overconfident_player_is_below_the_diagonal_at_the_top_and_above_at_the_bottom():
    service = make_service(seed_start=60)
    play(service, 1, "overconfident", 600)
    bins = service.summary(1)["bins"]
    assert any(b["mean_forecast"] > b["mean_probability"] + 0.05 for b in bins if b["mean_forecast"] > 0.7)
    assert any(b["mean_forecast"] < b["mean_probability"] - 0.05 for b in bins if b["mean_forecast"] < 0.3)


def test_the_summary_shows_which_kinds_of_question_a_player_is_biased_on():
    service = make_service(seed_start=3)
    for _ in range(200):
        q = service.next_question(1)
        p = true_probability(service, 1, q["question_id"])
        f = clip(p + 0.3) if q["kind"] == "poker" else p  # biased ONLY on poker questions
        service.answer(1, q["question_id"], f)
    kinds = {k["kind"]: k for k in service.summary(1)["summary"]["by_kind"]}
    assert kinds["poker"]["mean_signed_error"] > 0.1 and kinds["poker"]["mean_excess"] > 0.02
    assert all(k["mean_excess"] < 1e-12 for name, k in kinds.items() if name != "poker")


# ================================================================= summary


def test_the_summary_before_any_answer_is_empty_but_well_formed():
    assert make_service().summary(1) == {"n": 0, "summary": {"n": 0}, "bins": [], "series": None, "charts": None}


def test_the_summary_after_a_few_answers_is_complete_and_the_charts_are_valid_svg():
    service = make_service()
    play(service, 1, "overconfident", 25)
    data = service.summary(1)
    assert data["n"] == 25 and data["summary"]["n"] == 25
    assert len(data["series"]["index"]) == 25 and len(data["series"]["cumulative_brier"]) == 25
    assert [b["lower"] for b in data["bins"]] == sorted(b["lower"] for b in data["bins"])
    for svg in data["charts"].values():
        assert ET.fromstring(svg).tag == "{http://www.w3.org/2000/svg}svg"


def test_summaries_are_per_user():
    service = make_service()
    play(service, 1, "oracle", 12)
    play(service, 2, "fifty", 5)
    assert service.summary(1)["n"] == 12 and service.summary(2)["n"] == 5 and service.summary(3)["n"] == 0
