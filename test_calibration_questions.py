"""Every question's exact answer, checked by brute force; and every experiment, checked against it."""

import random
from fractions import Fraction
from itertools import product
from math import comb, sqrt

import pytest

from app.calibration.questions import (
    KINDS, MAX_PROBABILITY, MIN_PROBABILITY, POKER_CATEGORIES, build, generate_question, question_from_spec,
    run_experiment,
)
from app.probability.exact import coin_run_probability, poker_hand_counts


# ------------------------------------------------- exact answers by brute force


@pytest.mark.parametrize("dice", [2, 3, 4])
def test_dice_questions_equal_a_count_of_every_possible_roll(dice):
    outcomes = list(product(range(1, 7), repeat=dice))
    for total in range(dice, 6 * dice + 1):
        exact = build("dice_exact", {"dice": dice, "total": total}).probability
        at_least = build("dice_at_least", {"dice": dice, "total": total}).probability
        assert exact == Fraction(sum(1 for r in outcomes if sum(r) == total), len(outcomes))
        assert at_least == Fraction(sum(1 for r in outcomes if sum(r) >= total), len(outcomes))


@pytest.mark.parametrize("flips", range(1, 13))
def test_coin_run_probability_equals_a_count_of_every_sequence(flips):
    sequences = list(product([0, 1], repeat=flips))

    def longest(seq):
        best = current = 0
        for heads in seq:
            current = current + 1 if heads else 0
            best = max(best, current)
        return best

    for run in range(1, flips + 2):
        expected = Fraction(sum(1 for s in sequences if longest(s) >= run), len(sequences))
        assert coin_run_probability(flips, run) == expected, (flips, run)


def test_coin_run_known_values_and_validation():
    assert coin_run_probability(2, 2) == Fraction(1, 4)
    assert coin_run_probability(3, 2) == Fraction(3, 8)  # HHH, HHT, THH
    assert coin_run_probability(5, 1) == 1 - Fraction(1, 32)  # at least one head
    assert coin_run_probability(3, 4) == 0
    for args in ((0, 1), (5, 0)):
        with pytest.raises(ValueError):
            coin_run_probability(*args)
    with pytest.raises(TypeError):
        coin_run_probability(5.0, 2)


def test_poker_questions_match_the_exact_hand_counts():
    counts, total = poker_hand_counts(), 2_598_960
    for category in ("one_pair", "two_pair", "three_of_a_kind"):
        assert build("poker", {"category": category}).probability == Fraction(counts[category], total)
    assert build("poker", {"category": "pair_or_better"}).probability == Fraction(total - counts["high_card"], total)
    assert build("poker", {"category": "one_pair"}).probability == Fraction(352, 833)


def test_hypergeometric_questions_match_an_independent_formula():
    def pmf(population, successes, draws, k):
        return Fraction(comb(successes, k) * comb(population - successes, draws - k), comb(population, draws))

    for target, successes in (("aces", 4), ("hearts", 13), ("face_cards", 12)):
        for draws in (3, 5, 8):
            for count in range(1, min(draws, successes, 4) + 1):
                exactly = build("hypergeometric", {"target": target, "mode": "exactly", "draws": draws, "count": count})
                at_least = build("hypergeometric", {"target": target, "mode": "at_least", "draws": draws, "count": count})
                assert exactly.probability == pmf(52, successes, draws, count)
                assert at_least.probability == sum(pmf(52, successes, draws, k) for k in range(count, min(draws, successes) + 1))
    # a well-known value: exactly one ace in a 5-card hand is 778,320 / 2,598,960
    assert build("hypergeometric", {"target": "aces", "mode": "exactly", "draws": 5, "count": 1}).probability == Fraction(778_320, 2_598_960)


def test_birthday_and_duel_questions_match_independent_formulas():
    for people in (5, 23, 40):
        none_shared = Fraction(1)
        for i in range(people):
            none_shared *= Fraction(365 - i, 365)
        assert build("birthday", {"people": people}).probability == 1 - none_shared
    assert float(build("birthday", {"people": 23}).probability) == pytest.approx(0.5073, abs=1e-4)
    for roll in range(2, 7):
        wins = sum(1 for opponent in range(1, 7) if roll > opponent)
        assert build("duel_win", {"roll": roll}).probability == Fraction(wins, 6)


# --------------------------------------------------------- specs and validation


def test_every_question_survives_being_stored_and_rebuilt():
    rng = random.Random(1)
    for _ in range(300):
        q = generate_question(rng)
        assert question_from_spec(q.spec) == q
        assert q.spec["kind"] == q.kind and set(q.params) == set(q.spec) - {"kind"}


@pytest.mark.parametrize(
    ("kind", "params"),
    [
        ("dice_exact", {"dice": 1, "total": 3}), ("dice_exact", {"dice": 2, "total": 13}), ("dice_exact", {"dice": 2, "total": 1}),
        ("dice_at_least", {"dice": "2", "total": 5}), ("dice_exact", {"dice": True, "total": 2}),
        ("poker", {"category": "royal_flush"}), ("poker", {}),
        ("birthday", {"people": 4}), ("birthday", {"people": 500}),
        ("hypergeometric", {"target": "kings", "mode": "exactly", "draws": 5, "count": 1}),
        ("hypergeometric", {"target": "aces", "mode": "sometimes", "draws": 5, "count": 1}),
        ("hypergeometric", {"target": "aces", "mode": "exactly", "draws": 5, "count": 5}),
        ("duel_win", {"roll": 1}), ("duel_win", {"roll": 7}),
        ("coin_run", {"flips": 3, "run": 2}), ("coin_run", {"flips": 10, "run": 7}),
        ("lottery", {}),
    ],
)
def test_nonsense_questions_are_rejected(kind, params):
    with pytest.raises(ValueError):
        build(kind, params)


def test_a_spec_without_a_kind_is_rejected():
    with pytest.raises(ValueError):
        question_from_spec({"people": 10})


# ---------------------------------------------------------------- the generator


def test_generated_questions_are_never_trivial_and_cover_every_kind():
    rng = random.Random(7)
    questions = [generate_question(rng) for _ in range(600)]
    assert {q.kind for q in questions} == set(KINDS)
    assert all(MIN_PROBABILITY <= q.probability <= MAX_PROBABILITY for q in questions)
    assert all(0 < q.probability < 1 for q in questions)
    assert len({q.text for q in questions}) > 100  # plenty of variety
    probabilities = [float(q.probability) for q in questions]
    assert min(probabilities) < 0.1 and max(probabilities) > 0.9  # spans the whole range


def test_generation_is_reproducible_from_a_seed():
    assert [generate_question(random.Random(3)).spec for _ in range(1)] == [generate_question(random.Random(3)).spec]
    a, b = random.Random(9), random.Random(9)
    assert [generate_question(a).spec for _ in range(20)] == [generate_question(b).spec for _ in range(20)]


def test_poker_questions_only_use_categories_that_are_common_enough_to_be_interesting():
    assert set(POKER_CATEGORIES) == {"one_pair", "two_pair", "three_of_a_kind", "pair_or_better"}
    for category in POKER_CATEGORIES:
        assert build("poker", {"category": category}).probability > Fraction(1, 50)


# ------------------------------------------------- experiments vs exact answers


def test_every_experiment_happens_exactly_as_often_as_the_exact_probability_says():
    """Run each kind of experiment thousands of times (seeded) and compare with the exact answer."""
    rng = random.Random(2026)
    trials = 4000
    seen = set()
    checked = 0
    while len(seen) < len(KINDS) or checked < 70:
        q = generate_question(rng)
        seen.add(q.kind)
        hits = sum(run_experiment(q, rng)[0] for _ in range(trials))
        p = float(q.probability)
        standard_error = sqrt(p * (1 - p) / trials)
        assert abs(hits / trials - p) < 4.5 * standard_error, (q.text, hits / trials, p)
        checked += 1


def test_experiments_describe_what_happened_and_repeat_exactly_from_a_seed():
    for kind, params, token in (
        ("dice_exact", {"dice": 3, "total": 10}, "="), ("dice_at_least", {"dice": 2, "total": 7}, "="),
        ("poker", {"category": "one_pair"}, "dealt"), ("birthday", {"people": 23}, "birthday"),
        ("hypergeometric", {"target": "hearts", "mode": "at_least", "draws": 6, "count": 2}, "dealt"),
        ("duel_win", {"roll": 4}, "opponent rolled"), ("coin_run", {"flips": 8, "run": 3}, "longest run"),
    ):
        q = build(kind, params)
        first = run_experiment(q, random.Random(11))
        assert first == run_experiment(q, random.Random(11))
        assert token in first[1] and isinstance(first[0], bool)
        assert len({run_experiment(q, random.Random(s))[1] for s in range(30)}) > 1  # it really is random


def test_the_dice_narration_adds_up_and_the_verdict_matches_the_event():
    q = build("dice_exact", {"dice": 2, "total": 7})
    for seed in range(40):
        happened, what = run_experiment(q, random.Random(seed))
        total = int(what.rstrip(".").split("=")[1])
        parts = [int(x) for x in what.split("=")[0].replace("You rolled", "").split("+")]
        assert sum(parts) == total and happened == (total == 7)


def test_poker_narration_names_the_category_that_decides_the_event():
    q = build("poker", {"category": "pair_or_better"})
    for seed in range(60):
        happened, what = run_experiment(q, random.Random(seed))
        assert happened == (not what.rstrip(".").endswith("high card"))
