"""Exact probabilities, checked against well-known values."""

from fractions import Fraction
from math import comb

import pytest

from app.probability.exact import (
    TOTAL_POKER_HANDS,
    birthday_match_probability,
    birthday_no_match_probability,
    birthday_smallest_group,
    dice_sum_counts,
    dice_sum_probabilities,
    dice_sum_probability,
    hypergeometric_cdf,
    hypergeometric_pmf,
    poker_hand_counts,
    poker_hand_probabilities,
)

# ------------------------------------------------------------------------ dice


def test_two_dice_counts_are_the_classic_triangle():
    assert dice_sum_counts(2) == {
        2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1,
    }


def test_two_dice_known_probabilities():
    assert dice_sum_probability(2, 7) == Fraction(1, 6)
    assert dice_sum_probability(2, 2) == Fraction(1, 36)
    assert dice_sum_probability(2, 12) == Fraction(1, 36)
    assert dice_sum_probability(2, 13) == 0  # impossible total
    assert dice_sum_probability(2, 1) == 0


def test_three_dice_known_probabilities():
    assert dice_sum_probability(3, 3) == Fraction(1, 216)
    assert dice_sum_probability(3, 18) == Fraction(1, 216)
    assert dice_sum_probability(3, 10) == Fraction(1, 8)  # 27 / 216
    assert dice_sum_probability(3, 11) == Fraction(1, 8)


@pytest.mark.parametrize("num_dice", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("sides", [2, 4, 6, 20])
def test_counts_add_up_to_every_possible_outcome(num_dice, sides):
    assert sum(dice_sum_counts(num_dice, sides).values()) == sides**num_dice


@pytest.mark.parametrize("num_dice", [2, 3, 6])
def test_distribution_is_symmetric(num_dice):
    counts = dice_sum_counts(num_dice)
    for total, ways in counts.items():
        assert counts[7 * num_dice - total] == ways


@pytest.mark.parametrize(("num_dice", "sides"), [(1, 6), (2, 6), (4, 6), (3, 10)])
def test_mean_and_variance_match_theory_exactly(num_dice, sides):
    probs = dice_sum_probabilities(num_dice, sides)
    mean = sum(total * p for total, p in probs.items())
    variance = sum((total - mean) ** 2 * p for total, p in probs.items())
    assert mean == Fraction(num_dice * (sides + 1), 2)
    assert variance == Fraction(num_dice * (sides**2 - 1), 12)


def test_one_die_is_uniform():
    assert set(dice_sum_probabilities(1).values()) == {Fraction(1, 6)}


def test_dice_rejects_bad_arguments():
    with pytest.raises(ValueError):
        dice_sum_counts(0)
    with pytest.raises(ValueError):
        dice_sum_counts(2, sides=0)
    with pytest.raises(TypeError):
        dice_sum_counts(2.5)
    with pytest.raises(TypeError):
        dice_sum_counts(True)


# ----------------------------------------------------------------------- poker


def test_poker_counts_match_the_textbook_values():
    assert poker_hand_counts() == {
        "royal_flush": 4,
        "straight_flush": 36,
        "four_of_a_kind": 624,
        "full_house": 3_744,
        "flush": 5_108,
        "straight": 10_200,
        "three_of_a_kind": 54_912,
        "two_pair": 123_552,
        "one_pair": 1_098_240,
        "high_card": 1_302_540,
    }


def test_poker_counts_cover_every_hand_exactly_once():
    assert TOTAL_POKER_HANDS == 2_598_960
    assert sum(poker_hand_counts().values()) == 2_598_960
    assert sum(poker_hand_probabilities().values()) == 1


def test_poker_probabilities_match_known_decimals():
    probs = {k: float(v) for k, v in poker_hand_probabilities().items()}
    assert probs["royal_flush"] == pytest.approx(1 / 649_740)
    assert probs["four_of_a_kind"] == pytest.approx(1 / 4_165)
    assert probs["full_house"] == pytest.approx(0.0014406, rel=1e-3)
    assert probs["flush"] == pytest.approx(0.0019654, rel=1e-3)
    assert probs["straight"] == pytest.approx(0.0039246, rel=1e-3)
    assert probs["two_pair"] == pytest.approx(0.0475390, rel=1e-3)
    assert probs["one_pair"] == pytest.approx(0.422569, rel=1e-3)
    assert probs["high_card"] == pytest.approx(0.501177, rel=1e-3)


# --------------------------------------------------------------- hypergeometric


def test_exactly_two_aces_in_five_cards():
    # C(4,2) * C(48,3) / C(52,5) = 103,776 / 2,598,960 = 0.03993
    assert hypergeometric_pmf(52, 4, 5, 2) == Fraction(103_776, 2_598_960)
    assert float(hypergeometric_pmf(52, 4, 5, 2)) == pytest.approx(0.0399298, rel=1e-5)


def test_at_least_one_ace_in_five_cards():
    expected = 1 - Fraction(comb(48, 5), comb(52, 5))
    assert 1 - hypergeometric_cdf(52, 4, 5, 0) == expected
    assert float(expected) == pytest.approx(0.34114, rel=1e-3)


@pytest.mark.parametrize(("population", "successes", "draws"), [(52, 4, 5), (52, 13, 7), (20, 8, 6), (10, 10, 4)])
def test_pmf_sums_to_one_and_has_the_right_mean(population, successes, draws):
    pmf = [hypergeometric_pmf(population, successes, draws, k) for k in range(draws + 1)]
    assert sum(pmf) == 1
    assert sum(k * p for k, p in enumerate(pmf)) == Fraction(draws * successes, population)


def test_impossible_counts_have_zero_probability():
    assert hypergeometric_pmf(52, 4, 5, 5) == 0  # only 4 aces exist
    assert hypergeometric_pmf(52, 4, 5, -1) == 0
    assert hypergeometric_pmf(10, 8, 5, 0) == 0  # only 2 non-successes, must draw >= 3


def test_drawing_nothing_always_gives_zero_successes():
    assert hypergeometric_pmf(52, 4, 0, 0) == 1


def test_hypergeometric_agrees_with_poker_flush_count():
    """Cross-check two independent modules: all-five-same-suit hands."""
    one_suit = hypergeometric_pmf(52, 13, 5, 5)  # five cards, all from a given suit
    assert one_suit == Fraction(comb(13, 5), comb(52, 5))
    flush_like = 4 * comb(13, 5)  # includes straight flushes and royal flushes
    counts = poker_hand_counts()
    assert flush_like == counts["flush"] + counts["straight_flush"] + counts["royal_flush"]
    assert 4 * one_suit == Fraction(flush_like, TOTAL_POKER_HANDS)


def test_hypergeometric_rejects_bad_arguments():
    with pytest.raises(ValueError):
        hypergeometric_pmf(0, 0, 0, 0)
    with pytest.raises(ValueError):
        hypergeometric_pmf(10, 11, 5, 1)
    with pytest.raises(ValueError):
        hypergeometric_pmf(10, 5, 11, 1)
    with pytest.raises(TypeError):
        hypergeometric_pmf(10.0, 5, 5, 1)


# -------------------------------------------------------------------- birthday


def test_birthday_paradox_23_people():
    assert float(birthday_match_probability(23)) == pytest.approx(0.5072972343, abs=1e-9)
    assert float(birthday_match_probability(22)) < 0.5
    assert birthday_smallest_group(0.5) == 23


def test_birthday_other_known_values():
    assert float(birthday_match_probability(57)) == pytest.approx(0.990122, abs=1e-6)
    assert birthday_smallest_group(0.99) == 57
    assert float(birthday_match_probability(70)) > 0.999


def test_birthday_edge_cases():
    assert birthday_match_probability(0) == 0
    assert birthday_match_probability(1) == 0
    assert birthday_match_probability(2, days=2) == Fraction(1, 2)
    assert birthday_match_probability(366) == 1  # pigeonhole principle
    assert birthday_match_probability(2, days=1) == 1


def test_birthday_probabilities_add_up_and_increase():
    for people in (5, 23, 40):
        assert birthday_no_match_probability(people) + birthday_match_probability(people) == 1
    values = [birthday_match_probability(n) for n in range(0, 80)]
    assert values == sorted(values)


def test_birthday_rejects_bad_arguments():
    with pytest.raises(ValueError):
        birthday_match_probability(-1)
    with pytest.raises(ValueError):
        birthday_match_probability(5, days=0)
    with pytest.raises(ValueError):
        birthday_smallest_group(0)
    with pytest.raises(ValueError):
        birthday_smallest_group(1.5)
    with pytest.raises(TypeError):
        birthday_match_probability(2.5)
