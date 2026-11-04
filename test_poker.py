"""Hand classification, and a brute-force check of the exact counting formulas."""

import itertools
import random
from collections import Counter

import pytest

from app.probability.exact import poker_hand_counts
from app.probability.poker import classify_hand

RANKS = "23456789TJQKA"
SUITS = "cdhs"


def hand(text: str) -> list[int]:
    """'Ts Js Qs Ks As' -> card numbers (rank + 13 * suit)."""
    return [RANKS.index(card[0]) + 13 * SUITS.index(card[1]) for card in text.split()]


@pytest.mark.parametrize(
    ("cards", "expected"),
    [
        ("Ts Js Qs Ks As", "royal_flush"),
        ("5h 6h 7h 8h 9h", "straight_flush"),
        ("As 2s 3s 4s 5s", "straight_flush"),  # the "steel wheel" is not a royal flush
        ("9c 9d 9h 9s 2c", "four_of_a_kind"),
        ("3c 3d 3h 8c 8d", "full_house"),
        ("2h 5h 9h Jh Kh", "flush"),
        ("5c 6d 7h 8s 9c", "straight"),
        ("Ac 2d 3h 4s 5c", "straight"),  # ace plays low
        ("Tc Jd Qh Ks Ac", "straight"),  # ace plays high
        ("Qc Kd Ac 2d 3h", "high_card"),  # a straight cannot wrap around the ace
        ("7c 7d 7h Kc 2d", "three_of_a_kind"),
        ("4c 4d 9h 9s Ac", "two_pair"),
        ("Jc Jd 2h 5s 9c", "one_pair"),
        ("2c 5d 9h Js Kc", "high_card"),
    ],
)
def test_known_hands(cards, expected):
    assert classify_hand(hand(cards)) == expected


def test_order_of_cards_does_not_matter():
    rng = random.Random(7)
    for text, expected in [("Ts Js Qs Ks As", "royal_flush"), ("3c 3d 3h 8c 8d", "full_house"),
                           ("4c 4d 9h 9s Ac", "two_pair")]:
        cards = hand(text)
        for _ in range(20):
            rng.shuffle(cards)
            assert classify_hand(cards) == expected


@pytest.mark.parametrize(
    "bad",
    [[0, 1, 2, 3], [0, 1, 2, 3, 4, 5], [0, 0, 1, 2, 3], [0, 1, 2, 3, 52], [-1, 1, 2, 3, 4]],
)
def test_invalid_hands_are_rejected(bad):
    with pytest.raises(ValueError):
        classify_hand(bad)


@pytest.mark.slow
def test_formulas_match_brute_force_over_all_2_598_960_hands():
    """The independent check: enumerate EVERY hand, classify it, compare with the formulas."""
    tally = Counter(classify_hand(h) for h in itertools.combinations(range(52), 5))
    assert dict(tally) == poker_hand_counts()
