"""5-card poker hand classification.

A card is an integer 0..51:  rank = card % 13  (0 = two ... 12 = ace),  suit = card // 13.
"""

from collections import Counter
from collections.abc import Sequence

DECK_SIZE = 52
_WHEEL = frozenset({12, 0, 1, 2, 3})  # A-2-3-4-5: the ace counts low
_ROYAL_LOW_RANK = 8  # ten; ranks 8..12 are 10-J-Q-K-A


def classify_hand(cards: Sequence[int]) -> str:
    """Return the single best category of a 5-card hand.

    Categories are exclusive and ranked: royal_flush, straight_flush, four_of_a_kind,
    full_house, flush, straight, three_of_a_kind, two_pair, one_pair, high_card.

    Raises ValueError unless `cards` is 5 distinct integers in 0..51.
    """
    if len(cards) != 5 or len(set(cards)) != 5 or min(cards) < 0 or max(cards) > DECK_SIZE - 1:
        raise ValueError("a hand must be 5 distinct cards numbered 0..51")

    ranks = [c % 13 for c in cards]
    is_flush = len({c // 13 for c in cards}) == 1
    rank_set = set(ranks)
    is_straight = len(rank_set) == 5 and (max(ranks) - min(ranks) == 4 or rank_set == _WHEEL)

    if is_straight and is_flush:
        return "royal_flush" if min(ranks) == _ROYAL_LOW_RANK else "straight_flush"

    shape = sorted(Counter(ranks).values(), reverse=True)  # e.g. [3, 2] for a full house
    if shape[0] == 4:
        return "four_of_a_kind"
    if shape == [3, 2]:
        return "full_house"
    if is_flush:
        return "flush"
    if is_straight:
        return "straight"
    if shape[0] == 3:
        return "three_of_a_kind"
    if shape == [2, 2, 1]:
        return "two_pair"
    if shape[0] == 2:
        return "one_pair"
    return "high_card"
