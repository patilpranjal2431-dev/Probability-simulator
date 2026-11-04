"""Calibration questions, each with an EXACT answer and a real experiment that settles it.

Every question is described by a small JSON-friendly "spec" ({"kind": ..., params}), so it can be
stored and rebuilt later. The true probability is computed exactly (Fractions) by the Phase 3
functions. When a player answers, `run_experiment` actually performs the random experiment (rolls
the dice, deals the cards, ...) and reports whether the event happened.
"""

import random
from dataclasses import dataclass
from fractions import Fraction

from app.probability.poker import classify_hand
from app.probability.exact import (
    birthday_match_probability,
    coin_run_probability,
    dice_sum_counts,
    hypergeometric_cdf,
    hypergeometric_pmf,
    poker_hand_probabilities,
)
from app.probability.montecarlo import CardDrawer

KINDS = ("dice_exact", "dice_at_least", "poker", "birthday", "hypergeometric", "duel_win", "coin_run")
POKER_CATEGORIES = ("one_pair", "two_pair", "three_of_a_kind", "pair_or_better")
POKER_TEXT = {
    "one_pair": "exactly one pair (and nothing better)",
    "two_pair": "exactly two pair",
    "three_of_a_kind": "exactly three of a kind (not a full house or better)",
    "pair_or_better": "at least a pair (any hand better than high card)",
}
TARGETS = {  # name -> (cards of that kind in the deck, singular label, plural label, predicate on a card number)
    "aces": (4, "ace", "aces", lambda card: card % 13 == 12),
    "hearts": (13, "heart", "hearts", lambda card: card // 13 == 2),
    "face_cards": (12, "face card (J, Q or K)", "face cards (J, Q or K)", lambda card: card % 13 in (9, 10, 11)),
}
RANKS, SUITS = "23456789TJQKA", "♣♦♥♠"
MIN_PROBABILITY, MAX_PROBABILITY = Fraction(1, 50), Fraction(49, 50)


@dataclass(frozen=True)
class Question:
    kind: str
    params: dict
    text: str
    probability: Fraction  # the exact true probability

    @property
    def spec(self) -> dict:
        return {"kind": self.kind, **self.params}


def _int(params: dict, name: str, low: int, high: int) -> int:
    value = params.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be a whole number from {low} to {high}")
    return value


def build(kind: str, params: dict) -> Question:
    """Create a question from a kind and its parameters (validating them)."""
    if kind in ("dice_exact", "dice_at_least"):
        dice = _int(params, "dice", 2, 4)
        total = _int(params, "total", dice, 6 * dice)
        counts = dice_sum_counts(dice)
        ways = counts[total] if kind == "dice_exact" else sum(c for t, c in counts.items() if t >= total)
        word = "exactly" if kind == "dice_exact" else "at least"
        text = f"You roll {dice} fair six-sided dice. What is the probability that the total is {word} {total}?"
        return Question(kind, {"dice": dice, "total": total}, text, Fraction(ways, 6**dice))

    if kind == "poker":
        category = params.get("category")
        if category not in POKER_CATEGORIES:
            raise ValueError("unknown poker category")
        probabilities = poker_hand_probabilities()
        p = 1 - probabilities["high_card"] if category == "pair_or_better" else probabilities[category]
        text = f"You are dealt 5 cards from a shuffled 52-card deck. What is the probability that your hand is {POKER_TEXT[category]}?"
        return Question(kind, {"category": category}, text, p)

    if kind == "birthday":
        people = _int(params, "people", 5, 80)
        text = (f"In a room of {people} people (365 equally likely birthdays, ignoring leap years), "
                "what is the probability that at least two of them share a birthday?")
        return Question(kind, {"people": people}, text, birthday_match_probability(people))

    if kind == "hypergeometric":
        target = params.get("target")
        mode = params.get("mode")
        if target not in TARGETS or mode not in ("exactly", "at_least"):
            raise ValueError("unknown target or mode")
        successes, singular, plural, _ = TARGETS[target]
        draws = _int(params, "draws", 3, 12)
        count = _int(params, "count", 1, min(draws, successes))
        p = (hypergeometric_pmf(52, successes, draws, count) if mode == "exactly"
             else 1 - hypergeometric_cdf(52, successes, draws, count - 1))
        word = "exactly" if mode == "exactly" else "at least"
        noun = singular if count == 1 else plural
        text = f"You are dealt {draws} cards from a shuffled 52-card deck. What is the probability that you get {word} {count} {noun}?"
        return Question(kind, {"target": target, "mode": mode, "draws": draws, "count": count}, text, p)

    if kind == "duel_win":
        roll = _int(params, "roll", 2, 6)
        text = (f"In Dice Duel you roll a {roll}. Your opponent rolls one fair six-sided die. "
                "What is the probability that your roll is strictly higher than theirs?")
        return Question(kind, {"roll": roll}, text, Fraction(roll - 1, 6))

    if kind == "coin_run":
        flips = _int(params, "flips", 4, 14)
        run = _int(params, "run", 2, min(6, flips))
        text = f"You flip a fair coin {flips} times. What is the probability of at least one run of {run} heads in a row?"
        return Question(kind, {"flips": flips, "run": run}, text, coin_run_probability(flips, run))

    raise ValueError(f"unknown question kind {kind!r}")


def question_from_spec(spec: dict) -> Question:
    """Rebuild a stored question."""
    params = {k: v for k, v in spec.items() if k != "kind"}
    return build(spec.get("kind"), params)


def _random_params(kind: str, rng: random.Random) -> dict:
    if kind in ("dice_exact", "dice_at_least"):
        dice = rng.randint(2, 4)
        return {"dice": dice, "total": rng.randint(dice, 6 * dice)}
    if kind == "poker":
        return {"category": rng.choice(POKER_CATEGORIES)}
    if kind == "birthday":
        return {"people": rng.randint(8, 50)}
    if kind == "hypergeometric":
        target = rng.choice(list(TARGETS))
        draws = rng.randint(3, 10)
        return {"target": target, "mode": rng.choice(["exactly", "at_least"]), "draws": draws,
                "count": rng.randint(1, min(draws, 4))}
    if kind == "duel_win":
        return {"roll": rng.randint(2, 6)}
    return {"flips": rng.randint(5, 12), "run": rng.randint(2, 5)}


def generate_question(rng: random.Random) -> Question:
    """A random question whose answer is neither almost-certain nor almost-impossible."""
    for _ in range(200):
        kind = rng.choice(KINDS)
        try:
            question = build(kind, _random_params(kind, rng))
        except ValueError:
            continue
        if MIN_PROBABILITY <= question.probability <= MAX_PROBABILITY:
            return question
    return build("duel_win", {"roll": 4})  # unreachable in practice; keeps the function total


# ------------------------------------------------------------------ experiments


def card_name(card: int) -> str:
    return RANKS[card % 13] + SUITS[card // 13]


def run_experiment(question: Question, rng: random.Random) -> tuple[bool, str]:
    """Actually perform the random experiment. Returns (did the event happen?, what happened)."""
    p = question.params
    if question.kind in ("dice_exact", "dice_at_least"):
        rolls = [rng.randint(1, 6) for _ in range(p["dice"])]
        total = sum(rolls)
        occurred = total == p["total"] if question.kind == "dice_exact" else total >= p["total"]
        return occurred, f"You rolled {' + '.join(map(str, rolls))} = {total}."

    if question.kind == "poker":
        cards = CardDrawer().draw(rng, 5)
        category = classify_hand(cards)
        occurred = category != "high_card" if p["category"] == "pair_or_better" else category == p["category"]
        return occurred, f"You were dealt {' '.join(card_name(c) for c in cards)}: {category.replace('_', ' ')}."

    if question.kind == "birthday":
        days = [rng.randrange(365) for _ in range(p["people"])]
        occurred = len(set(days)) < len(days)
        return occurred, f"Among the {p['people']} people, " + ("at least two shared a birthday." if occurred else "all birthdays were different.")

    if question.kind == "hypergeometric":
        _, singular, plural, predicate = TARGETS[p["target"]]
        cards = CardDrawer().draw(rng, p["draws"])
        found = sum(1 for c in cards if predicate(c))
        occurred = found == p["count"] if p["mode"] == "exactly" else found >= p["count"]
        shown = " ".join(card_name(c) for c in cards)
        return occurred, f"You were dealt {shown}: {found} {singular if found == 1 else plural}."

    if question.kind == "duel_win":
        opponent = rng.randint(1, 6)
        return p["roll"] > opponent, f"You rolled {p['roll']}; your opponent rolled {opponent}."

    flips = [rng.random() < 0.5 for _ in range(p["flips"])]  # coin_run: True = heads
    longest = current = 0
    for heads in flips:
        current = current + 1 if heads else 0
        longest = max(longest, current)
    shown = "".join("H" if heads else "T" for heads in flips)
    return longest >= p["run"], f"Flips: {shown}. The longest run of heads was {longest}."
