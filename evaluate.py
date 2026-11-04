"""Exact expected value of every candidate action, by enumeration over the real game engine.

The question answered, for ONE decision a player faced:

    "If I take action A here, how many chips do I expect to win or lose in this hand?"

Assumptions (and nothing else):
  * Your own hidden value is fixed (you know it).
  * The opponent's value is uniformly random among the possibilities, then WEIGHTED by how likely
    the reference opponent was to have acted the way this opponent actually did so far.
    (A reference that never bets with a weak hand makes a bet strong evidence of a strong hand.)
    If no possible value could have produced that behaviour, the weighting is dropped.
  * From here on the opponent plays the reference strategy, and YOU play the best you can.
    "Best" is chosen at your information set: one action for all the opponent hands you cannot tell apart.

The numbers are net chips for the hand, including the ante already paid.
"""

import copy
import random
from collections.abc import Sequence

from app.games.base import CALL, CHECK, FOLD, Action, Game, Hand, bet

from .types import DecisionContext

_RNG = random.Random(0)  # never used for chance: deals are always given explicitly
MAX_ENUMERATED_ACTIONS = 2000


def legal_actions(hand: Hand) -> list[Action]:
    """Every legal action for the player to act."""
    d = hand.decision()
    if d.kind == "respond":
        return [CALL, FOLD]
    actions = [CHECK]
    if d.can_bet():
        if d.max_bet - d.min_bet + 1 > MAX_ENUMERATED_ACTIONS:
            raise ValueError("too many bet sizes to enumerate; pass candidate sizes instead")
        actions += [bet(size) for size in range(d.min_bet, d.max_bet + 1)]
    return actions


def _opponent_deals(game: Game, seat: int, own_value: int) -> list[tuple[int, int]]:
    """Every equally likely deal in which `seat` holds `own_value`."""
    return [deal for deal in game.all_deals() if deal[seat] == own_value]


def _replay(game: Game, ctx: DecisionContext, deal: tuple[int, int], reference) -> tuple[Hand, float]:
    """Re-create the hand up to the graded decision for one possible opponent value.

    Returns the hand and the likelihood of the opponent's observed actions under the reference."""
    hand = game.new_hand(_RNG, ctx.stacks, ctx.first, deal=deal)
    opponent = 1 - ctx.seat
    likelihood = 1.0
    for seat, kind, amount in ctx.history:
        action = Action(kind, amount)
        if seat == opponent:
            probs = reference.action_probs(hand)
            likelihood *= sum(p for a, p in probs.items() if a.kind == kind)
        hand.act(action)
    return hand, likelihood


def _evaluate(states: list[tuple[Hand, float]], me: int, reference) -> float:
    """Expected net chips for `me`, summed over `states` (hands at the same point in play that
    differ only in the opponent's hidden value, each carrying its probability weight)."""
    total = 0.0
    live: list[tuple[Hand, float]] = []
    for hand, weight in states:
        if hand.done:
            total += weight * hand.result.net[me]
        else:
            live.append((hand, weight))
    if not live:
        return total

    actor = live[0][0].to_act
    if actor == me:
        # One action for ALL opponent hands: you cannot see which one you face.
        best = None
        for action in legal_actions(live[0][0]):
            value = _evaluate(_advance(live, lambda hand, a=action: {a: 1.0}), me, reference)
            best = value if best is None or value > best else best
        return total + best

    # The opponent acts: each possible hand follows the reference strategy.
    branches: dict[Action, list[tuple[Hand, float]]] = {}
    for hand, weight in live:
        for action, probability in reference.action_probs(hand).items():
            clone = copy.deepcopy(hand)
            clone.act(action)
            branches.setdefault(action, []).append((clone, weight * probability))
    return total + sum(_evaluate(group, me, reference) for group in branches.values())


def _advance(states: list[tuple[Hand, float]], chooser) -> list[tuple[Hand, float]]:
    out = []
    for hand, weight in states:
        for action, probability in chooser(hand).items():
            clone = copy.deepcopy(hand)
            clone.act(action)
            out.append((clone, weight * probability))
    return out


def q_values(
    game: Game, ctx: DecisionContext, reference, candidates: Sequence[Action]
) -> dict[Action, float]:
    """Expected net chips of each candidate action at the decision described by `ctx`."""
    me = ctx.seat
    posterior: list[tuple[Hand, float]] = []
    for deal in _opponent_deals(game, me, ctx.own_value):
        hand, likelihood = _replay(game, ctx, deal, reference)
        posterior.append((hand, likelihood))
    total = sum(weight for _hand, weight in posterior)
    if total <= 0:  # the opponent did something the reference would never do: drop the inference
        posterior = [(hand, 1.0) for hand, _ in posterior]
        total = float(len(posterior))
    posterior = [(hand, weight / total) for hand, weight in posterior]

    values = {}
    for action in candidates:
        values[action] = _evaluate(_advance(posterior, lambda hand, a=action: {a: 1.0}), me, reference)
    return values
