"""Helpers for game-engine tests: replay a hand from its history and walk every possible line of play."""

import random
from fractions import Fraction

from app.games.base import CALL, CHECK, FOLD, bet


def replay(game, deal, stacks, first, history=()):
    """Rebuild a hand from scratch and apply `history` (a sequence of Actions)."""
    hand = game.new_hand(random.Random(0), stacks, first, deal=deal)
    for action in history:
        hand.act(action)
    return hand


def legal_actions(decision):
    """Every legal action in this decision."""
    if decision.kind == "respond":
        return [CALL, FOLD]
    actions = [CHECK]
    if decision.can_bet():
        actions += [bet(a) for a in range(decision.min_bet, decision.max_bet + 1)]
    return actions


def walk(game, deal, stacks, first, history=()):
    """Yield (history, HandResult) for every terminal line of play from this position."""
    hand = replay(game, deal, stacks, first, history)
    if hand.done:
        yield history, hand.result
        return
    for action in legal_actions(hand.decision()):
        yield from walk(game, deal, stacks, first, history + (action,))


CARD = ("J", "Q", "K")


def history_key(history):
    """Kuhn information-set history: 'p' = check/fold, 'b' = bet/call."""
    return "".join("b" if a.kind in ("bet", "call") else "p" for a in history)


def kuhn_value(game, strategy0, strategy1):
    """EXACT expected net chips for seat 0 (who acts first) under mixed strategies.

    A strategy maps (card_letter, history) -> probability of the aggressive action
    (bet when betting, call when responding).
    """
    strategies = (strategy0, strategy1)
    stacks = (50, 50)

    def node(deal, history):
        hand = replay(game, deal, stacks, 0, history)
        if hand.done:
            return Fraction(hand.result.net[0])
        seat = hand.to_act
        decision = hand.decision()
        p = Fraction(strategies[seat][(CARD[deal[seat]], history_key(history))])
        aggressive, passive = (bet(1), CHECK) if decision.kind == "bet" else (CALL, FOLD)
        value = Fraction(0)
        if p > 0:
            value += p * node(deal, history + (aggressive,))
        if p < 1:
            value += (1 - p) * node(deal, history + (passive,))
        return value

    deals = game.all_deals()
    return sum((node(deal, ()) for deal in deals), Fraction(0)) / len(deals)


def nash_strategies(alpha):
    """Kuhn's equilibrium family (alpha between 0 and 1/3). First player's value is -1/18."""
    alpha = Fraction(alpha)
    third = Fraction(1, 3)
    first = {
        ("J", ""): alpha, ("Q", ""): Fraction(0), ("K", ""): 3 * alpha,
        ("J", "pb"): Fraction(0), ("Q", "pb"): alpha + third, ("K", "pb"): Fraction(1),
    }
    second = {
        ("J", "b"): Fraction(0), ("Q", "b"): third, ("K", "b"): Fraction(1),
        ("J", "p"): third, ("Q", "p"): Fraction(0), ("K", "p"): Fraction(1),
    }
    return first, second
