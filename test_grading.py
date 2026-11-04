"""Decision grading: exact EVs checked against independent hand formulas and against theory."""

from fractions import Fraction

import pytest

from app.games.base import CALL, CHECK, FOLD, Action, bet
from app.games.dice_duel import DiceDuel
from app.games.kuhn import Kuhn
from app.grading.evaluate import legal_actions, q_values
from app.grading.grade import candidate_actions, grade_decision
from app.grading.reference import EVReference, KuhnNashReference, reference_for
from app.grading.types import DecisionContext

DICE, KUHN = DiceDuel(), Kuhn()
REF_DICE, REF_KUHN = reference_for("dice_duel"), reference_for("kuhn")
ANTE = 10


def ctx(own, kind, amount=0, history=(), seat=0, first=0, stacks=(1000, 1000), index=0):
    return DecisionContext(1, index, seat, own, stacks, first, tuple(history), kind, amount, False)


def sign(x):
    return (x > 0) - (x < 0)


# ---- independent formulas for Dice Duel (written separately from the engine-based evaluator)
def caller_calls(value, bet_size, sides=6):
    edge = ((value - 1) - (sides - value)) / sides
    return edge * (ANTE + bet_size) > -ANTE


def ev_check(roll, sides=6):
    return sum(sign(roll - c) * ANTE for c in range(1, sides + 1)) / sides


def ev_bet(roll, size, sides=6):
    total = 0
    for c in range(1, sides + 1):
        total += sign(roll - c) * (ANTE + size) if caller_calls(c, size) else ANTE
    return total / sides


def ev_respond(own, bet_size, call):
    """Facing a bet, the reference bettor's range is {4, 5, 6} (it only bets with an edge)."""
    if not call:
        return -ANTE
    return sum(sign(own - o) * (ANTE + bet_size) for o in (4, 5, 6)) / 3


# ============================================================ Dice Duel: the bettor


@pytest.mark.parametrize("roll", range(1, 7))
def test_check_and_bet_values_match_independent_formulas(roll):
    sizes = [1, 2, 9, 10, 11, 49, 50, 51, 200, 990]
    q = q_values(DICE, ctx(roll, "check"), REF_DICE, [CHECK] + [bet(s) for s in sizes])
    assert q[CHECK] == pytest.approx(ev_check(roll))
    for size in sizes:
        assert q[bet(size)] == pytest.approx(ev_bet(roll, size)), (roll, size)


def test_the_worked_example_from_the_docs():
    """A 4 against the reference: checking is worth +1.67, a 49-coin bet loses 6.50."""
    q = q_values(DICE, ctx(4, "check"), REF_DICE, [CHECK, bet(1), bet(49), bet(50)])
    assert q[CHECK] == pytest.approx(5 / 3)
    assert q[bet(1)] == pytest.approx(11 / 6)  # a tiny bet is called by everyone: a thin value bet
    assert q[bet(49)] == pytest.approx(-6.5)
    assert q[bet(50)] < -10


@pytest.mark.parametrize("stacks", [(1000, 1000), (500, 500), (200, 90), (45, 45), (1000, 60), (30, 30), (15, 400)])
@pytest.mark.parametrize("roll", range(1, 7))
def test_checking_a_handful_of_bet_sizes_finds_the_same_best_as_checking_every_size(roll, stacks):
    """The grader only evaluates sizes at the reference caller's break points. This proves that
    finds the true optimum by comparing with a brute-force scan of EVERY legal size."""
    context = ctx(roll, "check", stacks=stacks)
    graded = grade_decision(DICE, context, REF_DICE)
    cap = min(stacks) - ANTE
    brute = max([ev_check(roll)] + [ev_bet(roll, b) for b in range(1, cap + 1)]) if cap >= 1 else ev_check(roll)
    assert graded["ev_best"] == pytest.approx(brute)


def test_a_strong_hand_should_bet_big_and_a_weak_hand_should_not_bet():
    best_6 = grade_decision(DICE, ctx(6, "check"), REF_DICE)["best"]
    assert best_6 == bet(990)
    assert grade_decision(DICE, ctx(1, "bet", 990), REF_DICE)["best"] == CHECK


def test_regret_is_the_gap_to_the_best_and_zero_for_the_best():
    perfect = grade_decision(DICE, ctx(6, "bet", 990), REF_DICE)
    assert perfect["regret"] == 0 and perfect["ev_taken"] == pytest.approx(perfect["ev_best"])
    timid = grade_decision(DICE, ctx(6, "check"), REF_DICE)
    assert timid["regret"] == pytest.approx(timid["ev_best"] - timid["ev_taken"]) and timid["regret"] > 300
    reckless = grade_decision(DICE, ctx(4, "bet", 990), REF_DICE)
    assert reckless["regret"] == pytest.approx(1.8333333 + 328.33333, rel=1e-4)


def test_regret_is_never_negative_for_any_legal_decision():
    for roll in range(1, 7):
        for action in [CHECK, bet(1), bet(10), bet(49), bet(50), bet(500), bet(990)]:
            g = grade_decision(DICE, ctx(roll, action.kind, action.amount), REF_DICE)
            assert g["regret"] >= 0 and g["ev_best"] >= g["ev_taken"] - 1e-9


def test_tied_actions_report_the_simplest_one_as_best():
    """For a roll of 1, 'check' and 'bet 2' are worth exactly the same; report 'check'."""
    g = grade_decision(DICE, ctx(1, "bet", 990), REF_DICE)
    assert ev_bet(1, 2) == pytest.approx(ev_check(1))
    assert g["best"] == CHECK


def test_the_action_taken_is_reported_as_best_when_it_ties_for_best():
    g = grade_decision(DICE, ctx(1, "bet", 2), REF_DICE)
    assert g["regret"] == 0 and g["best"] == bet(2)


# ======================================================= Dice Duel: the responder


@pytest.mark.parametrize("own", range(1, 7))
@pytest.mark.parametrize("bet_size", [1, 5, 50, 500])
def test_call_and_fold_values_match_independent_formulas(own, bet_size):
    history = ((0, "bet", bet_size),)
    context = ctx(own, "call", seat=1, history=history)
    q = q_values(DICE, context, REF_DICE, [CALL, FOLD])
    assert q[CALL] == pytest.approx(ev_respond(own, bet_size, True))
    assert q[FOLD] == pytest.approx(ev_respond(own, bet_size, False))


def test_responder_decisions_that_are_clearly_right_or_wrong():
    # a 5 facing 100: the bettor's range is {4,5,6}; a call breaks even against it, folding loses the ante
    assert grade_decision(DICE, ctx(5, "call", seat=1, history=((0, "bet", 100),)), REF_DICE)["regret"] == 0
    assert grade_decision(DICE, ctx(5, "fold", seat=1, history=((0, "bet", 100),)), REF_DICE)["regret"] == pytest.approx(10)
    # a 2 facing 100 should fold; calling loses 110 vs 10
    assert grade_decision(DICE, ctx(2, "fold", seat=1, history=((0, "bet", 100),)), REF_DICE)["regret"] == 0
    assert grade_decision(DICE, ctx(2, "call", seat=1, history=((0, "bet", 100),)), REF_DICE)["regret"] == pytest.approx(100)


def test_a_bet_shifts_the_posterior_to_strong_hands():
    """After a bet the opponent is assumed to hold a 4, 5 or 6, so a 4 should fold to a big bet
    but call a tiny one (it risks only a coin more)."""
    big = grade_decision(DICE, ctx(4, "call", seat=1, history=((0, "bet", 100),)), REF_DICE)
    small = grade_decision(DICE, ctx(4, "fold", seat=1, history=((0, "bet", 1),)), REF_DICE)
    assert big["best"] == FOLD and small["best"] == CALL


def test_the_opponent_checking_means_a_weak_range_and_never_matters_in_dice():
    """A check ends a Dice Duel hand at once, so there is no later decision to grade."""
    first = ctx(3, "check")
    assert candidate_actions(DICE, first, REF_DICE)[0] == "bet"


# =============================================================== Kuhn: Nash theory

CARD_LETTER = "JQK"


def test_first_player_best_response_value_is_exactly_minus_one_eighteenth():
    """Against the equilibrium opponent, the best the first player can do is the game's value."""
    values = [grade_decision(KUHN, ctx(card, "check", stacks=(50, 50)), REF_KUHN)["ev_best"] for card in range(3)]
    assert sum(values) / 3 == pytest.approx(-1 / 18)


def test_second_player_best_response_value_is_exactly_plus_one_eighteenth():
    """Weight the best-response value at each information set by the chance of reaching it."""
    nash = KuhnNashReference.TABLE
    total = 0.0
    for card in range(3):  # the second player's card (each 1/3)
        for action in ("bet", "check"):  # what the first player did
            reach = sum(
                0.5 * (nash[(CARD_LETTER[o], "")] if action == "bet" else 1 - nash[(CARD_LETTER[o], "")])
                for o in range(3) if o != card
            )
            if reach == 0:
                continue
            history = ((0, action, 1 if action == "bet" else 0),)
            value = grade_decision(
                KUHN, ctx(card, "call" if action == "bet" else "check", seat=1, history=history, stacks=(50, 50)),
                REF_KUHN,
            )["ev_best"]
            total += (1 / 3) * reach * value
    assert total == pytest.approx(1 / 18)


EQUILIBRIUM_INFOSETS = [
    # (card, history actions, seat, kind of decision)
    *[(c, (), 0) for c in range(3)],  # first player's opening
    *[(c, ((0, "check", 0), (1, "bet", 1)), 0) for c in range(3)],  # first player facing a bet after a check
    *[(c, ((0, "bet", 1),), 1) for c in range(3)],  # second player facing a bet
    *[(c, ((0, "check", 0),), 1) for c in range(3)],  # second player after a check
]


@pytest.mark.parametrize(("card", "history", "seat"), EQUILIBRIUM_INFOSETS)
def test_every_equilibrium_action_has_exactly_zero_regret(card, history, seat):
    """THE definition of a Nash equilibrium: against an equilibrium opponent, each action the
    equilibrium plays with positive probability is a best response. If my evaluator or my
    reference table were wrong, this would fail."""
    letter = CARD_LETTER[card]
    key = "".join("b" if kind in ("bet", "call") else "p" for _seat, kind, _a in history)
    p_aggressive = KuhnNashReference.TABLE[(letter, key)]
    respond = bool(history) and history[-1][1] == "bet"
    aggressive = ("call", 0) if respond else ("bet", 1)
    passive = ("fold", 0) if respond else ("check", 0)
    for (kind, amount), probability in ((aggressive, p_aggressive), (passive, 1 - p_aggressive)):
        if probability > 1e-12:
            g = grade_decision(KUHN, ctx(card, kind, amount, history=history, seat=seat, stacks=(50, 50)), REF_KUHN)
            assert g["regret"] == pytest.approx(0, abs=1e-9), (letter, key, kind)


def test_a_clear_kuhn_mistake_has_positive_regret():
    # holding the King, folding to a bet can never be right
    history = ((0, "check", 0), (1, "bet", 1))
    g = grade_decision(KUHN, ctx(2, "fold", history=history, seat=0, stacks=(50, 50)), REF_KUHN)
    assert g["regret"] > 0.5 and g["best"] == CALL
    # holding the Jack, calling a bet is a mistake against the equilibrium
    g = grade_decision(KUHN, ctx(0, "call", history=((0, "bet", 1),), seat=1, stacks=(50, 50)), REF_KUHN)
    assert g["regret"] > 0 and g["best"] == FOLD


# ===================================================================== robustness


class NeverBets:
    """A stub reference that never bets, to force an 'impossible' observation."""

    def action_probs(self, hand):
        d = hand.decision()
        if d.kind == "respond":
            return {CALL: 1.0}
        return {CHECK: 1.0}

    def bet_breakpoints(self, hand):
        return []


def test_an_impossible_opponent_action_falls_back_to_a_uniform_belief():
    """If no opponent hand could have produced what we saw, drop the inference instead of crashing."""
    history = ((0, "check", 0), (1, "bet", 1))  # the stub says the opponent NEVER bets
    q = q_values(KUHN, ctx(1, "call", history=history, seat=0, stacks=(50, 50)), NeverBets(), [CALL, FOLD])
    # holding the Queen against a uniformly random Jack or King: a call wins or loses 2, fold loses 1
    assert q[CALL] == pytest.approx(0.0) and q[FOLD] == pytest.approx(-1.0)


def test_references_are_chosen_by_game_name():
    assert isinstance(reference_for("dice_duel"), EVReference)
    assert isinstance(reference_for("kuhn"), KuhnNashReference)
    with pytest.raises(ValueError):
        reference_for("poker")


def test_only_a_few_candidate_sizes_are_checked_for_big_stacks_but_the_taken_size_is_always_included():
    _kind, actions = candidate_actions(DICE, ctx(4, "bet", 333), REF_DICE)
    sizes = [a.amount for a in actions if a.kind == "bet"]
    assert 333 in sizes and 990 in sizes and 1 in sizes and len(sizes) < 20
    _kind, small = candidate_actions(DICE, ctx(4, "check", stacks=(30, 30)), REF_DICE)
    assert len([a for a in small if a.kind == "bet"]) == 20  # every size when there are few


def test_the_short_list_of_bet_sizes_finds_the_true_best_for_every_roll_and_every_stack():
    """Exhaustive: all 6 rolls x all 990 possible stack caps. The grader only evaluates a few bet
    sizes; this proves that is exactly as good as evaluating every legal size, for every case."""
    for roll in range(1, 7):
        everything = [ev_bet(roll, b) for b in range(1, 991)]  # EV of bet 1 .. bet 990
        prefix_best = []
        running = ev_check(roll)
        for value in everything:
            running = max(running, value)
            prefix_best.append(running)  # best of {check, bet 1 .. bet b}
        for cap in range(1, 991):
            stacks = (cap + ANTE, cap + ANTE)  # the bet limit is the smaller stack minus the ante
            _kind, actions = candidate_actions(DICE, ctx(roll, "check", stacks=stacks), REF_DICE)
            shortlist = [ev_check(roll)] + [ev_bet(roll, a.amount) for a in actions if a.kind == "bet"]
            assert max(shortlist) == pytest.approx(prefix_best[cap - 1]), (roll, cap)
