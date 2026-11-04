"""Rules of Dice Duel and Kuhn poker, legality, table stakes, and hidden information."""

import random
from fractions import Fraction

import pytest

from app.games.base import (
    CALL, CHECK, FOLD, Action, IllegalActionError, bet, secure_rng,
)
from app.games.dice_duel import DiceDuel
from app.games.kuhn import Kuhn
from helpers_games import kuhn_value, nash_strategies, replay, walk

RNG = random.Random(0)


def dice_hand(deal, stacks=(1000, 1000), first=0):
    return DiceDuel().new_hand(RNG, stacks, first, deal=deal)


def kuhn_hand(deal, stacks=(100, 100), first=0):
    return Kuhn().new_hand(RNG, stacks, first, deal=deal)


# ------------------------------------------------------------------- actions


def test_action_validation():
    assert bet(5) == Action("bet", 5)
    with pytest.raises(ValueError):
        Action("raise")
    with pytest.raises(ValueError):
        bet(0)
    with pytest.raises(ValueError):
        Action("check", 5)
    with pytest.raises(TypeError):
        Action("bet", 2.5)
    with pytest.raises(TypeError):
        Action("bet", True)


# ----------------------------------------------------------------- dice duel


def test_dice_equity_for_each_roll():
    for roll in range(1, 7):
        d = dice_hand((roll, 3)).decision()
        assert d.win_prob == pytest.approx((roll - 1) / 6)
        assert d.tie_prob == pytest.approx(1 / 6)
        assert d.lose_prob == pytest.approx((6 - roll) / 6)
        assert d.win_prob + d.tie_prob + d.lose_prob == pytest.approx(1)


def test_antes_are_posted_and_stacks_shrink():
    d = dice_hand((4, 2)).decision()
    assert (d.stack, d.my_invested, d.opp_invested) == (990, 10, 10)
    assert d.kind == "bet" and (d.min_bet, d.max_bet) == (1, 990)


def test_check_goes_straight_to_showdown_for_the_antes():
    hand = dice_hand((5, 2))
    hand.act(CHECK)
    assert hand.done and hand.result.net == (10, -10) and hand.result.ended_by == "showdown"


def test_tied_dice_return_everyones_chips():
    hand = dice_hand((3, 3))
    hand.act(bet(100))
    hand.act(CALL)
    assert hand.result.net == (0, 0)


def test_bet_and_call_higher_die_wins_ante_plus_bet():
    hand = dice_hand((2, 6))
    hand.act(bet(100))
    hand.act(CALL)
    assert hand.result.net == (-110, 110)


def test_bet_and_fold_first_player_wins_the_ante_only():
    hand = dice_hand((1, 6))
    hand.act(bet(100))
    hand.act(FOLD)
    assert hand.result.net == (10, -10)
    assert hand.result.ended_by == "fold" and hand.result.folded == 1


def test_second_player_sees_the_bet_and_what_a_call_costs():
    hand = dice_hand((5, 2))
    hand.act(bet(100))
    d = hand.decision()
    assert d.kind == "respond" and d.to_call == 100 and d.opp_invested == 110 and d.my_invested == 10


def test_first_actor_is_the_button_seat():
    hand = dice_hand((5, 2), first=1)  # seat 0 holds the 5, seat 1 holds the 2
    assert hand.to_act == 1  # seat 1 is on the button, so it acts first
    hand.act(CHECK)
    assert hand.result.net == (10, -10)  # showdown: seat 0's 5 beats seat 1's 2


def test_table_stakes_cap_the_bet_at_the_smaller_stack():
    d = dice_hand((5, 2), stacks=(1000, 50)).decision()
    assert d.max_bet == 40  # the short stack has 50 - 10 ante behind


def test_short_stacked_player_can_only_check():
    hand = dice_hand((5, 2), stacks=(10, 1000))
    d = hand.decision()
    assert d.max_bet == 0 and not d.can_bet()
    with pytest.raises(IllegalActionError):
        hand.act(bet(1))
    hand.act(CHECK)
    assert hand.done


def test_illegal_actions_are_rejected_and_change_nothing():
    hand = dice_hand((5, 2))
    for bad in (bet(991), CALL, FOLD):
        with pytest.raises(IllegalActionError):
            hand.act(bad)
    assert hand.history == [] and hand.to_act == 0 and not hand.done
    hand.act(bet(10))
    for bad in (CHECK, bet(5)):
        with pytest.raises(IllegalActionError):
            hand.act(bad)


def test_nothing_can_happen_after_the_hand_is_over():
    hand = dice_hand((5, 2))
    hand.act(CHECK)
    with pytest.raises(IllegalActionError):
        hand.act(CHECK)
    with pytest.raises(IllegalActionError):
        hand.decision()


def test_cannot_start_a_hand_without_the_ante():
    with pytest.raises(ValueError):
        DiceDuel().new_hand(RNG, (5, 100), 0)
    with pytest.raises(ValueError):
        DiceDuel().new_hand(RNG, (100, 100), 2)


def test_invalid_deals_and_dice_are_rejected():
    with pytest.raises(ValueError):
        DiceDuel().new_hand(RNG, (100, 100), 0, deal=(0, 7))
    with pytest.raises(ValueError):
        DiceDuel(sides=1)


def test_same_seed_gives_the_same_deal_and_secure_rng_works():
    game = DiceDuel()
    a = game.new_hand(random.Random(9), (1000, 1000), 0)
    b = game.new_hand(random.Random(9), (1000, 1000), 0)
    assert (a.private_view(0), a.private_view(1)) == (b.private_view(0), b.private_view(1))
    live = game.new_hand(secure_rng(), (1000, 1000), 0)
    assert all(1 <= live.private_view(s) <= 6 for s in (0, 1))
    assert isinstance(secure_rng(), random.SystemRandom)


# ---------------------------------------------------------------------- kuhn


def test_kuhn_equity():
    assert [kuhn_hand((c, (c + 1) % 3)).decision().win_prob for c in range(3)] == [0.0, 0.5, 1.0]


def test_kuhn_check_check_is_a_one_ante_showdown():
    hand = kuhn_hand((2, 0))
    hand.act(CHECK)
    assert hand.stage == "second_open" and hand.to_act == 1
    hand.act(CHECK)
    assert hand.result.net == (1, -1)


def test_kuhn_check_bet_call_is_a_two_chip_showdown():
    hand = kuhn_hand((0, 2))
    hand.act(CHECK)
    hand.act(bet(1))
    d = hand.decision()
    assert d.kind == "respond" and hand.to_act == 0 and d.to_call == 1
    hand.act(CALL)
    assert hand.result.net == (-2, 2)


def test_kuhn_check_bet_fold_loses_only_the_ante():
    hand = kuhn_hand((0, 2))
    hand.act(CHECK)
    hand.act(bet(1))
    hand.act(FOLD)
    assert hand.result.net == (-1, 1)


def test_kuhn_bet_is_fixed_at_one_chip():
    d = kuhn_hand((1, 2)).decision()
    assert (d.min_bet, d.max_bet) == (1, 1)
    with pytest.raises(IllegalActionError):
        kuhn_hand((1, 2)).act(bet(2))


# ---------------------------------------------------- exhaustive game trees


@pytest.mark.parametrize("first", [0, 1])
def test_every_line_of_dice_duel_is_zero_sum_and_within_stacks(first):
    game = DiceDuel(sides=3)
    game.ante = 1
    stacks = (4, 4)
    terminals = 0
    for deal in game.all_deals():
        for _, result in walk(game, deal, stacks, first):
            terminals += 1
            assert sum(result.net) == 0
            assert all(net >= -stack for net, stack in zip(result.net, stacks))
    # per deal: 1 check line + 3 bet sizes x (call or fold) = 7
    assert terminals == 9 * 7


@pytest.mark.parametrize("first", [0, 1])
def test_every_line_of_kuhn_is_zero_sum(first):
    game = Kuhn()
    count = 0
    for deal in game.all_deals():
        for _, result in walk(game, deal, (3, 3), first):
            count += 1
            assert sum(result.net) == 0 and abs(result.net[0]) <= 2
    assert count == 6 * 5  # per deal: check-check, check-bet-call, check-bet-fold, bet-call, bet-fold


# ---------------------------------------------------------- hidden information


@pytest.mark.parametrize(
    ("make", "deal_a", "deal_b"),
    [(dice_hand, (5, 2), (5, 6)), (kuhn_hand, (1, 0), (1, 2))],
)
def test_a_player_cannot_tell_what_the_opponent_holds(make, deal_a, deal_b):
    """Two hands that differ ONLY in the opponent's hidden value look identical to seat 0."""
    a, b = make(deal_a), make(deal_b)
    assert a.decision() == b.decision()
    assert a.public_state() == b.public_state()
    assert a.private_view(0) == b.private_view(0)
    assert a.private_view(1) != b.private_view(1)


def test_responder_also_learns_nothing_beyond_the_bet():
    a, b = dice_hand((5, 2)), dice_hand((6, 2))  # the bettor's value differs
    for hand in (a, b):
        hand.act(bet(100))
    assert a.decision() == b.decision()  # seat 1 cannot tell a 5 from a 6
    assert a.public_state() == b.public_state()


def test_folded_hands_stay_hidden_but_showdowns_are_revealed():
    folded = dice_hand((5, 2))
    folded.act(bet(10))
    folded.act(FOLD)
    assert folded.public_state()["showdown_values"] is None
    assert folded.result.private_values == (5, 2)  # server-side record still has them

    shown = dice_hand((5, 2))
    shown.act(CHECK)
    assert shown.public_state()["showdown_values"] == (5, 2)


def test_public_state_contains_no_private_fields_midhand():
    state = dice_hand((5, 2)).public_state()
    assert set(state) == {"game", "stage", "done", "to_act", "pot", "invested", "stacks", "history"}


# ------------------------------------------------ Kuhn poker's published value


@pytest.mark.parametrize("alpha", [0, Fraction(1, 6), Fraction(1, 3)])
def test_kuhn_equilibrium_gives_the_first_player_exactly_minus_one_eighteenth(alpha):
    """Kuhn (1950): with optimal play the first player loses 1/18 of an ante per hand.

    This runs the equilibrium strategies through MY engine's rules, so it checks that the
    engine implements Kuhn poker correctly.
    """
    first, second = nash_strategies(alpha)
    assert kuhn_value(Kuhn(), first, second) == Fraction(-1, 18)


def test_always_betting_against_a_player_who_always_folds_wins_one_ante_per_hand():
    always_bet = {(c, h): 1 for c in "JQK" for h in ("", "pb")}
    never_call_or_bet = {(c, h): 0 for c in "JQK" for h in ("b", "p")}
    assert kuhn_value(Kuhn(), always_bet, never_call_or_bet) == 1


def test_two_players_who_only_ever_check_have_equal_chances():
    never = {(c, h): 0 for c in "JQK" for h in ("", "p", "b", "pb")}
    assert kuhn_value(Kuhn(), never, never) == 0


def test_engine_simulation_converges_to_the_exact_kuhn_value():
    """Play 60,000 random hands through the engine with equilibrium strategies."""
    first, second = nash_strategies(Fraction(1, 6))
    strategies = (first, second)
    game, rng = Kuhn(), random.Random(2026)
    from helpers_games import CARD, history_key

    n, total, total_sq = 60_000, 0, 0
    for _ in range(n):
        hand = game.new_hand(rng, (100, 100), 0)
        history = ()
        while not hand.done:
            seat = hand.to_act
            kind = hand.decision().kind
            p = float(strategies[seat][(CARD[hand.private_view(seat)], history_key(history))])
            if kind == "bet":
                action = bet(1) if rng.random() < p else CHECK
            else:
                action = CALL if rng.random() < p else FOLD
            hand.act(action)
            history += (action,)
        net = hand.result.net[0]
        total += net
        total_sq += net * net
    mean = total / n
    standard_error = ((total_sq / n - mean**2) / n) ** 0.5
    assert abs(mean - (-1 / 18)) < 4 * standard_error


# ----------------------------------------------------------------- concede


def test_conceding_loses_what_you_put_in_even_when_it_is_not_your_turn():
    hand = dice_hand((5, 2))
    hand.act(bet(100))  # seat 0 has 110 in the pot; it is seat 1's turn
    hand.concede(0)
    assert hand.result.net == (-110, 110)
    assert hand.result.ended_by == "fold" and hand.result.folded == 0
    assert hand.result.showdown_values is None  # a concession reveals nothing


def test_conceding_at_the_start_costs_only_the_ante():
    hand = kuhn_hand((2, 0))
    hand.concede(1)
    assert hand.result.net == (1, -1)


def test_a_finished_hand_cannot_be_conceded():
    hand = dice_hand((5, 2))
    hand.act(CHECK)
    with pytest.raises(IllegalActionError):
        hand.concede(0)
