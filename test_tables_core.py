"""TableCore: the match state machine, with no sockets, timers or database."""

import random

import pytest

from app.games.base import CALL, CHECK, FOLD, bet
from app.games.bots import make_bot
from app.games.dice_duel import DiceDuel
from app.games.kuhn import Kuhn
from app.tables.common import TableError
from app.tables.core import Seat, TableCore


def make_core(game=None, buy_in=500, max_hands=3, bot=None, seed=1):
    game = game or DiceDuel()
    seats = (Seat(1, "alice"), Seat(2, "bob" if bot is None else f"bot_{bot}", make_bot(bot) if bot else None))
    return TableCore(game, seats, buy_in, random.Random(seed), (random.Random(seed + 1), random.Random(seed + 2)), max_hands)


def passive(core):
    return CHECK if core.view_for(core.to_act)["decision"]["kind"] == "bet" else CALL


def play_out(core):
    while core.status == "playing":
        core.submit(core.to_act, passive(core))


def test_a_match_runs_to_max_hands_and_conserves_chips():
    core = make_core(max_hands=5)
    play_out(core)
    assert core.status == "finished" and core.end_reason == "max_hands" and core.hands_played == 5
    assert sum(core.stacks) == 1000 and len(core.results) == 5


def test_the_button_alternates_every_hand():
    core = make_core(max_hands=4)
    firsts = []
    while core.status == "playing":
        firsts.append(core.to_act)
        core.submit(core.to_act, passive(core))
    assert firsts == [0, 1, 0, 1]


def test_only_the_player_to_act_may_move_and_state_is_unchanged_on_error():
    core = make_core()
    before = core.view_for(0)
    with pytest.raises(TableError) as info:
        core.submit(1, CHECK)
    assert info.value.code == "not_your_turn"
    assert core.view_for(0) == before and core.hands_played == 0


def test_illegal_actions_are_rejected_without_changing_anything():
    core = make_core()
    before = core.view_for(0)
    for bad in (bet(491), CALL, FOLD):
        with pytest.raises(TableError) as info:
            core.submit(0, bad)
        assert info.value.code == "illegal_action"
    assert core.view_for(0) == before


def test_no_moves_after_the_match_is_over():
    core = make_core(max_hands=1)
    play_out(core)
    with pytest.raises(TableError) as info:
        core.submit(0, CHECK)
    assert info.value.code == "match_over"
    assert core.auto_play(0, "timeout") == [] and core.concede(0, "x") == []


def test_bust_ends_the_match():
    core = make_core(buy_in=20, max_hands=50)  # ante 10: two hands can wipe a stack
    while core.status == "playing":
        seat = core.to_act
        d = core.view_for(seat)["decision"]
        core.submit(seat, bet(d["max_bet"]) if d["kind"] == "bet" and d["max_bet"] else passive(core))
    assert core.end_reason in ("bust", "max_hands") and sum(core.stacks) == 40
    if core.end_reason == "bust":
        assert min(core.stacks) < core.game.ante


def test_bots_act_inside_the_core_until_a_human_is_needed():
    core = make_core(bot="kelly", max_hands=4)
    assert core.to_act == 0  # the human opens hand 1
    play_out(core)
    assert core.hands_played == 4 and sum(core.stacks) == 1000


def test_a_bot_that_must_open_acts_immediately():
    core = make_core(bot="ev", max_hands=2)
    core.submit(0, CHECK)  # hand 1 ends; the bot is on the button for hand 2 and acts at once
    assert core.hands_played >= 1
    assert core.to_act in (0, None)  # never left waiting on a bot


def test_auto_play_checks_when_it_can_and_folds_when_facing_a_bet():
    core = make_core()
    events = core.auto_play(0, "timeout")
    assert events[0] == {"type": "auto_action", "seat": 0, "action": "check", "reason": "timeout"}

    core = make_core()
    core.submit(0, bet(100))
    events = core.auto_play(1, "disconnected")
    assert events[0]["action"] == "fold"
    result = [e for e in events if e["type"] == "hand_result"][0]
    assert result["ended_by"] == "fold" and result["folded"] == 1 and result["net"] == [10, -10]


def test_auto_play_does_nothing_out_of_turn():
    assert make_core().auto_play(1, "timeout") == []


def test_concede_loses_the_chips_in_the_hand_and_ends_the_match():
    core = make_core()
    core.submit(0, bet(100))  # seat 0 has 110 in; now it is seat 1's turn
    events = core.concede(0, "left")  # the bettor leaves while it is NOT their turn
    assert core.status == "finished" and core.end_reason == "left"
    assert core.stacks == [500 - 110, 500 + 110]
    over = [e for e in events if e["type"] == "match_over"][0]
    assert over["conceded_by"] == 0 and over["final_stacks"] == [390, 610]


def test_conceding_keeps_the_rest_of_the_stack_and_conserves_chips():
    core = make_core()
    core.concede(1, "disconnected")
    assert core.stacks == [510, 490] and sum(core.stacks) == 1000


def test_hand_results_hide_the_cards_after_a_fold_and_show_them_at_showdown():
    core = make_core()
    core.submit(0, bet(50))
    folded = [e for e in core.submit(1, FOLD) if e["type"] == "hand_result"][0]
    assert folded["showdown_values"] is None
    shown = [e for e in core.submit(1, CHECK) if e["type"] == "hand_result"][0]
    assert shown["showdown_values"] is not None


def test_a_players_view_never_depends_on_the_opponents_hidden_value():
    """Find two deals where seat 0 holds the same roll and seat 1 differs; seat 0's view must match."""
    views = {}
    for seed in range(200):
        core = make_core(seed=seed)
        mine, theirs = core.hand.private_view(0), core.hand.private_view(1)
        views.setdefault((mine, theirs), core.view_for(0))
    pairs = [(m, t) for (m, t) in views if m == 4]
    assert len(pairs) >= 3
    first = views[pairs[0]]
    assert all(views[p] == first for p in pairs)


def test_view_has_a_decision_only_for_the_player_to_act():
    core = make_core()
    assert core.view_for(0)["decision"]["actions"] == ["check", "bet"]
    assert core.view_for(1)["decision"] is None
    core.submit(0, bet(10))
    assert core.view_for(1)["decision"]["actions"] == ["call", "fold"]
    assert core.view_for(1)["decision"]["to_call"] == 10


def test_kuhn_works_at_a_table_too():
    core = make_core(game=Kuhn(), buy_in=50, max_hands=6, bot="kelly")
    play_out(core)
    assert core.hands_played == 6 and sum(core.stacks) == 100


def test_same_seeds_replay_the_same_match():
    def final(seed):
        core = make_core(bot="random", seed=seed, max_hands=10)
        play_out(core)
        return core.stacks

    assert final(3) == final(3)


def test_constructor_validates_arguments():
    with pytest.raises(ValueError):
        make_core(buy_in=5)
    with pytest.raises(ValueError):
        make_core(max_hands=0)
