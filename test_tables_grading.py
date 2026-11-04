"""Grading wired into live tables: what is recorded, what is graded, what is stored."""

import asyncio
import random

import pytest

from app.games.base import CALL, CHECK, FOLD, bet
from app.games.bots import make_bot
from app.games.dice_duel import DiceDuel
from app.games.kuhn import Kuhn
from app.grading.grade import grade_match
from app.grading.stats import aggregate
from app.grading.store import MemoryGradeStore
from app.tables.core import Seat, TableCore
from helpers_tables import (
    ALICE, BOB, BOT_USER_ID, START, SimClient, act, make_lobby, passive_action, play_passively, run, wait_for,
)


def make_core(game=None, bot="kelly", max_hands=4, seed=1, buy_in=500):
    game = game or DiceDuel()
    seats = (Seat(1, "alice"), Seat(2, f"bot_{bot}" if bot else "bob", make_bot(bot) if bot else None))
    return TableCore(game, seats, buy_in, random.Random(seed), (random.Random(seed + 1), random.Random(seed + 2)), max_hands)


def passive(core):
    return CHECK if core.view_for(core.to_act)["decision"]["kind"] == "bet" else CALL


def play_out(core):
    while core.status == "playing":
        core.submit(core.to_act, passive(core))


# ================================================================ the decision log


def test_only_human_decisions_are_recorded_and_each_snapshot_is_accurate():
    core = make_core(max_hands=3)
    play_out(core)
    assert core.decision_log and all(c.seat == 0 for c in core.decision_log)  # seat 1 is the bot
    assert [c.hand_number for c in core.decision_log] == sorted(c.hand_number for c in core.decision_log)
    first = core.decision_log[0]
    assert first.hand_number == 1 and first.decision_index == 0 and first.history == ()
    assert first.stacks == (500, 500) and first.first == 0 and first.auto is False


def test_a_snapshot_contains_the_players_own_value_but_not_the_opponents():
    core = make_core()
    mine = core.hand.private_view(0)
    core.submit(0, CHECK)
    assert core.decision_log[0].own_value == mine
    assert not hasattr(core.decision_log[0], "opponent_value")


def test_refused_moves_are_never_recorded():
    core = make_core()
    for bad in (bet(10_000), CALL, FOLD):
        try:
            core.submit(0, bad)
        except Exception:  # noqa: BLE001
            pass
    assert core.decision_log == []


def test_automatic_moves_are_flagged_and_concessions_are_not_decisions():
    core = make_core(bot=None, max_hands=5)
    core.auto_play(0, "timeout")
    assert core.decision_log[-1].auto is True
    core.concede(1, "left")
    assert len(core.decision_log) == 1


def test_decision_index_counts_a_players_earlier_decisions_in_the_same_hand():
    core = make_core(game=Kuhn(), bot=None, buy_in=50, max_hands=2)  # a human on both seats
    core.submit(0, CHECK)  # seat 0's decision 0
    core.submit(1, bet(1))  # seat 1's decision 0
    core.submit(0, CALL)  # seat 0's decision 1 in the same hand
    by_seat = [(c.seat, c.decision_index) for c in core.decision_log]
    assert by_seat == [(0, 0), (1, 0), (0, 1)]
    assert core.decision_log[2].history == ((0, "check", 0), (1, "bet", 1))


# ================================================================== grading a match


def test_a_clean_match_grades_every_decision_and_the_nets_add_up():
    core = make_core(max_hands=6)
    play_out(core)
    graded = grade_match("dice", core.decision_log, core.results)
    rows = graded[0]
    assert graded[1] == [] and len(rows) == len(core.decision_log)
    assert all(r.regret >= 0 and r.ev_best >= r.ev_taken - 1e-9 for r in rows)
    counted = [r for r in rows if r.counts_for_ev]
    decided_hands = {c.hand_number for c in core.decision_log}
    assert len(counted) == len(decided_hands) <= core.hands_played  # one counted decision per hand you decided
    # Hands in which the human never had a decision (the bot checked, ending the hand) are not
    # counted. Add their results back and the books balance exactly.
    undecided = sum(res.net[0] for i, res in enumerate(core.results, 1) if i not in decided_hands)
    assert sum(r.hand_net for r in counted) + undecided == core.stacks[0] - 500


def test_only_the_last_decision_in_a_hand_counts_towards_ev():
    core = make_core(game=Kuhn(), bot=None, buy_in=50, max_hands=1)
    core.submit(0, CHECK)
    core.submit(1, bet(1))
    core.submit(0, CALL)
    graded = grade_match("kuhn", core.decision_log, core.results)
    seat0 = graded[0]
    assert [r.counts_for_ev for r in seat0] == [False, True]  # check (not counted), then call (counted)
    assert [r.counts_for_ev for r in graded[1]] == [True]


def test_hands_where_the_table_played_for_you_do_not_count_and_the_auto_moves_are_not_graded():
    core = make_core(bot="kelly", max_hands=3)
    core.submit(0, CHECK)  # hand 1: a real decision
    core.auto_play(0, "timeout")  # hand 2: the table moves for the player
    graded = grade_match("dice", core.decision_log, core.results)[0]
    assert all(r.hand_number != 2 for r in graded)  # the automatic move is not graded
    assert len({r.hand_number for r in graded}) <= 2


def test_a_hand_that_mixes_own_and_automatic_moves_does_not_count_for_ev():
    core = make_core(game=Kuhn(), bot=None, buy_in=50, max_hands=1)
    core.submit(0, CHECK)  # the player's own decision
    core.submit(1, bet(1))
    core.auto_play(0, "disconnected")  # but the response was automatic
    graded = grade_match("kuhn", core.decision_log, core.results)[0]
    assert len(graded) == 1 and graded[0].counts_for_ev is False  # regret still counts, EV does not


def test_grades_never_depend_on_the_opponents_hidden_value():
    """Re-run the same decisions against different hidden opponent values: identical grades."""
    graded = {}
    for seed in range(60):
        core = make_core(seed=seed, max_hands=1)
        core.submit(0, CHECK) if core.hand.to_act == 0 else None
        if core.decision_log:
            key = (core.decision_log[0].own_value,)
            g = grade_match("dice", core.decision_log, core.results)[0][0]
            graded.setdefault(key, set()).add((round(g.ev_taken, 9), round(g.ev_best, 9)))
    assert graded and all(len(v) == 1 for v in graded.values())


# ============================================================ through a live table


def test_a_live_match_is_graded_stored_and_summarised():
    async def body():
        store = MemoryGradeStore()
        lobby, ledger = make_lobby(grade_store=store, max_hands=4)
        table = await lobby.create_table(ALICE, "dice", 500, bot="kelly")
        a = SimClient()
        await lobby.connect(ALICE, table.id, a)
        await play_passively(table, [(ALICE, a)])

        over = a.match_over
        stored = store.decisions_for(1)
        assert len(stored) >= 2 and all(r["regret"] >= 0 for r in stored)
        decided = {r["hand_number"] for r in stored}
        summary = over["grading"]
        assert summary["decisions"] == len(stored)
        assert summary["hands_counted"] == len(decided) <= 4
        assert summary["regret"] == pytest.approx(round(sum(r["regret"] for r in stored), 2))
        ev = sum(r["ev_taken"] for r in stored if r["counts_for_ev"])
        assert summary["ev_earned"] == pytest.approx(round(ev, 2))
        # stored actuals plus the hands with no human decision explain the coins won or lost
        undecided = sum(res.net[0] for i, res in enumerate(table.core.results, 1) if i not in decided)
        actual = sum(r["hand_net"] for r in stored if r["counts_for_ev"])
        assert actual + undecided == over["net_change"]
        assert summary["luck"] == pytest.approx(round(actual - ev, 2))
        assert store.decisions_for(BOT_USER_ID) == []  # bots are not graded

    run(body())


def test_both_humans_are_graded_separately_and_only_see_their_own_summary():
    async def body():
        store = MemoryGradeStore()
        lobby, _ = make_lobby(grade_store=store, max_hands=4)
        table = await lobby.create_table(ALICE, "dice", 500)
        a, b = SimClient(), SimClient()
        await lobby.connect(ALICE, table.id, a)
        await lobby.connect(BOB, table.id, b)
        await play_passively(table, [(ALICE, a), (BOB, b)])
        rows_a, rows_b = store.decisions_for(1), store.decisions_for(2)
        assert a.match_over["grading"]["decisions"] == len(rows_a) > 0
        assert b.match_over["grading"]["decisions"] == len(rows_b) > 0
        # in any hand BOTH players decided in, one won exactly what the other lost
        net_a = {r["hand_number"]: r["hand_net"] for r in rows_a}
        net_b = {r["hand_number"]: r["hand_net"] for r in rows_b}
        shared = net_a.keys() & net_b.keys()
        assert all(net_a[h] == -net_b[h] for h in shared)

    run(body())


def test_a_grading_failure_never_affects_the_money():
    class BrokenStore:
        def save(self, *args, **kwargs):
            raise RuntimeError("disk full")

    async def body():
        lobby, ledger = make_lobby(grade_store=BrokenStore(), max_hands=3)
        table = await lobby.create_table(ALICE, "dice", 500, bot="kelly")
        a = SimClient()
        await lobby.connect(ALICE, table.id, a)
        await play_passively(table, [(ALICE, a)])
        assert a.match_over["your_balance"] == ledger.balance(1)  # settled normally
        assert a.match_over["grading"] is None  # just no grades
        assert sum(ledger.balance(u) for u in START) == sum(START.values())

    run(body())


def test_a_match_that_could_not_settle_is_not_graded():
    from app.tables.ledger import LedgerError, MemoryLedger

    class DeadLedger(MemoryLedger):
        def settle(self, game_ref, payouts):
            raise LedgerError("database is down")

    async def body():
        store = MemoryGradeStore()
        lobby, ledger = make_lobby(ledger=DeadLedger(dict(START)), grade_store=store, max_hands=2, settle_retries=1)
        table = await lobby.create_table(ALICE, "dice", 500, bot="kelly")
        a = SimClient()
        await lobby.connect(ALICE, table.id, a)
        await play_passively(table, [(ALICE, a)])
        assert a.errors() == ["settlement_failed"]
        assert store.decisions_for(1) == []  # the match will be refunded, so it must leave no grades

    run(body())


def test_timeouts_are_not_graded_as_the_players_decisions():
    async def body():
        store = MemoryGradeStore()
        lobby, _ = make_lobby(grade_store=store, turn_timeout=0.03, max_consecutive_timeouts=3, max_hands=50)
        table = await lobby.create_table(ALICE, "dice", 500, bot="kelly")
        a = SimClient()
        await lobby.connect(ALICE, table.id, a)
        await wait_for(lambda: a.match_over, what="match ended by timeouts")
        assert a.match_over["reason"] == "timeouts"
        assert store.decisions_for(1) == []  # every move was automatic: nothing to grade
        assert a.match_over["grading"]["decisions"] == 0

    run(body())


def test_saving_the_same_match_twice_cannot_double_count():
    store = MemoryGradeStore()
    core = make_core(max_hands=3)
    play_out(core)
    rows = grade_match("dice", core.decision_log, core.results)[0]
    store.save(7, 1, rows)
    store.save(7, 1, rows)  # a retry
    assert len(store.decisions_for(1)) == len(rows)
    store.save(8, 1, rows)  # a DIFFERENT match is added, not merged
    assert len(store.decisions_for(1)) == 2 * len(rows)
