"""The two leaderboards, end to end: simulated players of different quality play real tables
against bots; the grades and the ranking must tell them apart."""

import asyncio

import pytest

from app.games.base import CALL, CHECK, FOLD, bet
from app.games.dice_duel import DiceDuel
from app.grading.grade import grade_decision
from app.grading.reference import reference_for
from app.grading.service import Standings
from app.grading.store import MemoryGradeStore
from app.grading.types import DecisionContext
from app.tables.common import UserRef
from helpers_tables import ALICE, BOB, BOT_USER_ID, CAROL, SimClient, act, make_lobby, run

GAME, REF = DiceDuel(), reference_for("dice_duel")
NAMES = {1: ("alice", False), 2: ("bob", False), 3: ("carol", False), BOT_USER_ID: ("bot_random", True)}


def current_context(table, seat) -> DecisionContext:
    core, hand = table.core, table.core.hand
    kind = hand.decision().kind
    return DecisionContext(
        core.hands_played + 1, sum(1 for s, _ in hand.history if s == seat), seat, hand.private_view(seat),
        (core.stacks[0], core.stacks[1]), hand.first, tuple((s, a.kind, a.amount) for s, a in hand.history),
        "check" if kind == "bet" else "call", 0, False,
    )


def skilled(table, seat, client):
    """Always plays the action the grader says is best."""
    return grade_decision(GAME, current_context(table, seat), REF)["best"]


def reckless(table, seat, client):
    d = client.state["decision"]
    return bet(d["max_bet"]) if d["kind"] == "bet" and d["max_bet"] else (CALL if d["kind"] == "respond" else CHECK)


def timid(table, seat, client):
    return CHECK if client.state["decision"]["kind"] == "bet" else FOLD


async def play_matches(lobby, user, policy, matches=3, bot="random", buy_in=100):
    for _ in range(matches):
        table = await lobby.create_table(user, "dice", buy_in, bot=bot)
        client = SimClient(user.name)
        await lobby.connect(user, table.id, client)
        for _ in range(5000):
            if table.state != "playing":
                break
            if client.my_turn():
                seat = client.state["you"]
                action = policy(table, seat, client)
                await act(table, user, client, action.kind, action.amount or None)
            await asyncio.sleep(0)
        assert table.state == "closed"


def build():
    store = MemoryGradeStore()
    lobby, ledger = make_lobby(grade_store=store, max_hands=12)
    standings = Standings(store, ledger, lambda: NAMES)
    return lobby, ledger, store, standings


def test_the_grades_tell_skilled_reckless_and_timid_players_apart():
    async def body():
        lobby, ledger, store, standings = build()
        await play_matches(lobby, ALICE, skilled)
        await play_matches(lobby, BOB, reckless)
        await play_matches(lobby, CAROL, timid)

        totals = store.skill_totals()
        assert totals[1].regret == pytest.approx(0, abs=1e-6)  # best action every time: zero regret
        assert totals[2].regret > 50 and totals[3].regret > 5
        per_hand = {u: totals[u].ev_earned / totals[u].hands for u in totals}
        assert per_hand[1] > per_hand[2] + 10  # reckless play is expensive even after the luck of the deal
        # Dice Duel has one decision per player per hand, so EV earned = deal value - regret EXACTLY
        for u in totals:
            assert totals[u].ev_earned == pytest.approx(totals[u].ev_potential - totals[u].regret)
        assert totals[1].ev_earned == pytest.approx(totals[1].ev_potential)  # perfect play earns the deal's full value

        boards = standings.both()
        ranks = {r["username"]: r["rank"] for r in boards["skill"]}
        assert ranks["alice"] < ranks["bob"]  # reckless play ranks below skilled play
        assert {r["username"] for r in boards["skill"]} == {"alice", "bob", "carol"}  # no bot
        assert {r["username"] for r in boards["coins"]} == {"alice", "bob", "carol"}

    run(body())


def test_every_row_shows_the_rank_on_the_other_board_consistently():
    async def body():
        lobby, ledger, store, standings = build()
        for user, policy in ((ALICE, skilled), (BOB, reckless), (CAROL, timid)):
            await play_matches(lobby, user, policy, matches=2)
        boards = standings.both()
        skill_rank = {r["user_id"]: r["rank"] for r in boards["skill"]}
        coin_rank = {r["user_id"]: r["rank"] for r in boards["coins"]}
        for row in boards["skill"]:
            assert row["coin_rank"] == coin_rank[row["user_id"]]
            assert row["rank_gap"] == coin_rank[row["user_id"]] - row["rank"]
        for row in boards["coins"]:
            assert row["skill_rank"] == skill_rank[row["user_id"]]

    run(body())


def test_luck_is_actual_coins_minus_expected_coins_and_nets_are_real():
    async def body():
        lobby, ledger, store, standings = build()
        await play_matches(lobby, ALICE, skilled, matches=3)
        row = standings.both()["skill"][0]
        assert row["luck"] == pytest.approx(row["actual_net"] - row["ev_earned"])
        assert row["ev_per_hand"] == pytest.approx(row["ev_earned"] / row["hands"])
        assert row["regret_per_decision"] == pytest.approx(row["regret"] / row["decisions"])

    run(body())


def test_the_coins_board_matches_the_ledger_and_no_coin_is_created():
    async def body():
        lobby, ledger, store, standings = build()
        await play_matches(lobby, ALICE, skilled)
        await play_matches(lobby, BOB, reckless)
        net = ledger.net_coin_rows()
        assert sum(net.values()) == 0  # every coin one player won, someone (a player or a bot) lost
        coins = {r["user_id"]: r["net_coins"] for r in standings.both()["coins"]}
        assert coins[1] == net[1] and coins[2] == net[2]
        assert ledger.balance(1) == 1000 + net[1] and ledger.balance(2) == 1000 + net[2]

    run(body())


def test_the_coins_board_can_disagree_with_the_skill_board():
    """Luck is real: across many seeds, somebody's coin rank and skill rank differ."""
    async def body():
        gaps = set()
        for _ in range(1):
            lobby, ledger, store, standings = build()
            for user, policy in ((ALICE, skilled), (BOB, timid), (CAROL, reckless)):
                await play_matches(lobby, user, policy, matches=2)
            gaps |= {r["rank_gap"] for r in standings.both()["skill"]}
        assert gaps  # computed for everyone (whether any is non-zero depends on the dice)

    run(body())


def test_min_hands_hides_players_with_too_little_data():
    async def body():
        lobby, ledger, store, standings = build()
        await play_matches(lobby, ALICE, skilled, matches=1)
        hands = standings.both()["skill"][0]["hands"]
        assert standings.both(min_hands=hands)["skill"] != []
        assert standings.both(min_hands=hands + 1)["skill"] == []
        assert standings.both(min_hands=hands + 1)["coins"] != []  # coins board has no minimum

    run(body())


def test_limit_truncates_but_ranks_are_computed_first():
    async def body():
        lobby, ledger, store, standings = build()
        for user, policy in ((ALICE, skilled), (BOB, reckless), (CAROL, timid)):
            await play_matches(lobby, user, policy, matches=1)
        top = standings.both(limit=1)
        assert len(top["skill"]) == 1 and len(top["coins"]) == 1
        assert top["skill"][0]["rank"] == 1 and top["coins"][0]["rank"] == 1

    run(body())


def test_a_player_who_has_never_played_is_on_neither_board():
    async def body():
        lobby, ledger, store, standings = build()
        await play_matches(lobby, ALICE, skilled, matches=1)
        names = {r["username"] for r in standings.both()["skill"]} | {r["username"] for r in standings.both()["coins"]}
        assert names == {"alice"}

    run(body())
