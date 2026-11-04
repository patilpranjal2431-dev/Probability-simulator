"""The async table runtime, driven by simulated clients and an in-memory ledger.

Covers: lobby rules, escrow and settlement, hidden information, illegal moves, timeouts,
disconnects and reconnects, leaving, settlement failures, and concurrency.
"""

import asyncio
import json
import random

import pytest

from app.tables.common import LobbyError, TableConfig, UserRef
from app.tables.ledger import LedgerError, MemoryLedger
from helpers_tables import (
    ALICE, BOB, BOT_USER_ID, CAROL, START, SimClient, act, make_lobby, passive_action,
    play_passively, run, wait_for,
)

TOTAL = sum(START.values())


def total(ledger):
    return sum(ledger.balance(uid) for uid in START)


async def open_human_table(lobby, **kwargs):
    """Alice creates a dice table and connects; Bob connects and the match starts."""
    table = await lobby.create_table(ALICE, "dice", 500)
    a, b = SimClient("alice"), SimClient("bob")
    await lobby.connect(ALICE, table.id, a)
    await lobby.connect(BOB, table.id, b)
    return table, a, b


async def open_bot_table(lobby, bot="kelly", game="dice", buy_in=500):
    table = await lobby.create_table(ALICE, game, buy_in, bot=bot)
    a = SimClient("alice")
    await lobby.connect(ALICE, table.id, a)
    return table, a


# ============================================================ the happy paths


def test_a_match_against_a_bot_escrows_plays_settles_and_closes():
    async def body():
        lobby, ledger = make_lobby()
        table, a = await open_bot_table(lobby)
        assert (ledger.balance(1), ledger.balance(BOT_USER_ID)) == (500, 500)  # both buy-ins escrowed
        await play_passively(table, [(ALICE, a)])

        over = a.match_over
        assert over["reason"] == "max_hands" and over["hands"] == 3
        assert over["your_payout"] + over["final_stacks"][1] == 1000
        assert over["your_balance"] == ledger.balance(1) == 500 + over["your_payout"]
        assert total(ledger) == TOTAL  # no coin created or destroyed
        assert a.closed == 1000 and table.state == "closed"
        assert lobby.get(table.id) is None and lobby.table_of(1) is None  # cleaned up

    run(body())


def test_two_humans_play_a_full_match():
    async def body():
        lobby, ledger = make_lobby()
        table = await lobby.create_table(ALICE, "dice", 500)
        a, b = SimClient(), SimClient()
        await lobby.connect(ALICE, table.id, a)
        assert a.of_type("waiting") and [t["id"] for t in lobby.list_open()] == [table.id]
        await lobby.connect(BOB, table.id, b)
        assert lobby.list_open() == []  # full tables are not listed
        assert (a.state["you"], b.state["you"]) == (0, 1)
        assert a.state["opponent"]["name"] == "bob" and b.state["opponent"]["name"] == "alice"
        assert (ledger.balance(1), ledger.balance(2)) == (500, 500)

        await play_passively(table, [(ALICE, a), (BOB, b)])
        assert a.match_over and b.match_over
        assert a.match_over["final_stacks"] == b.match_over["final_stacks"]
        assert a.match_over["your_payout"] + b.match_over["your_payout"] == 1000
        assert a.match_over["net_change"] == -b.match_over["net_change"]
        assert (ledger.balance(1), ledger.balance(2)) == (a.match_over["your_balance"], b.match_over["your_balance"])
        assert total(ledger) == TOTAL

    run(body())


def test_kuhn_tables_work_too():
    async def body():
        lobby, ledger = make_lobby()
        table, a = await open_bot_table(lobby, bot="cautious", game="kuhn", buy_in=50)
        await play_passively(table, [(ALICE, a)])
        assert a.match_over and total(ledger) == TOTAL

    run(body())


def test_a_user_can_play_again_after_a_match():
    async def body():
        lobby, ledger = make_lobby()
        for _ in range(2):
            table, a = await open_bot_table(lobby)
            await play_passively(table, [(ALICE, a)])
        assert total(ledger) == TOTAL

    run(body())


def test_bets_come_from_the_table_stack_not_the_wallet():
    async def body():
        lobby, ledger = make_lobby(max_hands=1)
        table, a = await open_bot_table(lobby)
        assert a.state["decision"]["max_bet"] == 490  # 500 buy-in minus the 10 ante
        assert ledger.balance(1) == 500  # the other 500 stay safely in the wallet
        await act(table, ALICE, a, "bet", 10_000)  # more than the table stack: refused
        assert a.errors() == ["illegal_action"]
        assert ledger.balance(1) == 500

    run(body())


# ==================================================== hidden information


def test_no_message_ever_contains_the_opponents_hidden_value():
    async def body():
        lobby, _ = make_lobby(max_hands=6)
        table, a, b = await open_human_table(lobby)
        await play_passively(table, [(ALICE, a), (BOB, b)])
        forbidden = {"private_values", "_private", "opponent_value", "your_opponents_value"}
        public_keys = {"game", "stage", "done", "to_act", "pot", "invested", "stacks", "history"}

        def keys(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    yield k
                    yield from keys(v)
            elif isinstance(node, list):
                for item in node:
                    yield from keys(item)

        for client in (a, b):
            for message in client.inbox:
                assert not forbidden & set(keys(message))
            for state in client.of_type("state"):
                assert set(state["hand"]) <= public_keys | {"ended_by", "net", "showdown_values"}
            for result in client.of_type("hand_result"):
                if result["ended_by"] == "fold":
                    assert result["showdown_values"] is None

    run(body())


def test_each_player_is_only_told_their_own_value():
    async def body():
        lobby, _ = make_lobby()
        table, a, b = await open_human_table(lobby)
        assert a.state["your_value"] == table.core.hand.private_view(0)
        assert b.state["your_value"] == table.core.hand.private_view(1)

    run(body())


# ====================================================== lobby rules


def test_one_table_per_user():
    async def body():
        lobby, _ = make_lobby()
        await lobby.create_table(ALICE, "dice", 500)
        with pytest.raises(LobbyError) as info:
            await lobby.create_table(ALICE, "kuhn", 50)
        assert info.value.code == "already_seated"

    run(body())


def test_cannot_join_a_second_table_while_seated():
    async def body():
        lobby, _ = make_lobby()
        t1 = await lobby.create_table(ALICE, "dice", 500)
        await lobby.connect(ALICE, t1.id, SimClient())
        t2 = await lobby.create_table(BOB, "dice", 500)
        await lobby.connect(BOB, t2.id, SimClient())
        with pytest.raises(LobbyError) as info:
            await lobby.connect(ALICE, t2.id, SimClient())  # Alice is already at t1
        assert info.value.code == "already_seated"
        assert lobby.table_of(2).id == t2.id and lobby.table_of(1).id == t1.id

    run(body())


def test_a_full_table_turns_away_a_third_player_and_frees_them_to_play_elsewhere():
    async def body():
        lobby, _ = make_lobby()
        table, a, b = await open_human_table(lobby)
        with pytest.raises(LobbyError) as info:
            await lobby.connect(CAROL, table.id, SimClient())
        assert info.value.code == "table_full"
        assert lobby.table_of(3) is None  # her reservation was rolled back
        await lobby.create_table(CAROL, "dice", 500)  # so she can open her own

    run(body())


def test_a_bot_table_is_never_joinable_by_a_human():
    async def body():
        lobby, _ = make_lobby()
        table, a = await open_bot_table(lobby)
        with pytest.raises(LobbyError) as info:
            await lobby.connect(BOB, table.id, SimClient())
        assert info.value.code == "table_full"

    run(body())


def test_a_guest_cannot_join_before_the_host_connects():
    async def body():
        lobby, _ = make_lobby()
        table = await lobby.create_table(ALICE, "dice", 500)
        with pytest.raises(LobbyError) as info:
            await lobby.connect(BOB, table.id, SimClient())
        assert info.value.code == "host_not_connected"
        assert lobby.table_of(2) is None

    run(body())


@pytest.mark.parametrize(
    ("game", "buy_in", "bot", "code"),
    [
        ("poker", 500, None, "unknown_game"),
        ("dice", 99, None, "bad_buy_in"),
        ("dice", 1001, None, "bad_buy_in"),
        ("dice", 500.5, None, "bad_buy_in"),
        ("dice", True, None, "bad_buy_in"),
        ("kuhn", 9, None, "bad_buy_in"),
        ("dice", 500, "genius", "unknown_bot"),
    ],
)
def test_bad_table_requests_are_refused(game, buy_in, bot, code):
    async def body():
        lobby, _ = make_lobby()
        with pytest.raises(LobbyError) as info:
            await lobby.create_table(ALICE, game, buy_in, bot=bot)
        assert info.value.code == code
        assert lobby.table_of(1) is None

    run(body())


def test_unknown_table_and_poor_creator_are_refused():
    async def body():
        lobby, _ = make_lobby(balances={1: 100, 2: 1000, 99: 1000})
        with pytest.raises(LobbyError) as info:
            await lobby.connect(ALICE, "nope", SimClient())
        assert info.value.code == "no_such_table"
        with pytest.raises(LobbyError) as info:
            await lobby.create_table(ALICE, "dice", 500)
        assert info.value.code == "insufficient_funds"

    run(body())


def test_default_buy_in_is_the_games_standard_stack():
    async def body():
        lobby, _ = make_lobby()
        table = await lobby.create_table(ALICE, "dice")
        assert table.buy_in == 1000

    run(body())


# ============================================================ escrow failures


def test_a_guest_who_cannot_cover_the_buy_in_is_refused_and_nothing_is_charged():
    async def body():
        lobby, ledger = make_lobby(balances={1: 1000, 2: 50, 3: 1000, BOT_USER_ID: 1000})
        table = await lobby.create_table(ALICE, "dice", 500)
        a, poor = SimClient(), SimClient()
        await lobby.connect(ALICE, table.id, a)
        with pytest.raises(LobbyError) as info:
            await lobby.connect(BOB, table.id, poor)
        assert info.value.code == "insufficient_funds"
        assert poor.errors() == ["insufficient_funds"]
        assert (ledger.balance(1), ledger.balance(2)) == (1000, 50)  # the host was not charged either
        assert lobby.table_of(2) is None and table.state == "waiting"
        assert [t["id"] for t in lobby.list_open()] == [table.id]

        rich = SimClient()
        await lobby.connect(CAROL, table.id, rich)  # the table is still open for someone else
        assert table.state == "playing" and ledger.balance(3) == 500

    run(body())


def test_a_host_who_lost_coins_meanwhile_cancels_the_table_and_the_guest_is_told_why():
    async def body():
        lobby, ledger = make_lobby()
        table = await lobby.create_table(ALICE, "dice", 500)
        host, guest = SimClient(), SimClient()
        await lobby.connect(ALICE, table.id, host)
        ledger._balances[1] = 100  # Alice's balance drops while she waits
        with pytest.raises(LobbyError) as info:
            await lobby.connect(BOB, table.id, guest)
        assert info.value.code == "host_unavailable"  # NOT "you cannot cover": Bob is fine
        assert host.errors() == ["insufficient_funds"]
        assert ledger.balance(2) == 1000 and ledger.balance(1) == 100  # nobody was charged
        assert table.state == "closed" and lobby.get(table.id) is None
        assert lobby.table_of(1) is None and lobby.table_of(2) is None

    run(body())


def test_a_host_who_cannot_cover_a_bot_match_is_told_at_connect():
    async def body():
        lobby, ledger = make_lobby()
        table = await lobby.create_table(ALICE, "dice", 500, bot="kelly")
        ledger._balances[1] = 100
        host = SimClient()
        with pytest.raises(LobbyError) as info:
            await lobby.connect(ALICE, table.id, host)
        assert info.value.code == "insufficient_funds" and table.state == "closed"
        assert ledger.balance(BOT_USER_ID) == 1000

    run(body())


def test_a_bankrupt_bot_cannot_start_a_match():
    async def body():
        lobby, ledger = make_lobby(balances={1: 1000, BOT_USER_ID: 10})
        table = await lobby.create_table(ALICE, "dice", 500, bot="kelly")
        a = SimClient()
        with pytest.raises(LobbyError) as info:
            await lobby.connect(ALICE, table.id, a)
        assert info.value.code == "bot_unavailable"
        assert ledger.balance(1) == 1000 and table.state == "closed"

    run(body())


def test_two_simultaneous_guests_only_one_gets_the_seat_and_only_one_escrow_happens():
    async def body():
        lobby, ledger = make_lobby()
        table = await lobby.create_table(ALICE, "dice", 500)
        await lobby.connect(ALICE, table.id, SimClient())
        results = await asyncio.gather(
            lobby.connect(BOB, table.id, SimClient()),
            lobby.connect(CAROL, table.id, SimClient()),
            return_exceptions=True,
        )
        assert sum(isinstance(r, LobbyError) for r in results) == 1
        assert sum(ledger.balance(u) == 500 for u in (1, 2, 3)) == 2  # host + exactly one guest
        assert sum(ledger.balance(u) == 1000 for u in (2, 3)) == 1

    run(body())


# ===================================================== waiting-room timeouts


def test_an_unfilled_table_expires():
    async def body():
        lobby, ledger = make_lobby(waiting_timeout=0.05)
        table = await lobby.create_table(ALICE, "dice", 500)
        a = SimClient()
        await lobby.connect(ALICE, table.id, a)
        await wait_for(lambda: table.state == "closed", what="the table to expire")
        assert a.errors() == ["table_expired"] and a.closed == 1000
        assert lobby.get(table.id) is None and lobby.table_of(1) is None
        assert ledger.balance(1) == 1000  # never charged
        await lobby.create_table(ALICE, "dice", 500)  # free to try again

    run(body())


def test_a_host_who_leaves_before_anyone_joins_cancels_the_table():
    async def body():
        lobby, ledger = make_lobby()
        table = await lobby.create_table(ALICE, "dice", 500)
        a = SimClient()
        await lobby.connect(ALICE, table.id, a)
        await table.disconnect(1, a)
        assert table.state == "closed" and lobby.list_open() == [] and lobby.table_of(1) is None
        assert ledger.balance(1) == 1000

    run(body())


# ============================================================= illegal moves


def test_moves_out_of_turn_or_illegal_are_refused_and_the_match_goes_on():
    async def body():
        lobby, _ = make_lobby(max_strikes=50)
        table, a, b = await open_human_table(lobby)
        seq = a.state["seq"]
        await act(table, BOB, b, "check")  # not Bob's turn
        await act(table, ALICE, a, "bet", 491)  # more than the stack allows
        await act(table, ALICE, a, "call")  # nothing to call
        await act(table, ALICE, a, "fold")
        assert b.errors() == ["not_your_turn"]
        assert a.errors() == ["illegal_action"] * 3
        assert a.state["seq"] == seq and table.state == "playing"  # nothing changed
        await play_passively(table, [(ALICE, a), (BOB, b)])
        assert a.match_over  # and the match still finished normally

    run(body())


def test_malformed_messages_get_an_error_not_a_crash():
    async def body():
        lobby, _ = make_lobby(max_strikes=50)
        table, a = await open_bot_table(lobby)
        for raw in ("not json", "[]", '{"type":"cheat"}', '{"type":"action"}', '{"type":"action","action":"bet","amount":1.5}',
                    json.dumps({"type": "ping", "pad": "x" * 5000}), b"\xff\xfe"):
            await table.handle(1, raw)
        assert a.errors() == ["bad_json", "bad_message", "unknown_type", "bad_action", "bad_action",
                              "message_too_large", "bad_json"]
        assert table.state == "playing"
        await table.handle(1, '{"type":"ping"}')
        assert a.of_type("pong")

    run(body())


def test_a_client_that_keeps_misbehaving_is_disconnected():
    async def body():
        lobby, _ = make_lobby(max_strikes=3, reconnect_grace=30.0)
        table, a, b = await open_human_table(lobby)
        for _ in range(3):
            await table.handle(1, "garbage")
        assert a.closed == 1008
        assert b.events("opponent_disconnected")
        assert table.state == "playing"  # Bob is still waiting; Alice has the grace period

    run(body())


# ================================================================ timeouts


def test_a_player_who_does_not_act_is_auto_played():
    async def body():
        lobby, ledger = make_lobby(turn_timeout=0.05, max_consecutive_timeouts=99, max_hands=2)
        table, a = await open_bot_table(lobby)
        await wait_for(lambda: a.of_type("auto_action"), what="an automatic action")
        first = a.of_type("auto_action")[0]
        assert first["reason"] == "timeout" and first["action"] in ("check", "fold") and first["seat"] == 0

    run(body())


def test_repeated_timeouts_end_the_match_and_settle_it():
    async def body():
        lobby, ledger = make_lobby(turn_timeout=0.03, max_consecutive_timeouts=3, max_hands=50)
        table, a, b = await open_human_table(lobby)
        # Alice never acts; Bob plays along
        async def bob():
            while table.state == "playing":
                if b.my_turn():
                    await act(table, BOB, b, passive_action(b))
                await asyncio.sleep(0.005)

        task = asyncio.create_task(bob())
        await wait_for(lambda: a.match_over, what="match over by timeouts")
        task.cancel()
        assert a.match_over["reason"] == "timeouts" and a.match_over["conceded_by"] == 0
        assert total(ledger) == TOTAL and table.state == "closed"
        assert b.match_over["your_balance"] == ledger.balance(2)

    run(body())


def test_acting_resets_the_timeout_counter():
    async def body():
        lobby, _ = make_lobby(turn_timeout=0.05, max_consecutive_timeouts=2, max_hands=50)
        table, a = await open_bot_table(lobby)
        await wait_for(lambda: len(a.of_type("auto_action")) >= 1, what="first timeout")
        assert table._timeouts[1] == 1
        await wait_for(lambda: a.my_turn(), what="my turn")
        await act(table, ALICE, a, passive_action(a))
        assert table._timeouts[1] == 0  # a real move wipes the streak

    run(body())


def test_the_very_first_state_tells_the_player_how_long_they_have():
    async def body():
        lobby, _ = make_lobby(turn_timeout=10.0)
        table, a = await open_bot_table(lobby)
        first = a.of_type("state")[0]
        assert first["decision"] is not None and 9.0 < first["deadline_s"] <= 10.0
        await table.close()

    run(body())


def test_the_player_not_on_the_clock_sees_no_deadline():
    async def body():
        lobby, _ = make_lobby(turn_timeout=10.0)
        table, a, b = await open_human_table(lobby)
        assert a.state["deadline_s"] is not None and b.state["deadline_s"] is None
        await table.close()

    run(body())


def test_reconnecting_cannot_reset_your_own_clock():
    """Otherwise a player could stall forever by dropping and reconnecting before each timeout."""
    async def body():
        lobby, _ = make_lobby(turn_timeout=10.0, reconnect_grace=30.0)
        table, a = await open_bot_table(lobby)
        before = table._turn_deadline
        await asyncio.sleep(0.05)
        for _ in range(3):
            await table.disconnect(1, a)
            a = SimClient()
            await lobby.connect(ALICE, table.id, a)
        assert table._turn_deadline == before  # same deadline as when the turn began
        assert a.state["deadline_s"] < 10.0 - 0.04  # the time already spent is not refunded
        await table.close()

    run(body())


def test_the_clock_keeps_running_while_you_are_disconnected():
    """Come back after your deadline has passed and you are timed out at once."""
    async def body():
        lobby, _ = make_lobby(turn_timeout=0.1, reconnect_grace=30.0, max_consecutive_timeouts=99, max_hands=50)
        table, a = await open_bot_table(lobby)
        await table.disconnect(1, a)
        await asyncio.sleep(0.2)  # longer than the turn timeout
        assert a.of_type("auto_action") == []  # nobody is waiting, so nothing happened yet
        a2 = SimClient()
        await lobby.connect(ALICE, table.id, a2)
        await wait_for(lambda: a2.of_type("auto_action"), what="immediate timeout on return")
        assert a2.of_type("auto_action")[0]["reason"] == "timeout"
        await table.close()

    run(body())


def test_acting_in_time_means_no_automatic_action():
    async def body():
        lobby, _ = make_lobby(turn_timeout=5.0, max_hands=2)
        table, a = await open_bot_table(lobby)
        await play_passively(table, [(ALICE, a)])
        assert a.of_type("auto_action") == [] and a.match_over

    run(body())


# =========================================================== disconnects


def test_a_disconnect_mid_turn_is_auto_played_so_the_opponent_is_not_stuck():
    async def body():
        lobby, _ = make_lobby(reconnect_grace=30.0, max_hands=50)
        table, a, b = await open_human_table(lobby)
        assert a.my_turn()
        await table.disconnect(1, a)  # Alice drops on her own turn
        assert b.events("opponent_disconnected")[0]["grace_s"] == 30.0
        auto = b.of_type("auto_action")
        assert auto and auto[0]["reason"] == "disconnected" and auto[0]["seat"] == 0
        assert table.state == "playing"
        assert b.my_turn()  # the table moved on to Bob
        await table.close()

    run(body())


def test_reconnecting_within_the_grace_period_resumes_the_match():
    async def body():
        lobby, ledger = make_lobby(reconnect_grace=30.0)
        table, a, b = await open_human_table(lobby)
        await table.disconnect(1, a)
        a2 = SimClient("alice again")
        await lobby.connect(ALICE, table.id, a2)  # same user, new connection
        assert a2.state["you"] == 0 and a2.state["opponent"]["connected"]
        assert b.events("opponent_reconnected")
        assert not table._grace_tasks.get(1)  # the abandonment timer was cancelled
        await play_passively(table, [(ALICE, a2), (BOB, b)])
        assert a2.match_over and b.match_over and total(ledger) == TOTAL

    run(body())


def test_staying_away_past_the_grace_period_abandons_the_match():
    async def body():
        lobby, ledger = make_lobby(reconnect_grace=0.05, max_hands=50)
        table, a, b = await open_human_table(lobby)
        await table.disconnect(2, b)
        await wait_for(lambda: a.match_over, what="match abandoned")
        over = a.match_over
        assert over["reason"] == "disconnected" and over["conceded_by"] == 1
        assert total(ledger) == TOTAL and table.state == "closed"
        assert ledger.balance(1) == over["your_balance"]
        assert a.closed == 1000

    run(body())


def test_a_disconnect_during_the_opponents_turn_waits_until_it_is_their_turn():
    async def body():
        lobby, _ = make_lobby(reconnect_grace=30.0, max_hands=50)
        table, a, b = await open_human_table(lobby)
        await table.disconnect(2, b)  # Bob drops, but it is Alice's turn
        assert b.of_type("auto_action") == [] and a.of_type("auto_action") == []
        await act(table, ALICE, a, "check")  # Alice moves; now it will be Bob's turn at some point
        await wait_for(lambda: a.my_turn() or a.of_type("auto_action"), what="progress")
        assert all(m["reason"] == "disconnected" and m["seat"] == 1 for m in a.of_type("auto_action"))
        await table.close()

    run(body())


def test_a_dead_socket_is_treated_as_a_disconnect():
    async def body():
        lobby, _ = make_lobby(reconnect_grace=30.0, max_hands=50)
        table, a, b = await open_human_table(lobby)
        b.fail_sends = True  # the network under Bob dies
        await act(table, ALICE, a, "check")
        assert 2 not in table.conns and a.events("opponent_disconnected")
        assert table.state == "playing"
        await table.close()

    run(body())


def test_both_players_gone_means_nobody_is_auto_played_forever_and_the_match_is_abandoned():
    async def body():
        lobby, ledger = make_lobby(reconnect_grace=0.05, max_hands=50)
        table, a, b = await open_human_table(lobby)
        await table.disconnect(1, a)
        await table.disconnect(2, b)
        await wait_for(lambda: table.state == "closed", what="abandonment")
        assert table.close_reason == "disconnected"
        assert total(ledger) == TOTAL
        assert table.core.hands_played <= 2  # no runaway auto-play

    run(body())


def test_disconnecting_from_a_bot_table_pauses_and_then_abandons():
    async def body():
        lobby, ledger = make_lobby(reconnect_grace=0.05, max_hands=50)
        table, a = await open_bot_table(lobby)
        await table.disconnect(1, a)
        assert table.core.hands_played == 0 and a.of_type("auto_action") == []  # no one is waiting
        await wait_for(lambda: table.state == "closed", what="abandonment")
        assert total(ledger) == TOTAL and table.close_reason == "disconnected"

    run(body())


def test_a_new_connection_replaces_the_old_one_and_the_stale_close_is_ignored():
    async def body():
        lobby, _ = make_lobby(reconnect_grace=30.0)
        table, a, b = await open_human_table(lobby)
        a2 = SimClient()
        await lobby.connect(ALICE, table.id, a2)
        assert a.closed == 4000  # the old socket was told to go away
        await table.disconnect(1, a)  # ...and its late "disconnected" notice changes nothing
        assert table.conns[1] is a2 and not b.events("opponent_disconnected")
        await table.close()

    run(body())


def test_a_stranger_cannot_hijack_a_seat_by_reconnecting():
    async def body():
        lobby, _ = make_lobby()
        table, a, b = await open_human_table(lobby)
        with pytest.raises(LobbyError) as info:
            await lobby.connect(CAROL, table.id, SimClient())
        assert info.value.code == "table_full" and table.conns[1] is a

    run(body())


# ================================================================= leaving


def test_leaving_ends_the_match_and_costs_the_chips_in_the_current_hand():
    async def body():
        lobby, ledger = make_lobby(max_hands=50)
        table, a, b = await open_human_table(lobby)
        await act(table, ALICE, a, "bet", 100)  # Alice has 110 in the pot
        await table.handle(1, '{"type":"leave"}')
        for client in (a, b):
            assert client.match_over["reason"] == "left" and client.match_over["conceded_by"] == 0
        assert ledger.balance(1) == 500 + (500 - 110) and ledger.balance(2) == 500 + (500 + 110)
        assert total(ledger) == TOTAL and table.state == "closed"

    run(body())


# =========================================================== settlement safety


class FlakyLedger(MemoryLedger):
    """Fails the first `failures` settle() calls (or all of them if failures is None)."""

    def __init__(self, balances, failures):
        super().__init__(balances)
        self.failures = failures
        self.attempts = 0

    def settle(self, game_ref, payouts):
        self.attempts += 1
        if self.failures is None or self.attempts <= self.failures:
            raise LedgerError("database is down")
        return super().settle(game_ref, payouts)


def test_a_temporary_settlement_failure_is_retried():
    async def body():
        ledger = FlakyLedger(dict(START), failures=2)
        lobby, _ = make_lobby(ledger=ledger, settle_retries=3)
        table, a = await open_bot_table(lobby)
        await play_passively(table, [(ALICE, a)])
        assert ledger.attempts == 3 and a.match_over and total(ledger) == TOTAL

    run(body())


def test_a_permanent_settlement_failure_keeps_the_escrow_and_a_restart_refunds_it():
    async def body():
        ledger = FlakyLedger(dict(START), failures=None)
        lobby, _ = make_lobby(ledger=ledger, settle_retries=2)
        table, a = await open_bot_table(lobby)
        await play_passively(table, [(ALICE, a)])
        assert a.errors() == ["settlement_failed"] and a.match_over is None
        assert table.close_reason == "settlement_failed" and a.closed == 1000
        assert ledger.balance(1) == 500  # coins are held, not lost
        assert lobby.table_of(1) is None  # the user is not trapped

        assert ledger.refund_unfinished() == 1  # what the next server start does
        assert total(ledger) == TOTAL and ledger.balance(1) == 1000

    run(body())


def test_a_settled_match_cannot_be_paid_twice_even_if_asked_again():
    async def body():
        lobby, ledger = make_lobby(max_hands=1)
        table, a = await open_bot_table(lobby)
        await play_passively(table, [(ALICE, a)])
        after = total(ledger)
        with pytest.raises(Exception):
            ledger.settle(table.escrow_ref, {1: 1000, BOT_USER_ID: 0})
        assert total(ledger) == after

    run(body())


# ================================================================ property


def test_coins_are_conserved_across_many_random_matches():
    async def body():
        rng = random.Random(2026)
        lobby, ledger = make_lobby(max_hands=8)
        played = 0
        for round_number in range(40):
            game = rng.choice(["dice", "kuhn"])
            low, high = (100, 500) if game == "dice" else (10, 50)
            human_vs_human = rng.random() < 0.5
            bot = None if human_vs_human else rng.choice(["random", "ev", "kelly", "cautious"])
            involved = [1, 2] if human_vs_human else [1, BOT_USER_ID]
            affordable = min(ledger.balance(uid) for uid in involved)
            if affordable < low:
                continue  # somebody is too poor for this table; skip the round
            buy_in = min(high, affordable)
            table = await lobby.create_table(ALICE, game, buy_in, bot=bot)
            a = SimClient()
            await lobby.connect(ALICE, table.id, a)
            clients = [(ALICE, a)]
            if human_vs_human:
                b = SimClient()
                await lobby.connect(BOB, table.id, b)
                clients.append((BOB, b))
            for _ in range(2000):
                if table.state != "playing":
                    break
                for user, client in clients:
                    if table.state == "playing" and client.my_turn():
                        d = client.state["decision"]
                        if d["kind"] == "bet" and "bet" in d["actions"] and rng.random() < 0.6:
                            await act(table, user, client, "bet", rng.randint(d["min_bet"], d["max_bet"]))
                        elif d["kind"] == "respond":
                            await act(table, user, client, rng.choice(["call", "fold"]))
                        else:
                            await act(table, user, client, "check")
                await asyncio.sleep(0)
                if rng.random() < 0.02 and table.state == "playing":  # occasionally someone leaves
                    await table.handle(1, '{"type":"leave"}')
            assert table.state == "closed", f"round {round_number} did not finish"
            assert total(ledger) == TOTAL, f"coins changed in round {round_number}"
            played += 1
        per_game = {}
        for _user, amount, _kind, ref, _after in ledger.transactions:
            per_game[ref] = per_game.get(ref, 0) + amount
        assert played >= 25 and len(per_game) == played
        assert all(v == 0 for v in per_game.values())  # every single match nets to zero

    run(body())
