"""REST + WebSocket glue, through FastAPI's TestClient.

The table logic itself is tested exhaustively in test_tables_*.py with simulated clients;
these tests only check what the web layer adds: login, origin checks, validation, and one
complete match over a real WebSocket.
"""

import itertools
import random

import pytest
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker
from starlette.websockets import WebSocketDisconnect

from app.grading.service import Standings
from app.grading.sql_store import SqlGradeStore
from app.main import app
from app.models import Wallet
from app.services.bots import ensure_bot_users, make_directory
from app.tables.common import TableConfig
from app.tables.sql_ledger import SqlLedger
from app.tables.wiring import build_lobby
from helpers import register as register_user


@pytest.fixture()
def play(engine, client):
    """The app wired to this test's own database, with a tiny match length and seeded dice."""
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as db:
        ensure_bot_users(db)
    seeds = itertools.count(1)
    ledger, grade_store = SqlLedger(factory), SqlGradeStore(factory)
    app.state.session_factory = factory
    app.state.grade_store = grade_store
    app.state.standings = Standings(grade_store, ledger, make_directory(factory))
    app.state.lobby = build_lobby(
        factory,
        TableConfig(max_hands=3, turn_timeout=30.0, reconnect_grace=30.0, waiting_timeout=30.0),
        lambda: random.Random(next(seeds)),
        ledger=ledger,
        grade_store=grade_store,
    )
    return client


def sign_up(client, name):
    """Register a user and return their session token, leaving the client logged out."""
    assert register_user(client, name).status_code == 201
    token = client.cookies.get("edge_session")
    client.cookies.clear()
    return token


def auth(token):
    return {"cookie": f"edge_session={token}"}


def play_to_the_end(ws):
    """Check/call until the server says the match is over."""
    while True:
        message = ws.receive_json()
        if message["type"] == "match_over":
            return message
        if message["type"] == "state" and message["decision"]:
            kind = message["decision"]["kind"]
            ws.send_json({"type": "action", "action": "check" if kind == "bet" else "call"})


# ------------------------------------------------------------------------ REST


def test_lobby_endpoints_require_login(play):
    assert play.get("/tables").status_code == 401
    assert play.post("/tables", json={"game": "dice"}).status_code == 401
    assert play.get("/tables/mine").status_code == 401


def test_bad_table_requests_are_refused_with_a_machine_readable_code(play):
    headers = auth(sign_up(play, "alice"))
    bad_game = play.post("/tables", json={"game": "poker"}, headers=headers)
    assert bad_game.status_code == 400 and bad_game.json()["detail"]["code"] == "unknown_game"
    bad_bot = play.post("/tables", json={"game": "dice", "bot": "genius"}, headers=headers)
    assert bad_bot.json()["detail"]["code"] == "unknown_bot"
    bad_buy_in = play.post("/tables", json={"game": "dice", "buy_in": 5}, headers=headers)
    assert bad_buy_in.json()["detail"]["code"] == "bad_buy_in"
    for wrong_type in ("500", 500.0, True):  # strict integer parsing
        assert play.post("/tables", json={"game": "dice", "buy_in": wrong_type}, headers=headers).status_code == 422


def test_one_table_per_user_over_http(play):
    headers = auth(sign_up(play, "alice"))
    first = play.post("/tables", json={"game": "dice", "buy_in": 100, "bot": "kelly"}, headers=headers)
    assert first.status_code == 201 and first.json()["vs_bot"] == "kelly"
    second = play.post("/tables", json={"game": "kuhn"}, headers=headers)
    assert second.status_code == 409 and second.json()["detail"]["code"] == "already_seated"
    assert play.get("/tables/mine", headers=headers).json()["table"]["id"] == first.json()["id"]


# ------------------------------------------------------------------- WebSocket


def test_websocket_refuses_users_who_are_not_logged_in(play):
    with pytest.raises(WebSocketDisconnect):
        with play.websocket_connect("/ws/tables/anything"):
            pass
    with pytest.raises(WebSocketDisconnect):
        with play.websocket_connect("/ws/tables/anything", headers=auth("made-up-token")):
            pass


def test_websocket_refuses_pages_from_other_websites(play):
    headers = {**auth(sign_up(play, "alice")), "origin": "https://evil.example"}
    with pytest.raises(WebSocketDisconnect):
        with play.websocket_connect("/ws/tables/anything", headers=headers):
            pass


def test_same_site_origin_is_accepted(play):
    headers = auth(sign_up(play, "alice"))
    table_id = play.post("/tables", json={"game": "dice", "buy_in": 100, "bot": "kelly"}, headers=headers).json()["id"]
    with play.websocket_connect(f"/ws/tables/{table_id}", headers={**headers, "origin": "http://testserver"}) as ws:
        assert ws.receive_json()["type"] == "state"


def test_connecting_to_an_unknown_table_returns_an_error_message(play):
    headers = auth(sign_up(play, "alice"))
    with play.websocket_connect("/ws/tables/does-not-exist", headers=headers) as ws:
        message = ws.receive_json()
        assert message["type"] == "error" and message["code"] == "no_such_table"


def test_illegal_and_malformed_messages_over_a_real_websocket(play):
    headers = auth(sign_up(play, "alice"))
    table_id = play.post("/tables", json={"game": "dice", "buy_in": 100, "bot": "cautious"}, headers=headers).json()["id"]
    with play.websocket_connect(f"/ws/tables/{table_id}", headers=headers) as ws:
        first = ws.receive_json()
        assert first["type"] == "state" and first["you"] == 0 and first["decision"]["actions"] == ["check", "bet"]
        ws.send_json({"type": "action", "action": "bet", "amount": 999_999})
        assert ws.receive_json()["code"] == "illegal_action"
        ws.send_text("not json")
        assert ws.receive_json()["code"] == "bad_json"
        ws.send_json({"type": "ping"})
        assert ws.receive_json()["type"] == "pong"


def test_a_full_match_against_a_bot_settles_the_wallets(play, engine):
    token = sign_up(play, "alice")
    headers = auth(token)
    table_id = play.post("/tables", json={"game": "dice", "buy_in": 100, "bot": "kelly"}, headers=headers).json()["id"]
    with play.websocket_connect(f"/ws/tables/{table_id}", headers=headers) as ws:
        over = play_to_the_end(ws)

    assert over["reason"] == "max_hands" and over["hands"] == 3
    me = play.get("/me", headers=headers).json()
    assert me["balance"] == over["your_balance"]  # the wallet really was updated
    assert me["balance"] == 1000 - 100 + over["your_payout"]
    history = play.get("/me/transactions", headers=headers).json()
    assert [t["kind"] for t in history][:2] == ["payout", "bet"] or over["your_payout"] == 0

    factory = sessionmaker(bind=engine)
    with factory() as db:
        total = db.execute(select(Wallet.balance)).scalars().all()
    # every coin in the system (humans + the four bots, 1000 each) is still there
    assert sum(total) == 5 * 1000
    assert play.get("/tables/mine", headers=headers).json() == {"table": None}  # free to play again


def test_a_human_cannot_register_a_bot_name(client):
    response = register_user(client, "bot_kelly")
    assert response.status_code == 422


# ------------------------------------------------------------------- grading


def test_grading_endpoints_require_login(play):
    assert play.get("/leaderboards").status_code == 401
    assert play.get("/me/decisions").status_code == 401


def test_leaderboard_query_parameters_are_validated(play):
    headers = auth(sign_up(play, "alice"))
    assert play.get("/leaderboards?min_hands=0", headers=headers).status_code == 422
    assert play.get("/leaderboards?limit=101", headers=headers).status_code == 422
    assert play.get("/me/decisions?limit=0", headers=headers).status_code == 422


def test_the_pages_are_served(play):
    for path in ("/leaderboards.html", "/skill-vs-luck.html", "/leaderboards.js"):
        assert play.get(path).status_code == 200, path
    assert "Skill vs luck" in play.get("/skill-vs-luck.html").text


def test_a_finished_match_produces_graded_decisions_and_both_leaderboard_rows(play):
    headers = auth(sign_up(play, "alice"))
    table_id = play.post("/tables", json={"game": "dice", "buy_in": 100, "bot": "kelly"}, headers=headers).json()["id"]
    with play.websocket_connect(f"/ws/tables/{table_id}", headers=headers) as ws:
        over = play_to_the_end(ws)

    assert over["grading"] is not None and over["grading"]["decisions"] >= 1
    decisions = play.get("/me/decisions", headers=headers).json()
    assert len(decisions) == over["grading"]["decisions"]
    assert all(d["regret"] >= 0 and d["ev_best"] >= d["ev_taken"] - 0.01 for d in decisions)
    assert {"action_kind", "best_kind", "ev_taken", "ev_best", "regret", "hand_net"} <= set(decisions[0])

    boards = play.get("/leaderboards", headers=headers).json()
    assert [r["username"] for r in boards["skill"]] == ["alice"]
    assert [r["username"] for r in boards["coins"]] == ["alice"]  # bots are on neither board
    skill = boards["skill"][0]
    assert skill["regret"] == pytest.approx(over["grading"]["regret"], abs=0.01)
    assert skill["coin_rank"] == 1 and boards["coins"][0]["skill_rank"] == 1
    assert boards["coins"][0]["net_coins"] == over["net_change"]
