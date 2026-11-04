"""Bankroll simulator and calibration game through FastAPI's TestClient.

The logic behind these endpoints is tested exhaustively elsewhere; these tests check what the web
layer adds: login, input validation and size limits, and the shape of the responses.
"""

import asyncio
import itertools
import random
import xml.etree.ElementTree as ET

import pytest
from sqlalchemy.orm import sessionmaker

from app.bankroll.presets import PRESETS
from app.calibration.service import CalibrationService
from app.calibration.sql_store import SqlCalibrationStore
from app.main import app
from helpers import register as register_user

SVG = "{http://www.w3.org/2000/svg}svg"
GOOD = {"win": 0.55, "lose": 0.45, "odds": 1, "strategy": "kelly", "value": 1, "bets": 60, "paths": 300, "seed": 5}


@pytest.fixture()
def api(engine, client):
    """The app wired to this test's own database, with seeded randomness for the calibration game."""
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    seeds = itertools.count(1)
    app.state.session_factory = factory
    app.state.calibration = CalibrationService(SqlCalibrationStore(factory), lambda: random.Random(next(seeds)))
    app.state.simulation_slots = asyncio.Semaphore(2)
    return client


def sign_up(client, name):
    assert register_user(client, name).status_code == 201
    token = client.cookies.get("edge_session")
    client.cookies.clear()
    return {"cookie": f"edge_session={token}"}


# ====================================================================== login


def test_every_endpoint_requires_login(api):
    assert api.get("/simulations/presets").status_code == 401
    assert api.post("/simulations/bankroll", json=GOOD).status_code == 401
    assert api.get("/calibration/question").status_code == 401
    assert api.post("/calibration/answer", json={"question_id": "x", "forecast": 0.5}).status_code == 401
    assert api.get("/calibration/summary").status_code == 401


def test_the_pages_are_served(api):
    for path in ("/bankroll.html", "/calibration.html", "/bankroll.js", "/calibration.js"):
        assert api.get(path).status_code == 200, path


# ============================================================== bankroll


def test_presets_are_listed(api):
    headers = sign_up(api, "alice")
    got = api.get("/simulations/presets", headers=headers).json()
    assert [p["id"] for p in got] == [p["id"] for p in PRESETS]


def test_a_simulation_returns_numbers_and_three_valid_charts(api):
    headers = sign_up(api, "alice")
    response = api.post("/simulations/bankroll", json=GOOD, headers=headers)
    assert response.status_code == 200
    data = response.json()
    summary = data["summary"]
    assert summary["seed"] == 5 and summary["paths"] == 300 and summary["bets"] == 60
    assert summary["kelly_fraction"] == pytest.approx(0.10) and summary["stake_fraction"] == pytest.approx(0.10)
    assert 0 <= summary["ruin_probability"] <= 1 and summary["ruin_ci"][0] <= summary["ruin_probability"] <= summary["ruin_ci"][1]
    assert data["breakeven_fraction"] == pytest.approx(0.1987, abs=1e-3)
    assert set(data["charts"]) == {"distribution", "paths", "kelly_curve"}
    for svg in data["charts"].values():
        assert ET.fromstring(svg).tag == SVG


def test_the_same_seed_repeats_exactly_and_a_new_seed_does_not(api):
    headers = sign_up(api, "alice")
    first = api.post("/simulations/bankroll", json=GOOD, headers=headers).json()["summary"]
    again = api.post("/simulations/bankroll", json=GOOD, headers=headers).json()["summary"]
    other = api.post("/simulations/bankroll", json={**GOOD, "seed": 6}, headers=headers).json()["summary"]
    assert first == again and first["median_final"] != other["median_final"]


def test_without_a_seed_each_run_is_fresh_and_reports_its_seed(api):
    headers = sign_up(api, "alice")
    body = {k: v for k, v in GOOD.items() if k != "seed"}
    a = api.post("/simulations/bankroll", json=body, headers=headers).json()["summary"]
    b = api.post("/simulations/bankroll", json=body, headers=headers).json()["summary"]
    assert a["seed"] != b["seed"] and isinstance(a["seed"], int)


def test_a_bet_with_no_edge_stakes_nothing_and_a_flat_stake_works(api):
    headers = sign_up(api, "alice")
    fair = api.post("/simulations/bankroll", json={**GOOD, "win": 0.5, "lose": 0.5}, headers=headers).json()["summary"]
    assert fair["stake_fraction"] == 0 and fair["median_final"] == 1000
    flat = api.post("/simulations/bankroll", json={**GOOD, "strategy": "flat", "value": 10, "ruin_fraction": 0}, headers=headers)
    assert flat.status_code == 200 and flat.json()["summary"]["stake_fraction"] is None


def test_staking_everything_returns_valid_json_even_though_its_growth_rate_is_minus_infinity(api):
    headers = sign_up(api, "alice")
    response = api.post("/simulations/bankroll", json={**GOOD, "strategy": "fraction", "value": 1, "bets": 20}, headers=headers)
    assert response.status_code == 200
    assert response.json()["summary"]["theory_growth_rate"] is None  # not "-Infinity", which is not valid JSON


@pytest.mark.parametrize(
    "change",
    [
        {"win": "0.5"}, {"win": True}, {"odds": "1"}, {"start": "1000"}, {"target": "1500"},
        {"win": 1.5}, {"win": -0.1}, {"lose": 2}, {"odds": 0}, {"odds": 5000}, {"strategy": "martingale"}, {"value": 0},
        {"value": -3}, {"bets": 0}, {"bets": 5001}, {"bets": 200.5}, {"bets": "200"}, {"paths": 0}, {"paths": 20001},
        {"start": 0}, {"ruin_fraction": 1}, {"ruin_fraction": -0.2}, {"seed": -1}, {"seed": 1.5}, {"target": -5},
        {"win": 0.7, "lose": 0.5},  # probabilities add up to more than 1
        {"win": 0, "lose": 0},  # a bet that can neither win nor lose
        {"strategy": "kelly", "value": 9},  # more than the supported Kelly multiple
        {"strategy": "fraction", "value": 1.5},
        {"target": 500},  # below the starting bankroll of 1000
    ],
)
def test_nonsense_simulations_are_refused(api, change):
    headers = sign_up(api, "alice")
    assert api.post("/simulations/bankroll", json={**GOOD, **change}, headers=headers).status_code == 422


def test_the_total_amount_of_work_is_capped(api):
    headers = sign_up(api, "alice")
    response = api.post("/simulations/bankroll", json={**GOOD, "bets": 2000, "paths": 3000}, headers=headers)
    assert response.status_code == 422 and "bets x paths" in response.json()["detail"]


def test_not_a_number_in_the_json_is_refused(api):
    headers = {**sign_up(api, "alice"), "content-type": "application/json"}
    body = '{"win": NaN, "lose": 0.45, "bets": 10, "paths": 10}'
    assert api.post("/simulations/bankroll", content=body, headers=headers).status_code == 422


def test_missing_required_fields_are_refused(api):
    headers = sign_up(api, "alice")
    assert api.post("/simulations/bankroll", json={"win": 0.5}, headers=headers).status_code == 422


# ============================================================== calibration


def test_a_full_round_of_the_calibration_game(api):
    headers = sign_up(api, "alice")
    q = api.get("/calibration/question", headers=headers).json()
    assert set(q) == {"question_id", "kind", "text", "answered_so_far"}
    assert api.get("/calibration/question", headers=headers).json()["question_id"] == q["question_id"]  # no rerolling

    r = api.post("/calibration/answer", json={"question_id": q["question_id"], "forecast": 0.3}, headers=headers)
    assert r.status_code == 200
    result = r.json()
    assert result["forecast"] == 0.3 and 0 < result["true_probability"] < 1
    assert result["brier"] == pytest.approx(0.49 if result["event_happened"] else 0.09)  # (0.3 - 1)^2 or (0.3 - 0)^2
    assert result["expected_brier"] == pytest.approx(result["uncertainty"] + result["excess_brier"], abs=1e-3)
    assert result["what_happened"] and result["verdict"] and isinstance(result["event_happened"], bool)

    summary = api.get("/calibration/summary", headers=headers).json()
    assert summary["n"] == 1 and summary["summary"]["n"] == 1 and len(summary["bins"]) == 1
    assert set(summary["charts"]) == {"brier", "reliability"}
    for svg in summary["charts"].values():
        assert ET.fromstring(svg).tag == SVG
    assert api.get("/calibration/question", headers=headers).json()["answered_so_far"] == 1


def test_a_question_can_only_be_answered_once(api):
    headers = sign_up(api, "alice")
    q = api.get("/calibration/question", headers=headers).json()["question_id"]
    assert api.post("/calibration/answer", json={"question_id": q, "forecast": 0.5}, headers=headers).status_code == 200
    again = api.post("/calibration/answer", json={"question_id": q, "forecast": 0.9}, headers=headers)
    assert again.status_code == 409


def test_unknown_questions_and_other_peoples_questions_are_404(api):
    alice, bob = sign_up(api, "alice"), sign_up(api, "bob")
    q = api.get("/calibration/question", headers=alice).json()["question_id"]
    assert api.post("/calibration/answer", json={"question_id": "invented", "forecast": 0.5}, headers=alice).status_code == 404
    assert api.post("/calibration/answer", json={"question_id": q, "forecast": 0.5}, headers=bob).status_code == 404
    assert api.get("/calibration/summary", headers=bob).json()["n"] == 0  # and bob's data is separate


@pytest.mark.parametrize("bad", [-0.1, 1.5, "0.5", None, True, [0.5]])
def test_bad_forecasts_are_refused_and_the_question_stays_open(api, bad):
    headers = sign_up(api, "alice")
    q = api.get("/calibration/question", headers=headers).json()["question_id"]
    response = api.post("/calibration/answer", json={"question_id": q, "forecast": bad}, headers=headers)
    assert response.status_code == 422
    assert api.post("/calibration/answer", json={"question_id": q, "forecast": 0.5}, headers=headers).status_code == 200


def test_a_not_a_number_forecast_is_refused(api):
    headers = {**sign_up(api, "alice"), "content-type": "application/json"}
    q = api.get("/calibration/question", headers=headers).json()["question_id"]
    body = '{"question_id": "%s", "forecast": NaN}' % q
    assert api.post("/calibration/answer", content=body, headers=headers).status_code == 422


def test_missing_fields_and_oversized_ids_are_refused(api):
    headers = sign_up(api, "alice")
    assert api.post("/calibration/answer", json={"forecast": 0.5}, headers=headers).status_code == 422
    assert api.post("/calibration/answer", json={"question_id": "x" * 100, "forecast": 0.5}, headers=headers).status_code == 422


def test_the_summary_is_empty_before_any_answer(api):
    headers = sign_up(api, "alice")
    assert api.get("/calibration/summary", headers=headers).json() == {"n": 0, "summary": {"n": 0}, "bins": [], "series": None, "charts": None}
