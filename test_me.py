"""Profile data: authentication is required, and users only see their own coins."""

from datetime import timedelta

from sqlalchemy import select

from app.models import TransactionKind, User, UserSession, utcnow
from app.services.wallet import apply_transaction
from helpers import register


# ---------- unauthenticated access ----------

def test_profile_routes_require_login(client):
    assert client.get("/me").status_code == 401
    assert client.get("/me/transactions").status_code == 401


def test_a_made_up_cookie_does_not_authenticate(client):
    client.cookies.set("edge_session", "totally-made-up-token")
    assert client.get("/me").status_code == 401
    assert client.get("/me/transactions").status_code == 401


def test_expired_session_is_rejected(client, db):
    register(client)
    row = db.execute(select(UserSession)).scalar_one()
    row.expires_at = utcnow() - timedelta(minutes=1)
    db.commit()
    assert client.get("/me").status_code == 401


# ---------- balance and history ----------

def test_profile_shows_balance_and_signup_bonus(client):
    register(client, "alice")
    me = client.get("/me").json()
    assert me["username"] == "alice"
    assert me["balance"] == 1000
    history = client.get("/me/transactions").json()
    assert len(history) == 1
    assert history[0]["kind"] == "signup_bonus"
    assert history[0]["amount"] == 1000
    assert history[0]["balance_after"] == 1000
    assert history[0]["created_at"].endswith(("Z", "+00:00"))


def test_history_is_newest_first_and_balance_updates(client, db):
    register(client, "alice")
    user = db.execute(select(User).where(User.username == "alice")).scalar_one()
    apply_transaction(db, user.wallet.id, -300, TransactionKind.BET)
    apply_transaction(db, user.wallet.id, 120, TransactionKind.PAYOUT)
    db.commit()

    assert client.get("/me").json()["balance"] == 820
    history = client.get("/me/transactions").json()
    assert [t["kind"] for t in history] == ["payout", "bet", "signup_bonus"]
    assert [t["balance_after"] for t in history] == [820, 700, 1000]


def test_history_pagination(client, db):
    register(client, "alice")
    user = db.execute(select(User).where(User.username == "alice")).scalar_one()
    for _ in range(4):
        apply_transaction(db, user.wallet.id, 10, TransactionKind.PAYOUT)
    db.commit()

    assert len(client.get("/me/transactions?limit=2").json()) == 2
    assert len(client.get("/me/transactions?limit=100").json()) == 5
    assert len(client.get("/me/transactions?limit=2&offset=4").json()) == 1
    assert client.get("/me/transactions?limit=0").status_code == 422
    assert client.get("/me/transactions?limit=101").status_code == 422
    assert client.get("/me/transactions?offset=-1").status_code == 422


def test_users_only_see_their_own_data(client, db):
    register(client, "alice")
    alice = db.execute(select(User).where(User.username == "alice")).scalar_one()
    apply_transaction(db, alice.wallet.id, -250, TransactionKind.BET)
    db.commit()
    client.post("/auth/logout")

    register(client, "bob")
    me = client.get("/me").json()
    assert me["username"] == "bob"
    assert me["balance"] == 1000
    history = client.get("/me/transactions").json()
    assert len(history) == 1
    assert history[0]["kind"] == "signup_bonus"


# ---------- frontend pages ----------

def test_frontend_pages_are_served(client):
    home = client.get("/")
    assert home.status_code == 200
    assert "Edge Lab" in home.text
    assert client.get("/profile.html").status_code == 200
    assert client.get("/profile.js").status_code == 200


def test_unknown_path_is_404(client):
    assert client.get("/definitely-not-a-page").status_code == 404
