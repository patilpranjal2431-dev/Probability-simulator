"""Register, login, logout: duplicate usernames, wrong passwords, session handling."""

from sqlalchemy import select

from app.models import User, UserSession
from app.services.wallet import create_user
from helpers import register


# ---------- register ----------

def test_register_creates_account_and_logs_in(client):
    response = register(client, "alice")
    assert response.status_code == 201
    assert response.json()["username"] == "alice"
    assert "edge_session" in client.cookies
    me = client.get("/me")
    assert me.status_code == 200
    assert me.json()["balance"] == 1000


def test_password_is_stored_hashed_never_plain(client, db):
    register(client, "alice", "correct-horse-9")
    user = db.execute(select(User).where(User.username == "alice")).scalar_one()
    assert user.password_hash != "correct-horse-9"
    assert user.password_hash.startswith("$argon2id$")


def test_session_cookie_is_httponly_and_samesite(client):
    header = register(client, "alice").headers["set-cookie"].lower()
    assert "httponly" in header
    assert "samesite=lax" in header


def test_duplicate_username_is_rejected(client):
    assert register(client, "alice").status_code == 201
    second = register(client, "alice", "a-different-pass-1")
    assert second.status_code == 409
    assert second.json()["detail"] == "Username already taken"


def test_duplicate_username_ignores_letter_case(client):
    assert register(client, "alice").status_code == 201
    assert register(client, "ALICE").status_code == 409
    assert register(client, "  Alice  ").status_code == 409


def test_duplicate_registration_does_not_overwrite_the_original_password(client):
    register(client, "alice", "original-pass-1")
    register(client, "alice", "attacker-pass-1")
    client.post("/auth/logout")
    assert client.post("/auth/login", json={"username": "alice", "password": "attacker-pass-1"}).status_code == 401
    assert client.post("/auth/login", json={"username": "alice", "password": "original-pass-1"}).status_code == 200


def test_register_rejects_bad_input(client):
    assert register(client, "alice", "short7!").status_code == 422          # 7 characters
    assert register(client, "alice", "x" * 129).status_code == 422          # too long
    assert register(client, "ab", "correct-horse-9").status_code == 422     # username too short
    assert register(client, "a" * 33, "correct-horse-9").status_code == 422
    assert register(client, "bad name!", "correct-horse-9").status_code == 422
    assert register(client, "<script>", "correct-horse-9").status_code == 422
    assert register(client, "bot_kelly", "correct-horse-9").status_code == 422  # reserved for bots
    assert client.post("/auth/register", json={"username": "alice"}).status_code == 422


# ---------- login ----------

def test_login_with_correct_password(client):
    register(client, "alice", "correct-horse-9")
    client.post("/auth/logout")
    response = client.post("/auth/login", json={"username": "Alice", "password": "correct-horse-9"})
    assert response.status_code == 200
    assert response.json()["username"] == "alice"
    assert client.get("/me").status_code == 200


def test_login_with_wrong_password_is_rejected_without_a_session(client, db):
    register(client, "alice", "correct-horse-9")
    client.post("/auth/logout")
    response = client.post("/auth/login", json={"username": "alice", "password": "wrong-password"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid username or password"
    assert "set-cookie" not in response.headers
    assert client.get("/me").status_code == 401


def test_unknown_user_gets_the_same_error_as_a_wrong_password(client):
    register(client, "alice", "correct-horse-9")
    client.post("/auth/logout")
    wrong_pw = client.post("/auth/login", json={"username": "alice", "password": "nope-nope-1"})
    no_user = client.post("/auth/login", json={"username": "nobody", "password": "nope-nope-1"})
    assert wrong_pw.status_code == no_user.status_code == 401
    assert wrong_pw.json() == no_user.json()


def test_empty_credentials_are_rejected(client):
    assert client.post("/auth/login", json={"username": "", "password": ""}).status_code == 422


def test_bot_accounts_cannot_log_in(client, db):
    create_user(db, "bot_kelly", "!", is_bot=True)
    db.commit()
    for password in ("!", "anything-at-all"):
        response = client.post("/auth/login", json={"username": "bot_kelly", "password": password})
        assert response.status_code == 401


def test_each_login_gets_a_new_session_token(client):
    register(client, "alice", "correct-horse-9")
    first = client.cookies.get("edge_session")
    client.post("/auth/login", json={"username": "alice", "password": "correct-horse-9"})
    assert client.cookies.get("edge_session") != first


# ---------- logout ----------

def test_logout_ends_the_session(client, db):
    register(client)
    assert client.get("/me").status_code == 200
    assert client.post("/auth/logout").status_code == 204
    assert client.get("/me").status_code == 401
    assert db.execute(select(UserSession)).first() is None


def test_logout_revokes_the_token_on_the_server_not_just_the_cookie(client):
    register(client)
    stolen_token = client.cookies.get("edge_session")
    client.post("/auth/logout")
    client.cookies.set("edge_session", stolen_token)  # replay the old token
    assert client.get("/me").status_code == 401


def test_logout_when_not_logged_in_is_harmless(client):
    assert client.post("/auth/logout").status_code == 204
