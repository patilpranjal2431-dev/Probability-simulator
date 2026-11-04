"""Session service: tokens are random, stored hashed, and can expire or be revoked."""

from datetime import timedelta

from sqlalchemy import select

from app.models import UserSession, utcnow
from app.services.sessions import (
    create_session,
    delete_expired_sessions,
    delete_session,
    get_user_for_token,
    hash_token,
)
from app.services.wallet import create_user


def test_raw_token_is_never_stored(db):
    user = create_user(db, "alice", "hash")
    token = create_session(db, user.id)
    rows = db.execute(select(UserSession)).scalars().all()
    assert len(rows) == 1
    assert rows[0].token_hash == hash_token(token)
    assert rows[0].token_hash != token


def test_tokens_are_unique_and_long(db):
    user = create_user(db, "alice", "hash")
    tokens = {create_session(db, user.id) for _ in range(20)}
    assert len(tokens) == 20
    assert all(len(t) >= 40 for t in tokens)


def test_valid_token_finds_its_user_and_unknown_token_does_not(db):
    user = create_user(db, "alice", "hash")
    token = create_session(db, user.id)
    assert get_user_for_token(db, token).id == user.id
    assert get_user_for_token(db, "not-a-token") is None


def test_expired_session_is_rejected_and_can_be_purged(db):
    user = create_user(db, "alice", "hash")
    token = create_session(db, user.id)
    row = db.execute(select(UserSession)).scalar_one()
    row.expires_at = utcnow() - timedelta(seconds=1)
    db.flush()
    assert get_user_for_token(db, token) is None
    assert delete_expired_sessions(db) == 1
    assert db.execute(select(UserSession)).first() is None


def test_deleting_a_session_revokes_it(db):
    user = create_user(db, "alice", "hash")
    token = create_session(db, user.id)
    delete_session(db, token)
    assert get_user_for_token(db, token) is None
