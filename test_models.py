"""Database rules: uniqueness, non-negative balances, foreign keys."""

import pytest
from sqlalchemy.exc import IntegrityError

from app.models import Game, Hand, Transaction, User, Wallet
from app.services.wallet import create_user


def test_username_must_be_unique(db):
    create_user(db, "alice", "hash")
    with pytest.raises(IntegrityError):
        create_user(db, "alice", "hash")
    db.rollback()


def test_database_rejects_negative_balance(db):
    user = User(username="bob", password_hash="hash")
    db.add(user)
    db.flush()
    db.add(Wallet(user_id=user.id, balance=-1))
    with pytest.raises(IntegrityError):
        db.flush()
    db.rollback()


def test_database_rejects_zero_amount_transaction(db):
    user = create_user(db, "carol", "hash")
    db.add(Transaction(wallet_id=user.wallet.id, amount=0, balance_after=1000, kind="bet"))
    with pytest.raises(IntegrityError):
        db.flush()
    db.rollback()


def test_foreign_keys_are_enforced(db):
    db.add(Transaction(wallet_id=9999, amount=5, balance_after=5, kind="bet"))
    with pytest.raises(IntegrityError):
        db.flush()
    db.rollback()


def test_hand_numbers_are_unique_within_a_game(db):
    game = Game(game_type="dice")
    db.add(game)
    db.flush()
    db.add(Hand(game_id=game.id, hand_number=1))
    db.flush()
    db.add(Hand(game_id=game.id, hand_number=1))
    with pytest.raises(IntegrityError):
        db.flush()
    db.rollback()


def test_new_game_starts_waiting(db):
    game = Game(game_type="dice")
    db.add(game)
    db.flush()
    assert game.status == "waiting"
    assert game.finished_at is None
