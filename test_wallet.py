"""Wallet and ledger behaviour."""

import random

import pytest
from sqlalchemy import func, select

from app.config import settings
from app.models import Transaction, TransactionKind
from app.services.wallet import (
    InsufficientFundsError,
    WalletNotFoundError,
    apply_transaction,
    create_user,
    verify_ledger,
)


def ledger_count(db, wallet_id: int) -> int:
    return db.execute(
        select(func.count()).select_from(Transaction).where(Transaction.wallet_id == wallet_id)
    ).scalar_one()


def test_new_user_gets_starting_coins_through_the_ledger(db):
    user = create_user(db, "alice", "hash")
    assert settings.starting_coins == 1000
    assert user.wallet.balance == 1000
    entries = user.wallet.transactions
    assert len(entries) == 1
    assert entries[0].kind == TransactionKind.SIGNUP_BONUS.value
    assert entries[0].amount == 1000
    assert entries[0].balance_after == 1000
    assert verify_ledger(db, user.wallet.id)


def test_bot_flag_is_stored(db):
    bot = create_user(db, "bot_kelly", "!", is_bot=True)
    human = create_user(db, "dave", "hash")
    assert bot.is_bot is True
    assert human.is_bot is False


def test_credit_and_debit_update_balance_and_ledger(db):
    wallet = create_user(db, "alice", "hash").wallet
    apply_transaction(db, wallet.id, -300, TransactionKind.BET)
    entry = apply_transaction(db, wallet.id, 500, TransactionKind.PAYOUT)
    db.refresh(wallet)
    assert wallet.balance == 1200
    assert entry.balance_after == 1200
    assert ledger_count(db, wallet.id) == 3
    assert verify_ledger(db, wallet.id)


def test_spending_everything_is_allowed(db):
    wallet = create_user(db, "alice", "hash").wallet
    apply_transaction(db, wallet.id, -1000, TransactionKind.BET)
    db.refresh(wallet)
    assert wallet.balance == 0


def test_overdraft_is_rejected_and_changes_nothing(db):
    wallet = create_user(db, "alice", "hash").wallet
    with pytest.raises(InsufficientFundsError):
        apply_transaction(db, wallet.id, -1001, TransactionKind.BET)
    db.refresh(wallet)
    assert wallet.balance == 1000
    assert ledger_count(db, wallet.id) == 1  # only the signup bonus
    assert verify_ledger(db, wallet.id)


def test_zero_amount_is_rejected(db):
    wallet = create_user(db, "alice", "hash").wallet
    with pytest.raises(ValueError):
        apply_transaction(db, wallet.id, 0, TransactionKind.BET)


@pytest.mark.parametrize("bad_amount", [1.5, "10", None, True])
def test_non_integer_amounts_are_rejected(db, bad_amount):
    wallet = create_user(db, "alice", "hash").wallet
    with pytest.raises(TypeError):
        apply_transaction(db, wallet.id, bad_amount, TransactionKind.BET)


def test_unknown_wallet_is_reported(db):
    with pytest.raises(WalletNotFoundError):
        apply_transaction(db, 424242, 10, TransactionKind.PAYOUT)
    with pytest.raises(WalletNotFoundError):
        verify_ledger(db, 424242)


def test_ledger_always_matches_balance_after_many_random_moves(db):
    """Seeded random stress test: whatever happens, the audit must pass."""
    rng = random.Random(2026)
    wallet = create_user(db, "alice", "hash").wallet
    for _ in range(300):
        amount = rng.randint(-400, 400) or 1
        kind = TransactionKind.PAYOUT if amount > 0 else TransactionKind.BET
        try:
            apply_transaction(db, wallet.id, amount, kind)
        except InsufficientFundsError:
            pass
        db.refresh(wallet)
        assert wallet.balance >= 0
    assert verify_ledger(db, wallet.id)


def test_audit_detects_tampering(db):
    wallet = create_user(db, "alice", "hash").wallet
    apply_transaction(db, wallet.id, 50, TransactionKind.PAYOUT)
    assert verify_ledger(db, wallet.id)
    wallet.balance += 999  # change the balance WITHOUT a ledger entry
    db.flush()
    assert not verify_ledger(db, wallet.id)
