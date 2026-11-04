"""Wallet and ledger logic.

Design:
* Every coin movement goes through `apply_transaction`, which updates the balance
  AND appends a ledger row in the same database transaction.
* The balance change is ONE conditional SQL UPDATE ("balance + amount >= 0"), so two
  concurrent requests can never both spend the same coins.
* These functions only `flush`. The caller decides when to `commit`, so several
  wallet changes (e.g. a bet and a payout) can succeed or fail together.
"""

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Transaction, TransactionKind, User, Wallet


class WalletError(Exception):
    """Base class for wallet problems."""


class InsufficientFundsError(WalletError):
    """The wallet does not have enough coins for this debit."""


class WalletNotFoundError(WalletError):
    """No wallet exists with the given id."""


def apply_transaction(
    db: Session,
    wallet_id: int,
    amount: int,
    kind: TransactionKind,
    game_id: int | None = None,
) -> Transaction:
    """Add (amount > 0) or remove (amount < 0) coins and record it in the ledger.

    Raises:
        TypeError: amount is not a whole number.
        ValueError: amount is zero.
        WalletNotFoundError: the wallet does not exist.
        InsufficientFundsError: the debit would make the balance negative.
    """
    if isinstance(amount, bool) or not isinstance(amount, int):
        raise TypeError("amount must be a whole number of coins")
    if amount == 0:
        raise ValueError("amount must not be zero")

    result = db.execute(
        update(Wallet)
        .where(Wallet.id == wallet_id, Wallet.balance + amount >= 0)
        .values(balance=Wallet.balance + amount),
        execution_options={"synchronize_session": False},
    )
    if result.rowcount != 1:
        exists = db.execute(select(Wallet.id).where(Wallet.id == wallet_id)).scalar_one_or_none()
        if exists is None:
            raise WalletNotFoundError(f"wallet {wallet_id} does not exist")
        raise InsufficientFundsError(f"wallet {wallet_id} cannot cover {amount} coins")

    wallet = db.get(Wallet, wallet_id)
    db.refresh(wallet)  # the UPDATE bypassed the ORM, so reload the new balance
    entry = Transaction(
        wallet_id=wallet_id,
        amount=amount,
        balance_after=wallet.balance,
        kind=kind.value,
        game_id=game_id,
    )
    db.add(entry)
    db.flush()
    return entry


def create_user(
    db: Session,
    username: str,
    password_hash: str,
    *,
    is_bot: bool = False,
) -> User:
    """Create a user and a wallet, then credit the starting coins through the ledger.

    The password must already be hashed (Phase 2 adds hashing and validation).
    A duplicate username raises sqlalchemy.exc.IntegrityError.
    """
    user = User(username=username, password_hash=password_hash, is_bot=is_bot)
    user.wallet = Wallet(balance=0)
    db.add(user)
    db.flush()
    apply_transaction(db, user.wallet.id, settings.starting_coins, TransactionKind.SIGNUP_BONUS)
    return user


def verify_ledger(db: Session, wallet_id: int) -> bool:
    """Audit a wallet: replaying its ledger must reproduce every snapshot and the balance."""
    wallet = db.get(Wallet, wallet_id)
    if wallet is None:
        raise WalletNotFoundError(f"wallet {wallet_id} does not exist")
    db.refresh(wallet)

    entries = db.execute(
        select(Transaction).where(Transaction.wallet_id == wallet_id).order_by(Transaction.id)
    ).scalars()

    running = 0
    for entry in entries:
        running += entry.amount
        if entry.balance_after != running:
            return False
    return running == wallet.balance
