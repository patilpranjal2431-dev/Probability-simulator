"""ORM models.

Money rule: coins are stored as whole integers (never floats), and every change
to a wallet is also written to the `transactions` table (see services/wallet.py),
so a balance can always be audited from its history.
"""

from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def utcnow() -> datetime:
    """Current time in UTC (timezone-aware)."""
    return datetime.now(timezone.utc)


class TransactionKind(str, Enum):
    """Why a wallet balance changed."""

    SIGNUP_BONUS = "signup_bonus"
    BET = "bet"
    PAYOUT = "payout"
    REFUND = "refund"


class GameStatus(str, Enum):
    """Lifecycle of a game."""

    WAITING = "waiting"
    IN_PROGRESS = "in_progress"
    FINISHED = "finished"
    ABORTED = "aborted"  # refunded after a server restart


class User(Base):
    """A human player or a bot. Bots cannot log in (their hash is a non-hash marker)."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    is_bot: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    wallet: Mapped["Wallet"] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan"
    )


class Wallet(Base):
    """A user's virtual coin balance. The database refuses negative balances."""

    __tablename__ = "wallets"
    __table_args__ = (CheckConstraint("balance >= 0", name="ck_wallet_balance_non_negative"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), unique=True
    )
    balance: Mapped[int] = mapped_column(Integer, default=0)

    user: Mapped["User"] = relationship(back_populates="wallet")
    transactions: Mapped[list["Transaction"]] = relationship(
        back_populates="wallet", order_by="Transaction.id", cascade="all, delete-orphan"
    )


class Game(Base):
    """One table/session of a game between players and/or bots."""

    __tablename__ = "games"

    id: Mapped[int] = mapped_column(primary_key=True)
    game_type: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), default=GameStatus.WAITING.value)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    hands: Mapped[list["Hand"]] = relationship(
        back_populates="game", order_by="Hand.hand_number", cascade="all, delete-orphan"
    )


class Hand(Base):
    """One round inside a game. Later phases add decisions and EV grading to this."""

    __tablename__ = "hands"
    __table_args__ = (UniqueConstraint("game_id", "hand_number", name="uq_hand_game_number"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id", ondelete="CASCADE"))
    hand_number: Mapped[int] = mapped_column(Integer)
    pot: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    game: Mapped["Game"] = relationship(back_populates="hands")


class Transaction(Base):
    """Append-only ledger entry. `amount` is signed; `balance_after` is a snapshot."""

    __tablename__ = "transactions"
    __table_args__ = (CheckConstraint("amount != 0", name="ck_transaction_amount_non_zero"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    wallet_id: Mapped[int] = mapped_column(
        ForeignKey("wallets.id", ondelete="CASCADE"), index=True
    )
    amount: Mapped[int] = mapped_column(Integer)
    balance_after: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(32))
    game_id: Mapped[int | None] = mapped_column(
        ForeignKey("games.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    wallet: Mapped["Wallet"] = relationship(back_populates="transactions")


class UserSession(Base):
    """A login session. Only a SHA-256 hash of the cookie token is stored, so a
    leaked database cannot be used to hijack sessions."""

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class GradedDecisionRecord(Base):
    """One graded decision of one human player. See app/grading for how the numbers are defined."""

    __tablename__ = "graded_decisions"
    __table_args__ = (
        UniqueConstraint("game_id", "user_id", "hand_number", "decision_index", name="uq_graded_decision"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id", ondelete="CASCADE"), index=True)
    hand_number: Mapped[int] = mapped_column(Integer)
    decision_index: Mapped[int] = mapped_column(Integer)
    seat: Mapped[int] = mapped_column(Integer)
    decision_kind: Mapped[str] = mapped_column(String(10))
    own_value: Mapped[int] = mapped_column(Integer)
    action_kind: Mapped[str] = mapped_column(String(8))
    action_amount: Mapped[int] = mapped_column(Integer)
    best_kind: Mapped[str] = mapped_column(String(8))
    best_amount: Mapped[int] = mapped_column(Integer)
    ev_taken: Mapped[float] = mapped_column(Float)
    ev_best: Mapped[float] = mapped_column(Float)
    regret: Mapped[float] = mapped_column(Float)
    hand_net: Mapped[int] = mapped_column(Integer)
    counts_for_ev: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class CalibrationQuestionRecord(Base):
    """A calibration question issued to a user, and (once answered) their forecast and the result."""

    __tablename__ = "calibration_questions"

    id: Mapped[int] = mapped_column(primary_key=True)
    token: Mapped[str] = mapped_column(String(24), unique=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    spec_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    answered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    forecast: Mapped[float | None] = mapped_column(Float, nullable=True)
    outcome: Mapped[bool | None] = mapped_column(nullable=True)
    probability: Mapped[float | None] = mapped_column(Float, nullable=True)
