"""SqlLedger: the production ledger, backed by the database (see ledger.py for the rules)."""

from collections.abc import Callable

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.models import Game, GameStatus, Transaction, TransactionKind, Wallet, utcnow
from app.services.wallet import InsufficientFundsError, apply_transaction

from .ledger import AlreadySettled, InsufficientFunds, SettlementMismatch, UnknownUser, _check_amounts



class SqlLedger:
    """Database ledger. Each method is ONE database transaction, committed once at the end,
    so a failure part-way rolls back everything. Uses the Phase 1 wallet service, whose
    balance update is a single conditional UPDATE (a wallet can never go negative)."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def balance(self, user_id: int) -> int:
        with self._session_factory() as db:
            balance = db.execute(select(Wallet.balance).where(Wallet.user_id == user_id)).scalar_one_or_none()
            if balance is None:
                raise UnknownUser(user_id)
            return balance

    def escrow(self, game_name: str, buy_ins: dict[int, int]) -> int:
        _check_amounts(buy_ins, "buy-in", allow_zero=False)
        with self._session_factory() as db:
            try:
                game = Game(game_type=game_name, status=GameStatus.IN_PROGRESS.value)
                db.add(game)
                db.flush()
                for user_id, amount in sorted(buy_ins.items()):  # fixed order avoids lock-order problems
                    wallet_id = self._wallet_id(db, user_id)
                    try:
                        apply_transaction(db, wallet_id, -amount, TransactionKind.BET, game_id=game.id)
                    except InsufficientFundsError:
                        raise InsufficientFunds(user_id) from None
                db.commit()
                return game.id
            except Exception:
                db.rollback()  # nothing is kept: neither the game row nor any debit
                raise

    def settle(self, game_ref: int, payouts: dict[int, int]) -> dict[int, int]:
        _check_amounts(payouts, "payout", allow_zero=True)
        with self._session_factory() as db:
            try:
                # Claim the game exactly once. A second settle matches 0 rows and is refused.
                claimed = db.execute(
                    update(Game)
                    .where(Game.id == game_ref, Game.status == GameStatus.IN_PROGRESS.value)
                    .values(status=GameStatus.FINISHED.value, finished_at=utcnow()),
                    execution_options={"synchronize_session": False},
                )
                if claimed.rowcount != 1:
                    raise AlreadySettled(f"game {game_ref} is missing or already settled")

                escrowed = self._escrowed(db, game_ref)
                if set(payouts) != set(escrowed) or sum(payouts.values()) != sum(escrowed.values()):
                    raise SettlementMismatch("payouts must go to the escrowing players and equal the escrow")

                for user_id, amount in sorted(payouts.items()):
                    if amount > 0:
                        apply_transaction(
                            db, self._wallet_id(db, user_id), amount, TransactionKind.PAYOUT, game_id=game_ref
                        )
                balances = {user_id: self._balance(db, user_id) for user_id in payouts}
                db.commit()
                return balances
            except Exception:
                db.rollback()
                raise

    def refund_unfinished(self) -> int:
        with self._session_factory() as db:
            try:
                stuck = db.execute(
                    select(Game.id).where(Game.status == GameStatus.IN_PROGRESS.value)
                ).scalars().all()
                for game_id in stuck:
                    for user_id, amount in sorted(self._escrowed(db, game_id).items()):
                        apply_transaction(
                            db, self._wallet_id(db, user_id), amount, TransactionKind.REFUND, game_id=game_id
                        )
                    db.execute(
                        update(Game).where(Game.id == game_id).values(
                            status=GameStatus.ABORTED.value, finished_at=utcnow()
                        ),
                        execution_options={"synchronize_session": False},
                    )
                db.commit()
                return len(stuck)
            except Exception:
                db.rollback()
                raise

    def game_status(self, game_ref: int) -> str:
        with self._session_factory() as db:
            return db.execute(select(Game.status).where(Game.id == game_ref)).scalar_one()

    def net_coin_rows(self) -> dict[int, int]:
        """Coins won (+) or lost (-) at tables, per user, counting only matches that are over.

        A match still in progress is ignored: its buy-in is escrowed, not lost."""
        kinds = [TransactionKind.BET.value, TransactionKind.PAYOUT.value, TransactionKind.REFUND.value]
        with self._session_factory() as db:
            rows = db.execute(
                select(Wallet.user_id, func.sum(Transaction.amount))
                .select_from(Transaction)
                .join(Wallet, Wallet.id == Transaction.wallet_id)
                .join(Game, Game.id == Transaction.game_id)
                .where(Transaction.kind.in_(kinds), Game.status != GameStatus.IN_PROGRESS.value)
                .group_by(Wallet.user_id)
            ).all()
        return {user_id: int(total) for user_id, total in rows}

    # ---- helpers
    @staticmethod
    def _wallet_id(db: Session, user_id: int) -> int:
        wallet_id = db.execute(select(Wallet.id).where(Wallet.user_id == user_id)).scalar_one_or_none()
        if wallet_id is None:
            raise UnknownUser(user_id)
        return wallet_id

    @staticmethod
    def _balance(db: Session, user_id: int) -> int:
        return db.execute(select(Wallet.balance).where(Wallet.user_id == user_id)).scalar_one()

    @staticmethod
    def _escrowed(db: Session, game_id: int) -> dict[int, int]:
        """What each user put into this game, read back from the ledger (not trusted from memory)."""
        rows = db.execute(
            select(Wallet.user_id, func.sum(Transaction.amount))
            .select_from(Transaction)  # say which side to join FROM: the select names two tables
            .join(Wallet, Wallet.id == Transaction.wallet_id)
            .where(Transaction.game_id == game_id, Transaction.kind == TransactionKind.BET.value)
            .group_by(Wallet.user_id)
        ).all()
        # int(): on PostgreSQL SUM() of integers comes back as a Decimal
        return {user_id: int(-total) for user_id, total in rows}
