"""SqlLedger against the SAME contract as MemoryLedger, plus database-specific audits.

(Needs SQLAlchemy: runs in your normal test environment.)
"""

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

import ledger_contract as contract
from app.models import Game, Transaction
from app.services.wallet import create_user, verify_ledger
from app.tables.sql_ledger import SqlLedger


@pytest.fixture()
def sql(engine):
    """A SqlLedger over a fresh database with two users holding 1,000 coins each."""
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as db:
        a = create_user(db, "alice", "hash")
        b = create_user(db, "bob", "hash")
        db.commit()
        ids = (a.id, b.id)
    return SqlLedger(factory), ids[0], ids[1], factory


@pytest.mark.parametrize("check", contract.ALL_CHECKS, ids=lambda f: f.__name__)
def test_contract(sql, check):
    ledger, a, b, _ = sql
    check(ledger, a, b)


@pytest.mark.parametrize("which", ["b_short", "a_short"])
def test_contract_all_or_nothing(sql, which):
    ledger, a, b, _ = sql
    contract.check_escrow_is_all_or_nothing(ledger, a, b, which)


@pytest.mark.parametrize("bad", [0, -5, 2.5, "10", None, True])
def test_contract_nonsense_amounts(sql, bad):
    ledger, a, b, _ = sql
    contract.check_nonsense_amounts_are_refused(ledger, a, b, bad)


def test_failed_escrow_leaves_no_game_row_behind(sql):
    ledger, a, b, factory = sql
    with pytest.raises(Exception):
        ledger.escrow("dice", {a: 500, b: 1500})
    with factory() as db:
        assert db.execute(select(func.count()).select_from(Game)).scalar_one() == 0
        assert db.execute(select(func.count()).select_from(Transaction).where(Transaction.kind == "bet")).scalar_one() == 0


def test_wallet_ledgers_still_audit_after_play(sql):
    ledger, a, b, factory = sql
    ref = ledger.escrow("dice", {a: 500, b: 500})
    ledger.settle(ref, {a: 650, b: 350})
    with factory() as db:
        from app.models import Wallet

        for user_id in (a, b):
            wallet_id = db.execute(select(Wallet.id).where(Wallet.user_id == user_id)).scalar_one()
            assert verify_ledger(db, wallet_id)


def test_every_game_nets_to_zero_in_the_database(sql):
    ledger, a, b, factory = sql
    ref = ledger.escrow("kuhn", {a: 40, b: 40})
    ledger.settle(ref, {a: 10, b: 70})
    with factory() as db:
        total = db.execute(select(func.sum(Transaction.amount)).where(Transaction.game_id == ref)).scalar_one()
        assert total == 0


def test_transactions_are_tagged_with_their_game(sql):
    ledger, a, b, factory = sql
    ref = ledger.escrow("dice", {a: 100, b: 100})
    ledger.settle(ref, {a: 120, b: 80})
    with factory() as db:
        kinds = sorted(db.execute(select(Transaction.kind).where(Transaction.game_id == ref)).scalars())
        assert kinds == ["bet", "bet", "payout", "payout"]
