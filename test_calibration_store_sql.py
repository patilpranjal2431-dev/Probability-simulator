"""SqlCalibrationStore against the SAME contract as MemoryCalibrationStore (needs SQLAlchemy)."""

import pytest
from sqlalchemy.orm import sessionmaker

import calibration_store_contract as contract
from app.calibration.sql_store import SqlCalibrationStore
from app.services.wallet import create_user


@pytest.fixture()
def sql(engine):
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as db:
        users = tuple(create_user(db, name, "hash").id for name in ("alice", "bob", "carol"))
        db.commit()
    return SqlCalibrationStore(factory), users


@pytest.mark.parametrize("check", contract.ALL_CHECKS, ids=lambda f: f.__name__)
def test_contract(sql, check):
    store, users = sql
    check(store, users)
