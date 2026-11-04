"""SqlGradeStore against the SAME contract as MemoryGradeStore (needs SQLAlchemy)."""

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

import grade_store_contract as contract
from app.grading.sql_store import SqlGradeStore
from app.models import Game, GradedDecisionRecord
from app.services.wallet import create_user


@pytest.fixture()
def sql(engine):
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as db:
        users = [create_user(db, name, "hash").id for name in ("alice", "bob", "carol")]
        db.commit()

    def new_game() -> int:
        with factory() as db:
            game = Game(game_type="dice", status="finished")
            db.add(game)
            db.commit()
            return game.id

    return SqlGradeStore(factory), tuple(users), new_game, factory


@pytest.mark.parametrize("check", contract.ALL_CHECKS, ids=lambda f: f.__name__)
def test_contract(sql, check):
    store, users, new_game, _ = sql
    check(store, users, new_game)


def test_the_same_decision_cannot_be_stored_twice_even_by_a_buggy_caller(sql):
    store, users, new_game, factory = sql
    game = new_game()
    store.save(game, users[0], [contract.row(hand=1)])
    with factory() as db:
        before = db.execute(select(func.count()).select_from(GradedDecisionRecord)).scalar_one()
    store.save(game, users[0], [contract.row(hand=1)])  # replace, not add
    with factory() as db:
        assert db.execute(select(func.count()).select_from(GradedDecisionRecord)).scalar_one() == before


def test_a_failed_save_leaves_the_old_grades_in_place(sql):
    store, users, new_game, _ = sql
    game = new_game()
    store.save(game, users[0], [contract.row(hand=1)])
    with pytest.raises(Exception):
        store.save(game, users[0], [contract.row(hand=1), contract.row(hand=1)])  # violates the unique key
    assert len(store.decisions_for(users[0])) == 1  # the earlier, valid grades survive
