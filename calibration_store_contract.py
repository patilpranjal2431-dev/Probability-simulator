"""The behaviour EVERY calibration store must have (MemoryCalibrationStore and SqlCalibrationStore alike).

Each check receives (store, users): three user ids."""

import pytest

from app.calibration.scoring import Answer
from app.calibration.store import AlreadyAnswered, NoSuchQuestion

SPEC_A = {"kind": "duel_win", "roll": 4}
SPEC_B = {"kind": "poker", "category": "one_pair"}
SPEC_C = {"kind": "coin_run", "flips": 8, "run": 3}


def check_issued_questions_have_unique_ids_and_can_be_read_back(store, users):
    ids = {store.issue(users[0], SPEC_A) for _ in range(5)}
    assert len(ids) == 5
    qid = store.issue(users[0], SPEC_B)
    got = store.get(users[0], qid)
    assert got == {"id": qid, "spec": SPEC_B, "answered": False}  # the spec survives storage exactly
    assert store.get(users[0], "missing") is None


def check_a_question_belongs_to_one_user(store, users):
    qid = store.issue(users[0], SPEC_A)
    assert store.get(users[1], qid) is None
    with pytest.raises(NoSuchQuestion):
        store.record_answer(users[1], qid, 0.5, True, 0.5)  # someone else cannot answer it
    assert store.get(users[0], qid)["answered"] is False  # and nothing changed


def check_pending_is_the_oldest_unanswered_question(store, users):
    assert store.pending(users[0]) is None
    first = store.issue(users[0], SPEC_A)
    second = store.issue(users[0], SPEC_B)
    assert store.pending(users[0])["id"] == first and store.pending(users[0])["spec"] == SPEC_A
    store.record_answer(users[0], first, 0.5, True, 0.5)
    assert store.pending(users[0])["id"] == second
    store.record_answer(users[0], second, 0.5, False, 0.4)
    assert store.pending(users[0]) is None
    assert store.pending(users[1]) is None  # other users are unaffected


def check_a_question_can_be_answered_only_once(store, users):
    qid = store.issue(users[0], SPEC_A)
    store.record_answer(users[0], qid, 0.7, True, 0.5)
    with pytest.raises(AlreadyAnswered):
        store.record_answer(users[0], qid, 0.1, False, 0.5)
    assert store.answers(users[0]) == [Answer("duel_win", 0.7, 0.5, True)]  # the first answer stands
    assert store.get(users[0], qid)["answered"] is True


def check_unknown_questions_are_refused(store, users):
    with pytest.raises(NoSuchQuestion):
        store.record_answer(users[0], "nope", 0.5, True, 0.5)


def check_answers_come_back_in_the_order_they_were_answered(store, users):
    first = store.issue(users[0], SPEC_A)
    second = store.issue(users[0], SPEC_B)
    third = store.issue(users[0], SPEC_C)
    store.record_answer(users[0], third, 0.3, False, 0.35)  # answered out of the order they were issued
    store.record_answer(users[0], first, 0.6, True, 0.5)
    store.record_answer(users[0], second, 0.4, False, 0.42)
    assert [a.kind for a in store.answers(users[0])] == ["coin_run", "duel_win", "poker"]
    assert store.answers(users[0])[0] == Answer("coin_run", 0.3, 0.35, False)


def check_unanswered_questions_are_not_answers_and_users_are_kept_apart(store, users):
    store.issue(users[0], SPEC_A)
    mine = store.issue(users[1], SPEC_B)
    store.record_answer(users[1], mine, 0.2, True, 0.1)
    assert store.answers(users[0]) == [] and len(store.answers(users[1])) == 1 and store.answers(users[2]) == []


def check_stored_numbers_round_trip_exactly(store, users):
    qid = store.issue(users[0], SPEC_A)
    store.record_answer(users[0], qid, 0.123456789, False, 0.4226090436)
    stored = store.answers(users[0])[0]
    assert stored.forecast == pytest.approx(0.123456789) and stored.probability == pytest.approx(0.4226090436)
    assert stored.outcome is False and isinstance(stored.outcome, bool)


ALL_CHECKS = [
    check_issued_questions_have_unique_ids_and_can_be_read_back,
    check_a_question_belongs_to_one_user,
    check_pending_is_the_oldest_unanswered_question,
    check_a_question_can_be_answered_only_once,
    check_unknown_questions_are_refused,
    check_answers_come_back_in_the_order_they_were_answered,
    check_unanswered_questions_are_not_answers_and_users_are_kept_apart,
    check_stored_numbers_round_trip_exactly,
]
