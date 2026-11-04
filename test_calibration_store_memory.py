"""MemoryCalibrationStore must satisfy the shared calibration-store contract."""

import threading

import pytest

import calibration_store_contract as contract
from app.calibration.store import AlreadyAnswered, MemoryCalibrationStore


@pytest.mark.parametrize("check", contract.ALL_CHECKS, ids=lambda f: f.__name__)
def test_contract(check):
    check(MemoryCalibrationStore(), (1, 2, 3))


def test_simultaneous_answers_to_one_question_count_exactly_once():
    store = MemoryCalibrationStore()
    qid = store.issue(1, contract.SPEC_A)
    results = []

    def attempt(forecast):
        try:
            store.record_answer(1, qid, forecast, True, 0.5)
            results.append("recorded")
        except AlreadyAnswered:
            results.append("refused")

    threads = [threading.Thread(target=attempt, args=(i / 20,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count("recorded") == 1 and results.count("refused") == 19
    assert len(store.answers(1)) == 1
