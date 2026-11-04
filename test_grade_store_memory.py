"""MemoryGradeStore must satisfy the shared grade-store contract."""

import itertools

import pytest

import grade_store_contract as contract
from app.grading.store import MemoryGradeStore


@pytest.mark.parametrize("check", contract.ALL_CHECKS, ids=lambda f: f.__name__)
def test_contract(check):
    counter = itertools.count(1)
    check(MemoryGradeStore(), (1, 2, 3), lambda: next(counter))
