"""MemoryLedger must satisfy the shared ledger contract, plus its own thread-safety checks."""

import threading

import pytest

from app.tables.ledger import InsufficientFunds, MemoryLedger
import ledger_contract as contract


def fresh():
    return MemoryLedger({1: 1000, 2: 1000}), 1, 2


@pytest.mark.parametrize("check", contract.ALL_CHECKS, ids=lambda f: f.__name__)
def test_contract(check):
    ledger, a, b = fresh()
    check(ledger, a, b)


@pytest.mark.parametrize("which", ["b_short", "a_short"])
def test_contract_all_or_nothing(which):
    ledger, a, b = fresh()
    contract.check_escrow_is_all_or_nothing(ledger, a, b, which)


@pytest.mark.parametrize("bad", [0, -5, 2.5, "10", None, True])
def test_contract_nonsense_amounts(bad):
    ledger, a, b = fresh()
    contract.check_nonsense_amounts_are_refused(ledger, a, b, bad)


def test_every_game_nets_to_zero_in_the_transaction_log():
    ledger, a, b = fresh()
    ref = ledger.escrow("dice", {a: 400, b: 400})
    ledger.settle(ref, {a: 100, b: 700})
    per_game = {}
    for _user, amount, _kind, game, _after in ledger.transactions:
        per_game[game] = per_game.get(game, 0) + amount
    assert per_game == {ref: 0}


def test_concurrent_escrows_can_never_overspend_a_wallet():
    """20 threads each try to escrow 300 from a 1,000-coin wallet: exactly 3 can succeed."""
    ledger = MemoryLedger({1: 1000, 2: 100_000})
    outcomes = []

    def attempt():
        try:
            ledger.escrow("dice", {1: 300, 2: 300})
            outcomes.append("ok")
        except InsufficientFunds:
            outcomes.append("refused")

    threads = [threading.Thread(target=attempt) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert outcomes.count("ok") == 3 and outcomes.count("refused") == 17
    assert ledger.balance(1) == 100 and ledger.balance(2) == 100_000 - 900


def test_concurrent_settles_pay_exactly_once():
    ledger = MemoryLedger({1: 1000, 2: 1000})
    ref = ledger.escrow("dice", {1: 500, 2: 500})
    results = []

    def attempt():
        try:
            ledger.settle(ref, {1: 900, 2: 100})
            results.append("paid")
        except Exception as error:  # noqa: BLE001
            results.append(type(error).__name__)

    threads = [threading.Thread(target=attempt) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count("paid") == 1 and results.count("AlreadySettled") == 9
    assert (ledger.balance(1), ledger.balance(2)) == (1400, 600)
