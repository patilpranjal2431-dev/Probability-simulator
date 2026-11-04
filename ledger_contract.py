"""The behaviour EVERY ledger must have. Run against MemoryLedger and SqlLedger alike.

Each check receives a fresh ledger in which users `a` and `b` each hold 1,000 coins.
"""

import pytest

from app.tables.ledger import (
    AlreadySettled, InsufficientFunds, LedgerError, SettlementMismatch, UnknownUser,
)


def balances(ledger, a, b):
    return ledger.balance(a), ledger.balance(b)


def check_escrow_debits_every_player(ledger, a, b):
    ref = ledger.escrow("dice", {a: 300, b: 200})
    assert balances(ledger, a, b) == (700, 800)
    assert ledger.game_status(ref) == "in_progress"


def check_escrow_is_all_or_nothing(ledger, a, b, which="b_short"):
    request = {a: 500, b: 1500} if which == "b_short" else {a: 1500, b: 500}
    with pytest.raises(InsufficientFunds) as info:
        ledger.escrow("dice", request)
    assert info.value.user_id == (b if which == "b_short" else a)
    assert balances(ledger, a, b) == (1000, 1000)  # the player who COULD pay was not charged


def check_settle_pays_out_and_conserves_coins(ledger, a, b):
    ref = ledger.escrow("dice", {a: 500, b: 500})
    result = ledger.settle(ref, {a: 700, b: 300})
    assert result == {a: 1200, b: 800}
    assert balances(ledger, a, b) == (1200, 800)
    assert sum(balances(ledger, a, b)) == 2000
    assert ledger.game_status(ref) == "finished"


def check_a_match_settles_only_once(ledger, a, b):
    ref = ledger.escrow("dice", {a: 500, b: 500})
    ledger.settle(ref, {a: 600, b: 400})
    with pytest.raises(AlreadySettled):
        ledger.settle(ref, {a: 600, b: 400})
    assert balances(ledger, a, b) == (1100, 900)  # not paid twice


def check_bad_payouts_are_refused_and_change_nothing(ledger, a, b):
    ref = ledger.escrow("dice", {a: 500, b: 500})
    bad_attempts = [
        {a: 600, b: 401},  # creates a coin
        {a: 600, b: 399},  # destroys a coin
        {a: 1000},  # leaves out a player
        {a: 500, b: 300, 424242: 200},  # pays someone who never escrowed
    ]
    for payouts in bad_attempts:
        with pytest.raises(SettlementMismatch):
            ledger.settle(ref, payouts)
    assert balances(ledger, a, b) == (500, 500)
    assert ledger.game_status(ref) == "in_progress"
    assert ledger.settle(ref, {a: 450, b: 550}) == {a: 950, b: 1050}  # a correct settle still works


def check_a_loser_can_be_paid_nothing(ledger, a, b):
    ref = ledger.escrow("dice", {a: 500, b: 500})
    ledger.settle(ref, {a: 1000, b: 0})
    assert balances(ledger, a, b) == (1500, 500)


def check_nonsense_amounts_are_refused(ledger, a, b, bad=0):
    with pytest.raises(LedgerError):
        ledger.escrow("dice", {a: bad, b: 100})
    with pytest.raises(LedgerError):
        ledger.escrow("dice", {})
    ref = ledger.escrow("dice", {a: 100, b: 100})
    with pytest.raises(LedgerError):
        ledger.settle(ref, {a: -1 if bad == 0 and bad is not False else bad, b: 100})
    assert balances(ledger, a, b) == (900, 900)


def check_unknown_users_are_refused_without_side_effects(ledger, a, b):
    with pytest.raises(UnknownUser):
        ledger.escrow("dice", {a: 10, 424242: 10})
    assert balances(ledger, a, b) == (1000, 1000)
    with pytest.raises(UnknownUser):
        ledger.balance(424242)


def check_unknown_game_cannot_be_settled(ledger, a, b):
    with pytest.raises(LedgerError):
        ledger.settle(987654, {a: 1, b: 1})


def check_refund_returns_the_escrow_of_unfinished_matches(ledger, a, b):
    stuck1 = ledger.escrow("dice", {a: 300, b: 200})
    stuck2 = ledger.escrow("kuhn", {a: 50, b: 50})
    done = ledger.escrow("dice", {a: 100, b: 100})
    ledger.settle(done, {a: 150, b: 50})
    assert balances(ledger, a, b) == (1000 - 300 - 50 - 100 + 150, 1000 - 200 - 50 - 100 + 50)

    assert ledger.refund_unfinished() == 2
    assert balances(ledger, a, b) == (1050, 950)  # only the finished match changed anything
    assert ledger.game_status(stuck1) == ledger.game_status(stuck2) == "aborted"
    assert ledger.game_status(done) == "finished"
    with pytest.raises(AlreadySettled):
        ledger.settle(stuck1, {a: 300, b: 200})  # cannot be paid after being refunded
    assert ledger.refund_unfinished() == 0  # idempotent


def check_net_coins_counts_only_finished_matches(ledger, a, b):
    assert ledger.net_coin_rows() == {}
    open_match = ledger.escrow("dice", {a: 300, b: 300})
    assert ledger.net_coin_rows() == {}  # escrowed coins are not "lost" while the match is on
    ledger.settle(open_match, {a: 450, b: 150})
    assert ledger.net_coin_rows() == {a: 150, b: -150}
    second = ledger.escrow("kuhn", {a: 50, b: 50})
    ledger.settle(second, {a: 0, b: 100})
    assert ledger.net_coin_rows() == {a: 100, b: -100}
    stuck = ledger.escrow("dice", {a: 200, b: 200})
    assert ledger.net_coin_rows() == {a: 100, b: -100}
    ledger.refund_unfinished()  # a refunded match changes nobody's net
    assert ledger.net_coin_rows() == {a: 100, b: -100}
    assert sum(ledger.net_coin_rows().values()) == 0
    assert ledger.game_status(stuck) == "aborted"


ALL_CHECKS = [
    check_net_coins_counts_only_finished_matches,
    check_escrow_debits_every_player,
    check_settle_pays_out_and_conserves_coins,
    check_a_match_settles_only_once,
    check_bad_payouts_are_refused_and_change_nothing,
    check_a_loser_can_be_paid_nothing,
    check_unknown_users_are_refused_without_side_effects,
    check_unknown_game_cannot_be_settled,
    check_refund_returns_the_escrow_of_unfinished_matches,
]
