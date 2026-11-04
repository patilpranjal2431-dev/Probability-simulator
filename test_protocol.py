"""Client messages are parsed strictly."""

import json

import pytest

from app.games.base import CALL, CHECK, FOLD, bet
from app.tables.protocol import MAX_MESSAGE_BYTES, ProtocolError, parse_action, parse_client_message


def code_of(raw):
    with pytest.raises(ProtocolError) as info:
        parse_client_message(raw)
    return info.value.code


def action_code(message):
    with pytest.raises(ProtocolError) as info:
        parse_action(message)
    return info.value.code


def test_valid_messages_parse():
    assert parse_client_message('{"type": "ping"}') == {"type": "ping"}
    assert parse_client_message(b'{"type": "leave"}')["type"] == "leave"
    assert parse_action({"type": "action", "action": "check"}) == CHECK
    assert parse_action({"type": "action", "action": "call"}) == CALL
    assert parse_action({"type": "action", "action": "fold"}) == FOLD
    assert parse_action({"type": "action", "action": "bet", "amount": 25}) == bet(25)
    assert parse_action({"type": "action", "action": "check", "amount": 0}) == CHECK


@pytest.mark.parametrize("raw", ["", "not json", "{", "[1, 2]", "42", "null", '"text"', '{"no_type": 1}', b"\xff\xfe"])
def test_garbage_is_rejected(raw):
    assert code_of(raw) in ("bad_json", "bad_message")


def test_unknown_type_is_rejected():
    assert code_of('{"type": "cheat"}') == "unknown_type"
    assert code_of('{"type": 5}') == "unknown_type"


def test_oversized_messages_are_rejected_before_parsing():
    assert code_of(json.dumps({"type": "ping", "pad": "x" * MAX_MESSAGE_BYTES})) == "message_too_large"


@pytest.mark.parametrize(
    "message",
    [
        {"type": "action"},
        {"type": "action", "action": "raise"},
        {"type": "action", "action": 5},
        {"type": "action", "action": None},
        {"type": "action", "action": "bet"},  # no amount
        {"type": "action", "action": "bet", "amount": 0},
        {"type": "action", "action": "bet", "amount": -5},
        {"type": "action", "action": "bet", "amount": 2.5},
        {"type": "action", "action": "bet", "amount": "10"},
        {"type": "action", "action": "bet", "amount": True},
        {"type": "action", "action": "bet", "amount": 10**12},
        {"type": "action", "action": "bet", "amount": None},
        {"type": "action", "action": "call", "amount": 5},
        {"type": "action", "action": "check", "amount": True},
    ],
)
def test_bad_actions_are_rejected(message):
    assert action_code(message) == "bad_action"
