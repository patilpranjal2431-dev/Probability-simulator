"""Client -> server messages: strict parsing, so garbage never reaches the game logic.

Messages are JSON objects:
    {"type": "action", "action": "check"}
    {"type": "action", "action": "bet", "amount": 50}
    {"type": "action", "action": "call"}      (or "fold")
    {"type": "leave"}
    {"type": "ping"}
"""

import json

from app.games.base import ACTION_KINDS, Action

MAX_MESSAGE_BYTES = 2048
MAX_AMOUNT = 10**9
CLIENT_TYPES = ("action", "leave", "ping")


class ProtocolError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def parse_client_message(raw: str | bytes) -> dict:
    """Return the decoded message, or raise ProtocolError."""
    if len(raw) > MAX_MESSAGE_BYTES:
        raise ProtocolError("message_too_large", f"messages are limited to {MAX_MESSAGE_BYTES} bytes")
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        raise ProtocolError("bad_json", "message is not valid JSON") from None
    if not isinstance(data, dict) or "type" not in data:
        raise ProtocolError("bad_message", 'expected a JSON object with a "type"')
    if data["type"] not in CLIENT_TYPES:
        raise ProtocolError("unknown_type", f"unknown message type; use one of {', '.join(CLIENT_TYPES)}")
    return data


def parse_action(data: dict) -> Action:
    """Turn an "action" message into an Action, or raise ProtocolError."""
    kind = data.get("action")
    if not isinstance(kind, str) or kind not in ACTION_KINDS:
        raise ProtocolError("bad_action", f"action must be one of {', '.join(ACTION_KINDS)}")
    amount = data.get("amount", 0)
    if kind == "bet":
        if isinstance(amount, bool) or not isinstance(amount, int) or not 1 <= amount <= MAX_AMOUNT:
            raise ProtocolError("bad_action", "a bet needs a whole-number amount of at least 1")
        return Action("bet", amount)
    if amount not in (0, None) or isinstance(amount, bool):
        raise ProtocolError("bad_action", f"{kind} takes no amount")
    return Action(kind)
