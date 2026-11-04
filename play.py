"""Lobby REST endpoints and the table WebSocket.

REST (needs the login cookie):
    GET  /tables         tables waiting for an opponent
    POST /tables         create a table   {"game": "dice", "buy_in": 500, "bot": "kelly"}
    GET  /tables/mine    the table you are currently seated at, if any

WebSocket:   /ws/tables/{table_id}
    Connect with your login cookie. The first connection to your own table makes you the host;
    another user connecting to an open human table takes the second seat; a seated user who
    reconnects resumes. See app/tables/protocol.py and table.py for the messages.
"""

import logging
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket
from pydantic import BaseModel, Field, StrictInt

from app.config import settings
from app.deps import get_current_user
from app.models import User
from app.services.sessions import get_user_for_token
from app.tables.common import LobbyError, UserRef

log = logging.getLogger("edge_lab.play")
router = APIRouter(tags=["play"])

CLIENT_ERRORS = {"already_seated": 409}  # every other LobbyError is a 400


class CreateTableIn(BaseModel):
    game: str = Field(min_length=1, max_length=20)
    buy_in: StrictInt | None = None  # strict: "500", 500.0 and true are all rejected
    bot: str | None = Field(default=None, max_length=20)


def _lobby(request_or_ws):
    return request_or_ws.app.state.lobby


@router.get("/tables")
def list_tables(request: Request, _user: User = Depends(get_current_user)) -> list[dict]:
    return _lobby(request).list_open()


@router.post("/tables", status_code=201)
async def create_table(body: CreateTableIn, request: Request, user: User = Depends(get_current_user)) -> dict:
    try:
        table = await _lobby(request).create_table(UserRef(user.id, user.username), body.game, body.buy_in, body.bot)
    except LobbyError as error:
        raise HTTPException(
            status_code=CLIENT_ERRORS.get(error.code, 400),
            detail={"code": error.code, "message": error.message},
        ) from None
    return table.info()


@router.get("/tables/mine")
def my_table(request: Request, user: User = Depends(get_current_user)) -> dict:
    table = _lobby(request).table_of(user.id)
    return {"table": table.info() if table else None}


# ------------------------------------------------------------------- WebSocket


class WebSocketConnection:
    """Adapts a Starlette WebSocket to the small Connection interface the Table uses."""

    def __init__(self, websocket: WebSocket) -> None:
        self._ws = websocket

    async def send(self, message: dict) -> None:
        await self._ws.send_json(message)

    async def close(self, code: int = 1000) -> None:
        await self._ws.close(code=code)


def _origin_allowed(websocket: WebSocket) -> bool:
    """Browsers always send Origin on a WebSocket handshake. Refuse other websites, so a
    malicious page cannot drive your session (cross-site WebSocket hijacking). Scripts and
    tests that send no Origin are allowed: they still need a valid login cookie."""
    origin = websocket.headers.get("origin")
    if not origin:
        return True
    return urlparse(origin).netloc == websocket.headers.get("host")


def _authenticate(websocket: WebSocket) -> UserRef | None:
    token = websocket.cookies.get(settings.session_cookie_name)
    if not token:
        return None
    with websocket.app.state.session_factory() as db:
        user = get_user_for_token(db, token)
        if user is None or user.is_bot:
            return None
        return UserRef(user.id, user.username)


@router.websocket("/ws/tables/{table_id}")
async def table_socket(websocket: WebSocket, table_id: str) -> None:
    # Refuse BEFORE accepting: the client gets an HTTP 403 and no socket is ever opened.
    if not _origin_allowed(websocket):
        await websocket.close(code=1008)
        return
    user = _authenticate(websocket)
    if user is None:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    connection = WebSocketConnection(websocket)
    try:
        table = await _lobby(websocket).connect(user, table_id, connection)
    except LobbyError as error:
        await connection.send({"type": "error", "code": error.code, "message": error.message})
        await connection.close(1008)
        return

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            raw = message.get("text") if message.get("text") is not None else message.get("bytes")
            if raw is not None:
                await table.handle(user.id, raw)
    except Exception:  # noqa: BLE001  (never let one bad socket take the server down)
        log.exception("table socket failed for user %s", user.id)
    finally:
        await table.disconnect(user.id, connection)
