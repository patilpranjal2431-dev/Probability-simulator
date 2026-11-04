"""FastAPI application entry point."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.orm import Session

from app import __version__, models  # noqa: F401  (importing models registers the tables)
from app.database import Base, SessionLocal, engine, get_db
from app.calibration.service import CalibrationService
from app.calibration.sql_store import SqlCalibrationStore
from app.grading.service import Standings
from app.grading.sql_store import SqlGradeStore
from app.routers import auth, bankroll, calibration, leaderboards, me, play
from app.services.bots import ensure_bot_users, make_directory
from app.tables.sql_ledger import SqlLedger
from app.tables.wiring import build_lobby

log = logging.getLogger("edge_lab")

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start up: create tables and bot accounts, return coins from matches a crash left
    unfinished, and open the lobby. (Assumes ONE server process: see README.)"""
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        ensure_bot_users(db)
    ledger = SqlLedger(SessionLocal)
    refunded = ledger.refund_unfinished()
    if refunded:
        log.warning("refunded the escrow of %d unfinished match(es) from a previous run", refunded)
    grade_store = SqlGradeStore(SessionLocal)
    app.state.session_factory = SessionLocal
    app.state.grade_store = grade_store
    app.state.standings = Standings(grade_store, ledger, make_directory(SessionLocal))
    app.state.lobby = build_lobby(SessionLocal, ledger=ledger, grade_store=grade_store)
    app.state.calibration = CalibrationService(SqlCalibrationStore(SessionLocal))
    app.state.simulation_slots = asyncio.Semaphore(2)  # at most two heavy bankroll simulations at once
    yield
    await app.state.lobby.close_all()


app = FastAPI(title="Edge Lab", version=__version__, lifespan=lifespan)

app.include_router(auth.router)
app.include_router(me.router)
app.include_router(play.router)
app.include_router(leaderboards.router)
app.include_router(bankroll.router)
app.include_router(calibration.router)


@app.get("/health")
def health(db: Session = Depends(get_db)) -> dict[str, str]:
    """Liveness check that also proves the database is reachable."""
    db.execute(text("SELECT 1"))
    return {"status": "ok", "version": __version__}


# Must be LAST: it catches every path that no API route handled.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
