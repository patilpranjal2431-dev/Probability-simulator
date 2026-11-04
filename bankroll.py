"""Bankroll simulator endpoints. The simulation is CPU-heavy, so it runs in a worker thread and
at most two run at once; input sizes are capped (see app/bankroll/simulate.py)."""

import asyncio
import secrets
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, StrictInt, field_validator

from app.bankroll.charts import fan_svg, histogram_svg, kelly_curve_svg
from app.bankroll.model import BetSpec
from app.bankroll.presets import PRESETS
from app.bankroll.simulate import MAX_BETS, MAX_PATHS, Strategy, simulate_bankroll
from app.deps import get_current_user
from app.models import User

router = APIRouter(prefix="/simulations", tags=["simulations"])


class BankrollIn(BaseModel):
    win: float = Field(ge=0, le=1, allow_inf_nan=False, description="chance each bet wins")
    lose: float = Field(ge=0, le=1, allow_inf_nan=False, description="chance each bet loses (the rest is a push)")
    odds: float = Field(default=1.0, gt=0, le=1000, allow_inf_nan=False, description="net payout per 1 staked")
    strategy: Literal["kelly", "fraction", "flat"] = "kelly"
    value: float = Field(default=1.0, gt=0, le=1e9, allow_inf_nan=False)
    bets: StrictInt = Field(default=200, ge=1, le=MAX_BETS)
    paths: StrictInt = Field(default=2000, ge=1, le=MAX_PATHS)
    start: float = Field(default=1000.0, gt=0, le=1e9, allow_inf_nan=False)
    ruin_fraction: float = Field(default=0.1, ge=0, lt=1, allow_inf_nan=False)
    target: float | None = Field(default=None, gt=0, le=1e12, allow_inf_nan=False)
    seed: StrictInt | None = Field(default=None, ge=0, lt=2**31, description="omit for a fresh random run")

    @field_validator("win", "lose", "odds", "value", "start", "ruin_fraction", "target", mode="before")
    @classmethod
    def _numbers_only(cls, value):  # noqa: ANN001
        """Pydantic would quietly turn "0.5" or true into a number. Refuse both."""
        if value is None or (isinstance(value, (int, float)) and not isinstance(value, bool)):
            return value
        raise ValueError("must be a number")


@router.get("/presets")
def presets(_user: User = Depends(get_current_user)) -> list[dict]:
    return PRESETS


@router.post("/bankroll")
async def run_bankroll(body: BankrollIn, request: Request, _user: User = Depends(get_current_user)) -> dict:
    """Simulate a bankroll. Returns the numbers and three SVG charts."""
    seed = secrets.randbelow(2**31) if body.seed is None else body.seed
    try:
        spec = BetSpec(body.win, body.lose, body.odds)
        strategy = Strategy(body.strategy, body.value)
        async with request.app.state.simulation_slots:  # at most two heavy simulations at a time
            result = await asyncio.to_thread(
                simulate_bankroll, spec, strategy, body.bets, body.paths, body.start,
                body.ruin_fraction, body.target, seed,
            )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    return {
        "summary": result.summary(),
        "breakeven_fraction": spec.breakeven_fraction(),
        "charts": {
            "distribution": histogram_svg(result),
            "paths": fan_svg(result),
            "kelly_curve": kelly_curve_svg(spec),
        },
    }
