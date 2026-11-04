"""The calibration game: get a question, give a probability, see what happened and your Brier score."""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from app.calibration.store import AlreadyAnswered, NoSuchQuestion
from app.deps import get_current_user
from app.models import User

router = APIRouter(prefix="/calibration", tags=["calibration"])


class AnswerIn(BaseModel):
    question_id: str = Field(min_length=1, max_length=40)
    forecast: float = Field(ge=0, le=1, allow_inf_nan=False, description="your probability that the event happens, 0 to 1")

    @field_validator("forecast", mode="before")
    @classmethod
    def _number_only(cls, value):  # noqa: ANN001
        """Pydantic would quietly turn "0.5" or true into a number. Refuse both."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("forecast must be a number")
        return value


def rounded(value, digits: int = 4):
    """Round every float in a nested structure (the stored values stay exact)."""
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {k: rounded(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [rounded(v, digits) for v in value]
    return value


@router.get("/question")
def question(request: Request, user: User = Depends(get_current_user)) -> dict:
    """Your current question (you keep it until you answer, so it cannot be rerolled)."""
    return request.app.state.calibration.next_question(user.id)


@router.post("/answer")
def answer(body: AnswerIn, request: Request, user: User = Depends(get_current_user)) -> dict:
    """Lock in a forecast. Only then is the random experiment run and the answer revealed."""
    try:
        return rounded(request.app.state.calibration.answer(user.id, body.question_id, body.forecast))
    except NoSuchQuestion:
        raise HTTPException(status_code=404, detail="no such question") from None
    except AlreadyAnswered:
        raise HTTPException(status_code=409, detail="that question was already answered") from None
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None


@router.get("/summary")
def summary(request: Request, user: User = Depends(get_current_user)) -> dict:
    """Your scores over time, your calibration, and two charts."""
    data = request.app.state.calibration.summary(user.id)
    charts = data.pop("charts")
    out = rounded(data)
    out["charts"] = charts  # SVG text must not be touched
    return out
