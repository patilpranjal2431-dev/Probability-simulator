"""Plain data types for grading (no database, no web)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionContext:
    """What the table records each time a player acts. SERVER-SIDE ONLY: it contains the
    player's own hidden value and enough history to re-create the decision exactly."""

    hand_number: int  # 1-based
    decision_index: int  # how many decisions this seat had already made in this hand
    seat: int
    own_value: int  # this player's own die / card (the opponent's value is NOT recorded)
    stacks: tuple[int, int]  # chips each seat held at the start of the hand
    first: int  # seat that acted first
    history: tuple[tuple[int, str, int], ...]  # (seat, kind, amount) of every action before this one
    action_kind: str
    action_amount: int
    auto: bool  # True if the table acted for the player (timeout / disconnect)


@dataclass(frozen=True)
class GradedDecision:
    hand_number: int
    decision_index: int
    seat: int
    decision_kind: str  # "bet" (check or bet) or "respond" (call or fold)
    own_value: int
    action_kind: str
    action_amount: int
    best_kind: str
    best_amount: int
    ev_taken: float  # expected net chips of the action taken, with best play afterwards
    ev_best: float  # expected net chips of the best available action
    regret: float  # ev_best - ev_taken, never negative
    hand_net: int  # what ACTUALLY happened to this player in the hand (includes luck)
    counts_for_ev: bool  # True for the player's last own decision in a hand with no automatic moves


@dataclass(frozen=True)
class SkillTotals:
    """One player's graded history, summed."""

    user_id: int
    hands: int  # hands that count towards EV
    decisions: int
    ev_earned: float  # sum of ev_taken over the counted hands
    regret: float  # sum of regret over all graded decisions
    actual_net: int  # chips actually won or lost in the counted hands
    ev_potential: float = 0.0  # sum of the BEST EV in the counted hands: what perfect play would earn from these cards

    @property
    def luck(self) -> float:
        """Actual result minus expected result. Positive = ran better than the decisions deserved."""
        return self.actual_net - self.ev_earned
