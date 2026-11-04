"""Play a whole match between two bots."""

import random
from dataclasses import dataclass

from .base import Bot, Game


@dataclass(frozen=True)
class MatchResult:
    hands: int
    start_stack: int
    final_stacks: tuple[int, int]
    ended_by: str  # "max_hands" or "bust" (someone could no longer pay the ante)

    def net(self, seat: int) -> int:
        """Chips won (+) or lost (-) by `seat` over the whole match."""
        return self.final_stacks[seat] - self.start_stack

    @property
    def winner(self) -> int | None:
        """Seat with more chips at the end, or None for a draw."""
        a, b = self.final_stacks
        return None if a == b else (0 if a > b else 1)


def play_match(
    game: Game,
    bots: tuple[Bot, Bot],
    seed: int,
    starting_stack: int | None = None,
    max_hands: int = 200,
) -> MatchResult:
    """Play hands until `max_hands` or until someone cannot pay the ante.

    The button alternates every hand. `seed` fixes the deals AND each bot's randomness,
    so a match can be replayed exactly. Total chips are conserved.
    """
    stack = game.default_stack if starting_stack is None else starting_stack
    if stack < game.ante:
        raise ValueError("starting stack must cover the ante")
    if max_hands < 1:
        raise ValueError("max_hands must be at least 1")

    master = random.Random(seed)
    game_rng = random.Random(master.getrandbits(63))
    bot_rngs = (random.Random(master.getrandbits(63)), random.Random(master.getrandbits(63)))

    stacks = [stack, stack]
    hands = 0
    ended_by = "max_hands"
    while hands < max_hands:
        if min(stacks) < game.ante:
            ended_by = "bust"
            break
        hand = game.new_hand(game_rng, (stacks[0], stacks[1]), first=hands % 2)
        while not hand.done:
            seat = hand.to_act
            hand.act(bots[seat].act(hand.decision(), bot_rngs[seat]))
        stacks[0] += hand.result.net[0]
        stacks[1] += hand.result.net[1]
        hands += 1

    return MatchResult(hands, stack, (stacks[0], stacks[1]), ended_by)
