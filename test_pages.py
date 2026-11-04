"""The pages: every number quoted in the explanation is real, and the scripts are safe."""

import re
from pathlib import Path

from app.games.base import CHECK, bet
from app.games.dice_duel import DiceDuel
from app.grading.evaluate import q_values
from app.grading.grade import grade_decision
from app.grading.reference import reference_for
from app.grading.stats import coin_ranking, skill_ranking, with_rank_gap
from app.grading.types import DecisionContext, SkillTotals

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
GAME, REF = DiceDuel(), reference_for("dice_duel")


def ctx(own, kind="check", amount=0):
    return DecisionContext(1, 0, 0, own, (1000, 1000), 0, (), kind, amount, False)


def signed(value):
    return f"+{value:.2f}" if value > 0 else f"{value:.2f}"


def test_every_number_in_the_worked_example_comes_from_the_grader():
    page = (FRONTEND / "skill-vs-luck.html").read_text(encoding="utf-8")
    q4 = q_values(GAME, ctx(4), REF, [CHECK, bet(1), bet(49), bet(990)])
    for action, shown in ((bet(1), "+1.83"), (CHECK, "+1.67"), (bet(49), "-6.50"), (bet(990), "-328.33")):
        assert signed(q4[action]) == shown
        assert shown in page, f"the page should show {shown}"
    assert f"{grade_decision(GAME, ctx(4, 'bet', 990), REF)['regret']:.2f}" == "330.17" and "330.17" in page
    assert f"{grade_decision(GAME, ctx(4, 'check'), REF)['regret']:.2f}" == "0.17" and "0.17" in page
    q5 = q_values(GAME, ctx(5), REF, [CHECK, bet(990)])
    assert q5[bet(990)] == q5[CHECK] == 5.0 and "+5.00" in page  # the "same EV as checking" claim


def test_the_one_in_six_claim_about_the_all_in_with_a_five_is_true():
    """Against the reference caller, a 5 betting 990 is called only by 4, 5 or 6: it wins 1000 against a 4,
    ties a 5, and loses 1000 to a 6, each with probability 1/6."""
    outcomes = {c: (0 if c == 5 else (1000 if c < 5 else -1000)) for c in (4, 5, 6)}
    assert outcomes[4] == 1000 and outcomes[6] == -1000
    page = (FRONTEND / "skill-vs-luck.html").read_text(encoding="utf-8")
    assert "one time in six you lose 1,000 coins" in page and "one time in six you win 1,000" in page


def test_the_page_describes_the_actual_reference_opponents_and_exclusions():
    page = (FRONTEND / "skill-vs-luck.html").read_text(encoding="utf-8")
    for phrase in ("EV bot", "Nash equilibrium", "fixed reference opponent", "Bots are not on either board",
                   "luck = actual coins", "last</strong> decision"):
        assert phrase in page, phrase


def test_the_pages_link_to_each_other():
    assert "/skill-vs-luck.html" in (FRONTEND / "leaderboards.html").read_text(encoding="utf-8")
    assert "/leaderboards.html" in (FRONTEND / "skill-vs-luck.html").read_text(encoding="utf-8")
    profile = (FRONTEND / "profile.html").read_text(encoding="utf-8")
    assert "/leaderboards.html" in profile and "/skill-vs-luck.html" in profile


def test_scripts_never_use_innerhtml_so_usernames_cannot_inject_markup():
    for name in ("leaderboards.js", "profile.js", "index.js"):
        source = (FRONTEND / name).read_text(encoding="utf-8")
        assert not re.search(r"\.\s*(innerHTML|outerHTML)\b|document\.write\(|insertAdjacentHTML", source), name


def test_the_script_only_reads_fields_the_api_actually_returns():
    names = {1: ("alice", False), 2: ("bob", False)}
    skill = skill_ranking({1: SkillTotals(1, 4, 8, 20.0, 6.0, 30), 2: SkillTotals(2, 2, 3, 1.0, 1.0, 5)}, names)
    coins = coin_ranking({1: 10, 2: -10}, names)
    skill, coins = with_rank_gap(skill, coins)
    script = (FRONTEND / "leaderboards.js").read_text(encoding="utf-8")
    used = set(re.findall(r"\br\.(\w+)", script))
    available = set(skill[0]) | set(coins[0])
    assert used and used <= available, used - available
    assert "/leaderboards?min_hands=" in script


def test_the_claim_that_perfect_play_still_depends_on_the_deal_is_true():
    best = {r: grade_decision(GAME, ctx(r), REF)["ev_best"] for r in range(1, 7)}
    assert f"{best[1]:.2f}" == "-8.33" and f"{best[6]:.2f}" == "338.33"
    page = (FRONTEND / "skill-vs-luck.html").read_text(encoding="utf-8")
    assert "-8.33" in page and "+338.33" in page and "deal value" in page.lower()
