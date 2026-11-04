"""The Phase 7 pages: they only read fields the API returns, only send fields it accepts, and every
number the text quotes is true."""

import itertools
import random
import re
from pathlib import Path

import pytest

from app.bankroll.model import BetSpec
from app.bankroll.presets import PRESETS
from app.bankroll.simulate import Strategy, simulate_bankroll
from app.calibration.questions import KINDS
from app.calibration.scoring import brier, summarize
from app.calibration.service import CalibrationService
from app.calibration.store import MemoryCalibrationStore

ROOT = Path(__file__).resolve().parents[2]
FRONTEND, ROUTERS = ROOT / "frontend", ROOT / "backend" / "app" / "routers"


def read(path):
    return path.read_text(encoding="utf-8")


def used(source, variable):
    """Field names read as `variable.field` in a script."""
    return set(re.findall(rf"\b{variable}\.(\w+)", source))


def model_fields(router_source, class_name):
    block = router_source[router_source.index(f"class {class_name}"):]
    block = block[: block.index("\n\n\n")] if "\n\n\n" in block else block
    return set(re.findall(r"^    (\w+): ", block, flags=re.M))


def routes(router_source):
    prefix = re.search(r'prefix="([^"]*)"', router_source).group(1)
    return {(m.group(1).upper(), prefix + m.group(2)) for m in re.finditer(r'@router\.(get|post)\("([^"]*)"', router_source)}


def calibration_service_with_answers(n=20):
    seeds = itertools.count(1)
    service = CalibrationService(MemoryCalibrationStore(), lambda: random.Random(next(seeds)))
    result = None
    for _ in range(n):
        q = service.next_question(1)
        result = service.answer(1, q["question_id"], 0.4)
    return service, q, result


# ================================================================== safety


def test_scripts_never_insert_markup():
    for name in ("bankroll.js", "calibration.js"):
        source = read(FRONTEND / name)
        assert not re.search(r"\.\s*(innerHTML|outerHTML)\b|document\.write\(|insertAdjacentHTML|eval\(", source), name


def test_charts_are_shown_through_blob_images_not_inline_markup():
    for name in ("bankroll.js", "calibration.js"):
        source = read(FRONTEND / name)
        assert "image/svg+xml" in source and "createObjectURL" in source and "revokeObjectURL" in source


# =========================================== scripts and API agree on the data


def test_the_bankroll_script_reads_only_fields_the_api_returns():
    script = read(FRONTEND / "bankroll.js")
    result = simulate_bankroll(BetSpec(0.55, 0.45), Strategy("kelly", 1), 20, 50, seed=1)
    summary = result.summary()
    assert used(script, "s") <= set(summary), used(script, "s") - set(summary)
    assert {"strategy", "ruin_ci"} <= set(summary) and set(summary["strategy"]) == {"kind", "value"}
    assert {"value"} <= used(script, "s") | {"value"}
    assert used(script, "data") <= {"summary", "charts", "breakeven_fraction", "detail"} , used(script, "data")
    assert used(script, "p") <= set(PRESETS[0]) | {"id"}, used(script, "p")  # the preset fields it uses
    router = read(ROUTERS / "bankroll.py")
    for chart in ("distribution", "paths", "kelly_curve"):
        assert chart in script and f'"{chart}"' in router


def test_the_bankroll_script_sends_only_fields_the_api_accepts():
    script = read(FRONTEND / "bankroll.js")
    payload = re.search(r"const payload = \{(.*?)\n  \};", script, flags=re.S).group(1)
    sent = set(re.findall(r"^\s{4}(\w+)[:,]", payload, flags=re.M))
    sent |= set(re.findall(r"payload\.(\w+) =", script))
    accepted = model_fields(read(ROUTERS / "bankroll.py"), "BankrollIn")
    assert sent and sent <= accepted, sent - accepted
    assert {"win", "lose", "odds", "strategy", "value", "bets", "paths", "start", "ruin_fraction", "target", "seed"} == accepted


def test_the_calibration_script_reads_only_fields_the_api_returns():
    script = read(FRONTEND / "calibration.js")
    service, last_question, last_result = calibration_service_with_answers()
    data = service.summary(1)
    assert used(script, "result") <= set(last_result), used(script, "result") - set(last_result)
    assert used(script, "q") <= set(service.next_question(1)), used(script, "q")
    assert used(script, "sum") <= set(data["summary"]), used(script, "sum") - set(data["summary"])
    assert used(script, "row") <= set(data["summary"]["by_kind"][0]), used(script, "row")
    assert used(script, "data") <= set(data) | {"n"}, used(script, "data")
    assert {"brier", "reliability"} == set(data["charts"]) and "charts.brier" in script and "charts.reliability" in script


def test_the_calibration_script_sends_only_fields_the_api_accepts():
    script = read(FRONTEND / "calibration.js")
    body = re.findall(r"JSON\.stringify\(\{ (.*?) \}\)", script)[0]
    sent = set(re.findall(r"(\w+):", body))
    assert sent == model_fields(read(ROUTERS / "calibration.py"), "AnswerIn") == {"question_id", "forecast"}


def test_every_url_the_scripts_call_exists_in_the_routers():
    known = routes(read(ROUTERS / "bankroll.py")) | routes(read(ROUTERS / "calibration.py"))
    for name in ("bankroll.js", "calibration.js"):
        for url in re.findall(r'"(/(?:simulations|calibration)/[a-z]+)', read(FRONTEND / name)):
            assert any(path == url for _method, path in known), url


def test_the_kind_labels_cover_every_question_kind():
    script = read(FRONTEND / "calibration.js")
    labelled = set(re.findall(r"^\s+(\w+): \"", script[script.index("const KIND_NAMES"):script.index("};", script.index("const KIND_NAMES"))], flags=re.M))
    assert labelled == set(KINDS)


def test_the_by_kind_table_always_gets_a_row_for_every_kind_the_server_sends():
    service, *_ = calibration_service_with_answers(60)
    kinds_sent = {row["kind"] for row in service.summary(1)["summary"]["by_kind"]}
    assert kinds_sent <= set(KINDS)


# ======================================================== the text is true


def test_the_brier_examples_on_the_calibration_page_are_correct():
    page = read(FRONTEND / "calibration.html")
    assert brier(0.5, True) == brier(0.5, False) == 0.25 and "scores 0.25" in page
    assert round(brier(0.7, True), 2) == 0.09 and round(brier(0.7, False), 2) == 0.49
    assert "0.09" in page and "0.49" in page
    assert "Lower is better" in page and "Brier score" in page


def test_the_kelly_numbers_on_the_bankroll_page_are_correct():
    page = read(FRONTEND / "bankroll.html")
    spec = BetSpec(0.55, 0.45, 1)
    assert round(spec.kelly_fraction(), 3) == 0.10 and "it is 10%" in page
    assert f"{spec.breakeven_fraction():.1%}" == "19.9%" and "19.9%" in page
    assert 1.9 < spec.breakeven_fraction() / spec.kelly_fraction() < 2.0 and "twice</em> Kelly" in page


def test_the_default_preset_values_in_the_form_match_the_55_percent_example():
    page = read(FRONTEND / "bankroll.html")
    assert 'id="win" type="number" step="0.1" min="0" max="100" value="55"' in page
    assert 'id="lose" type="number" step="0.1" min="0" max="100" value="45"' in page


def test_the_pages_link_to_each_other_and_from_the_profile():
    profile = read(FRONTEND / "profile.html")
    assert "/bankroll.html" in profile and "/calibration.html" in profile
    assert "/calibration.html" in read(FRONTEND / "bankroll.html") and "/bankroll.html" in read(FRONTEND / "calibration.html")
    for page in ("bankroll.html", "calibration.html"):
        assert "/profile.html" in read(FRONTEND / page) and "No real money" in read(FRONTEND / page)
