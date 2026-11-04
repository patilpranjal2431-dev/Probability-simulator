"use strict";

const $ = (id) => document.getElementById(id);
const imageUrls = new Map();
let current = null; // the question being shown

const KIND_NAMES = {
  dice_exact: "Dice: an exact total",
  dice_at_least: "Dice: at least a total",
  poker: "Poker hands",
  birthday: "Birthday problem",
  hypergeometric: "Drawing cards",
  duel_win: "Dice Duel",
  coin_run: "Runs of heads",
};

function showSvg(imageId, svgText) {
  const old = imageUrls.get(imageId);
  if (old) URL.revokeObjectURL(old);
  const url = URL.createObjectURL(new Blob([svgText], { type: "image/svg+xml" }));
  imageUrls.set(imageId, url);
  $(imageId).src = url;
}

function pct(x, digits = 1) {
  return (x * 100).toFixed(digits) + "%";
}

function fact(key, value) {
  const box = document.createElement("div");
  box.className = "fact";
  const v = document.createElement("div");
  v.className = "v";
  v.textContent = value;
  const k = document.createElement("div");
  k.className = "k";
  k.textContent = key;
  box.append(v, k);
  return box;
}

function describeError(body) {
  if (body && typeof body.detail === "string") return body.detail;
  if (body && Array.isArray(body.detail)) return body.detail.map((d) => d.msg).join("; ");
  return "Something went wrong. Please try again.";
}

async function get(url) {
  const response = await fetch(url);
  if (response.status === 401) {
    window.location.href = "/";
    return null;
  }
  return response.ok ? response.json() : null;
}

async function loadQuestion() {
  $("message").textContent = "";
  $("result").hidden = true;
  $("answer-form").hidden = false;
  const q = await get("/calibration/question");
  if (!q) {
    $("question-text").textContent = "Could not load a question.";
    return;
  }
  current = q;
  $("question-text").textContent = q.text;
  $("slider").value = 50;
  $("number").value = 50;
}

function syncFrom(source) {
  const raw = parseFloat(source.value);
  const clean = Number.isFinite(raw) ? Math.min(100, Math.max(0, raw)) : 50;
  $("slider").value = clean;
  if (source === $("slider")) $("number").value = clean;
}

async function submit() {
  if (!current) return;
  const percent = parseFloat($("number").value);
  if (!Number.isFinite(percent) || percent < 0 || percent > 100) {
    $("message").textContent = "Enter a probability between 0 and 100.";
    return;
  }
  $("submit").disabled = true;
  try {
    const response = await fetch("/calibration/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question_id: current.question_id, forecast: percent / 100 }),
    });
    if (response.status === 401) {
      window.location.href = "/";
      return;
    }
    const result = await response.json().catch(() => null);
    if (!response.ok) {
      $("message").textContent = describeError(result);
      if (response.status === 409) await loadQuestion();
      return;
    }
    showResult(result);
    await loadSummary();
  } catch (error) {
    $("message").textContent = "Cannot reach the server.";
  } finally {
    $("submit").disabled = false;
  }
}

function showResult(result) {
  $("answer-form").hidden = true;
  const outcome = $("outcome");
  outcome.textContent = result.event_happened ? "The event happened." : "The event did not happen.";
  outcome.className = "big " + (result.event_happened ? "happened" : "didnt");
  $("what-happened").textContent = result.what_happened;
  const facts = $("result-facts");
  facts.replaceChildren();
  const truth = pct(result.true_probability) +
    (result.true_probability_fraction ? " (" + result.true_probability_fraction + ")" : "");
  facts.append(
    fact("Your answer", pct(result.forecast, 0)),
    fact("The true probability", truth),
    fact("How close you were", result.verdict),
    fact("Your Brier score this time (lower is better)", result.brier.toFixed(3)),
    fact("Noise-free expected score for this answer", result.expected_brier.toFixed(3)),
    fact("Best anyone could expect on this question", result.uncertainty.toFixed(3)),
    fact("Cost of your estimate beyond that", result.excess_brier.toFixed(4)),
  );
  $("result").hidden = false;
}

async function loadSummary() {
  const data = await get("/calibration/summary");
  if (!data || data.n === 0) return;
  const sum = data.summary;
  const facts = $("summary-facts");
  facts.replaceChildren();
  const se = sum.brier_standard_error === null ? "" : " (+/- " + sum.brier_standard_error.toFixed(3) + ")";
  facts.append(
    fact("Questions answered", String(sum.n)),
    fact("Your average Brier score" + se, sum.mean_brier.toFixed(3)),
    fact("Noise-free expected score", sum.mean_expected_brier.toFixed(3)),
    fact("A perfect forecaster would score", sum.mean_uncertainty.toFixed(3)),
    fact("Always saying 50% scores", sum.baseline_brier.toFixed(3)),
    fact("Skill vs always saying 50%", pct(sum.skill_vs_always_50_expected, 0)),
    fact("Reliability cost (zero is perfect)", sum.mean_excess.toFixed(4)),
    fact("Average error (you - truth)", (sum.mean_signed_error > 0 ? "+" : "") + sum.mean_signed_error.toFixed(3)),
  );
  let advice;
  if (sum.n < 15) advice = "Keep going: with so few answers the scores are mostly noise.";
  else if (sum.mean_signed_error > 0.03) advice = "You tend to guess probabilities too high.";
  else if (sum.mean_signed_error < -0.03) advice = "You tend to guess probabilities too low.";
  else advice = "No clear bias: your guesses are as likely to be too high as too low.";
  $("advice").textContent = advice;

  showSvg("chart-brier", data.charts.brier);
  showSvg("chart-reliability", data.charts.reliability);
  const body = $("by-kind");
  body.replaceChildren();
  for (const row of sum.by_kind) {
    const tr = document.createElement("tr");
    const cells = [
      [KIND_NAMES[row.kind] || row.kind, ""],
      [String(row.n), "num"],
      [(row.mean_signed_error > 0 ? "+" : "") + row.mean_signed_error.toFixed(3), "num"],
      [row.mean_excess.toFixed(4), "num"],
    ];
    for (const [text, cls] of cells) {
      const td = document.createElement("td");
      td.textContent = text;
      if (cls) td.className = cls;
      tr.appendChild(td);
    }
    body.appendChild(tr);
  }
  $("summary-section").hidden = false;
}

$("slider").addEventListener("input", () => { $("number").value = $("slider").value; });
$("number").addEventListener("input", () => syncFrom($("number")));
$("submit").addEventListener("click", submit);
$("number").addEventListener("keydown", (event) => { if (event.key === "Enter") submit(); });
$("next").addEventListener("click", loadQuestion);
loadQuestion();
loadSummary();
