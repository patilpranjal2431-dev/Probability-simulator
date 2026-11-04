"use strict";

const $ = (id) => document.getElementById(id);
const imageUrls = new Map(); // so old Blob URLs can be released

/** Show an SVG string in an <img>. An SVG used this way cannot run scripts, and nothing is parsed as HTML. */
function showSvg(imageId, svgText) {
  const image = $(imageId);
  const old = imageUrls.get(imageId);
  if (old) URL.revokeObjectURL(old);
  const url = URL.createObjectURL(new Blob([svgText], { type: "image/svg+xml" }));
  imageUrls.set(imageId, url);
  image.src = url;
}

function pct(x, digits = 1) {
  return (x * 100).toFixed(digits) + "%";
}

function coins(x) {
  return Math.round(x).toLocaleString();
}

function describeError(body) {
  if (body && typeof body.detail === "string") return body.detail;
  if (body && Array.isArray(body.detail)) return body.detail.map((d) => d.loc.slice(1).join(".") + ": " + d.msg).join("; ");
  return "Something went wrong. Please try again.";
}

function fact(key, value, kind) {
  const box = document.createElement("div");
  box.className = "fact" + (kind ? " " + kind : "");
  const v = document.createElement("div");
  v.className = "v";
  v.textContent = value;
  const k = document.createElement("div");
  k.className = "k";
  k.textContent = key;
  box.append(v, k);
  return box;
}

const VALUE_LABELS = {
  kelly: ["Multiple of Kelly (1 = full, 0.5 = half)", "1", "0.05"],
  fraction: ["Percent of bankroll staked each bet", "5", "0.5"],
  flat: ["Coins staked each bet", "10", "1"],
};

function updateValueLabel() {
  const [label, example, step] = VALUE_LABELS[$("strategy").value];
  $("value-label").textContent = label;
  $("value").value = example;
  $("value").step = step;
}

async function loadPresets() {
  const response = await fetch("/simulations/presets");
  if (response.status === 401) {
    window.location.href = "/";
    return;
  }
  const presets = await response.json();
  const select = $("preset");
  const custom = document.createElement("option");
  custom.value = "";
  custom.textContent = "Custom";
  select.appendChild(custom);
  for (const p of presets) {
    const option = document.createElement("option");
    option.value = p.id;
    option.textContent = p.name;
    select.appendChild(option);
  }
  select.addEventListener("change", () => {
    const chosen = presets.find((p) => p.id === select.value);
    if (!chosen) return;
    $("win").value = +(chosen.win * 100).toFixed(4);
    $("lose").value = +(chosen.lose * 100).toFixed(4);
    $("odds").value = chosen.odds;
  });
}

function notes(s, data) {
  const out = [];
  if (s.kelly_fraction === 0) {
    out.push("This bet has no edge, so the Kelly stake is zero. Staking anything on it loses money in the long run.");
  }
  if (s.stake_fraction !== null && data.breakeven_fraction !== null && s.stake_fraction > data.breakeven_fraction) {
    out.push("You are staking " + pct(s.stake_fraction) + " of your bankroll, above the break-even fraction of " +
      pct(data.breakeven_fraction) + ". Growth is negative in the long run even though every bet is favourable.");
  }
  if (s.mean_final > 5 * s.median_final && s.median_final > 0) {
    out.push("The average (" + coins(s.mean_final) + ") is far above the median (" + coins(s.median_final) +
      "): a few very lucky paths dominate the average. The median is the typical outcome.");
  }
  if (s.fraction_wiped_out > 0) {
    out.push(pct(s.fraction_wiped_out) + " of paths ended with nothing left.");
  }
  if (s.theory_growth_rate !== null && s.mean_log_growth !== null) {
    out.push("Measured growth was " + s.mean_log_growth.toFixed(5) + " per bet; the formula predicts " +
      s.theory_growth_rate.toFixed(5) + ".");
  }
  return out;
}

async function run(event) {
  event.preventDefault();
  $("message").textContent = "";
  $("run").disabled = true;
  const strategy = $("strategy").value;
  const rawValue = parseFloat($("value").value);
  const payload = {
    win: parseFloat($("win").value) / 100,
    lose: parseFloat($("lose").value) / 100,
    odds: parseFloat($("odds").value),
    strategy,
    value: strategy === "fraction" ? rawValue / 100 : rawValue,
    bets: parseInt($("bets").value, 10),
    paths: parseInt($("paths").value, 10),
    start: parseFloat($("start").value),
    ruin_fraction: parseFloat($("ruin").value) / 100,
  };
  if ($("target").value !== "") payload.target = parseFloat($("target").value);
  if ($("seed").value !== "") payload.seed = parseInt($("seed").value, 10);

  try {
    const response = await fetch("/simulations/bankroll", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (response.status === 401) {
      window.location.href = "/";
      return;
    }
    const data = await response.json().catch(() => null);
    if (!response.ok) {
      $("message").textContent = describeError(data);
      return;
    }
    render(data);
  } catch (error) {
    $("message").textContent = "Cannot reach the server.";
  } finally {
    $("run").disabled = false;
  }
}

function render(data) {
  const s = data.summary;
  const facts = $("facts");
  facts.replaceChildren();
  const ruinRange = pct(s.ruin_ci[0]) + " to " + pct(s.ruin_ci[1]);
  facts.append(
    fact("Kelly fraction for this bet", s.kelly_fraction === 0 ? "0% (no edge)" : pct(s.kelly_fraction)),
    fact("Stake used each bet", s.stake_fraction === null ? coins(s.strategy.value) + " coins (flat)" : pct(s.stake_fraction)),
    fact("Risk of ruin (95% range " + ruinRange + ")", pct(s.ruin_probability), s.ruin_probability > 0.1 ? "bad" : "good"),
    fact("Median final bankroll", coins(s.median_final), s.median_final >= s.start ? "good" : "bad"),
    fact("Mean final bankroll", coins(s.mean_final)),
    fact("Finished below the start", pct(s.prob_below_start), s.prob_below_start > 0.4 ? "bad" : ""),
    fact("Finished at 2x or more", pct(s.prob_double)),
    fact("Typical worst drawdown (median)", pct(s.median_max_drawdown, 0)),
    fact("Worst-case drawdown (90th percentile)", pct(s.p90_max_drawdown, 0)),
  );
  if (s.target_probability !== null) facts.append(fact("Reached the profit goal", pct(s.target_probability)));
  if (s.median_time_to_ruin !== null) facts.append(fact("Median time to ruin (bets)", String(Math.round(s.median_time_to_ruin))));
  facts.append(fact("Seed (reuse it to repeat this run)", String(s.seed)));

  const list = $("notes");
  list.replaceChildren();
  for (const text of notes(s, data)) {
    const li = document.createElement("li");
    li.textContent = text;
    list.appendChild(li);
  }
  showSvg("chart-distribution", data.charts.distribution);
  showSvg("chart-paths", data.charts.paths);
  showSvg("chart-kelly", data.charts.kelly_curve);
  $("results").hidden = false;
}

$("strategy").addEventListener("change", updateValueLabel);
$("form").addEventListener("submit", run);
loadPresets();
