"use strict";

/** Build a table cell with textContent only, so a username can never inject HTML. */
function cell(text, className) {
  const td = document.createElement("td");
  td.textContent = text;
  if (className) td.className = className;
  return td;
}

function signed(value) {
  const text = Number(value).toFixed(2);
  return value > 0 ? "+" + text : text;
}

function fill(bodyId, emptyId, rows, columns) {
  const body = document.getElementById(bodyId);
  body.replaceChildren();
  document.getElementById(emptyId).hidden = rows.length > 0;
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const column of columns) tr.appendChild(column(row));
    body.appendChild(tr);
  }
}

async function load() {
  const minHands = Math.max(1, parseInt(document.getElementById("min-hands").value, 10) || 1);
  const response = await fetch("/leaderboards?min_hands=" + minHands);
  if (response.status === 401) {
    window.location.href = "/";
    return;
  }
  if (!response.ok) {
    document.getElementById("message").textContent = "Could not load the leaderboards.";
    return;
  }
  const boards = await response.json();

  fill("skill", "skill-empty", boards.skill, [
    (r) => cell(r.rank),
    (r) => cell(r.username),
    (r) => cell(r.hands, "num"),
    (r) => cell(signed(r.ev_earned), "num"),
    (r) => cell(signed(r.ev_potential), "num"),
    (r) => cell(r.regret_per_decision.toFixed(2), "num"),
    (r) => cell(signed(r.actual_net), "num"),
    (r) => cell(signed(r.luck), "num " + (r.luck >= 0 ? "luckpos" : "luckneg")),
    (r) => cell(r.coin_rank === null ? "-" : r.coin_rank, "num"),
  ]);

  fill("coins", "coins-empty", boards.coins, [
    (r) => cell(r.rank),
    (r) => cell(r.username),
    (r) => cell(r.net_coins > 0 ? "+" + r.net_coins : r.net_coins, "num"),
    (r) => cell(r.skill_rank === null ? "-" : r.skill_rank, "num"),
  ]);
}

document.getElementById("refresh").addEventListener("click", load);
load();
