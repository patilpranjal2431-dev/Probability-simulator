"use strict";

/** Build a table cell using textContent only (never innerHTML), so data can't inject HTML. */
function cell(text, className) {
  const td = document.createElement("td");
  td.textContent = text;
  if (className) td.className = className;
  return td;
}

async function load() {
  const meResponse = await fetch("/me");
  if (meResponse.status === 401) {
    window.location.href = "/";
    return;
  }
  const me = await meResponse.json();
  document.getElementById("username").textContent = me.username;
  document.getElementById("balance").textContent = me.balance.toLocaleString() + " coins";

  const txResponse = await fetch("/me/transactions?limit=50");
  const transactions = await txResponse.json();
  const body = document.getElementById("history");
  document.getElementById("empty").hidden = transactions.length > 0;

  for (const t of transactions) {
    const row = document.createElement("tr");
    row.appendChild(cell(new Date(t.created_at).toLocaleString()));
    row.appendChild(cell(t.kind.replace("_", " ")));
    const sign = t.amount > 0 ? "+" : "";
    row.appendChild(cell(sign + t.amount, t.amount > 0 ? "pos" : "neg"));
    row.appendChild(cell(t.balance_after.toLocaleString()));
    body.appendChild(row);
  }
}

document.getElementById("logout").addEventListener("click", async () => {
  await fetch("/auth/logout", { method: "POST" });
  window.location.href = "/";
});

load();
