"use strict";

const message = document.getElementById("message");

/** Turn a FastAPI error body into readable text. */
function describeError(body) {
  if (body && typeof body.detail === "string") return body.detail;
  if (body && Array.isArray(body.detail)) {
    return body.detail.map((d) => d.msg).join("; ");
  }
  return "Something went wrong. Please try again.";
}

async function submitForm(event, url) {
  event.preventDefault();
  message.textContent = "";
  const data = Object.fromEntries(new FormData(event.target));
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (response.ok) {
      window.location.href = "/profile.html";
      return;
    }
    message.textContent = describeError(await response.json().catch(() => null));
  } catch (error) {
    message.textContent = "Cannot reach the server.";
  }
}

document.getElementById("login-form")
  .addEventListener("submit", (e) => submitForm(e, "/auth/login"));
document.getElementById("register-form")
  .addEventListener("submit", (e) => submitForm(e, "/auth/register"));
