/* kvrot playground frontend — vanilla JS, no build step (heimdall-style). */

const qs = (s) => document.querySelector(s);
const TOKEN = new URLSearchParams(location.search).get("token");

let session = null;      // latest state view from the server
let busy = false;

function api(path, opts = {}) {
  const headers = { "Content-Type": "application/json", ...(opts.headers || {}) };
  if (TOKEN) headers["X-Kvrot-Token"] = TOKEN;
  return fetch(path, { ...opts, headers }).then(async (r) => {
    if (!r.ok) {
      let detail = r.statusText;
      try { detail = (await r.json()).detail || detail; } catch {}
      throw new Error(`${r.status}: ${detail}`);
    }
    return r.json();
  });
}

/* ── health ──────────────────────────────────────────────────────── */

async function pollHealth() {
  try {
    const h = await api("/api/health");
    qs("#conn-dot").className = "status-dot " + (h.vllm === "up" ? "ok" : "err");
    qs("#conn-status").textContent = h.vllm === "up" ? "vLLM up" : "vLLM down";
    qs("#model-tag").textContent = h.model || h.bot_name || "—";
  } catch (e) {
    qs("#conn-dot").className = "status-dot err";
    qs("#conn-status").textContent = "backend unreachable";
  }
}

/* ── session lifecycle ───────────────────────────────────────────── */

function currentConfig() {
  return {
    policy: qs("#ctl-policy").value,
    budget: parseInt(qs("#ctl-budget").value, 10),
    num_sink_tokens: parseInt(qs("#ctl-sinks").value, 10),
    max_reply_tokens: parseInt(qs("#ctl-maxreply").value, 10),
    temperature: parseFloat(qs("#ctl-temp").value),
  };
}

async function newSession() {
  session = await api("/api/sessions", {
    method: "POST",
    body: JSON.stringify({ config: currentConfig() }),
  });
  render(session, null);
}

async function applyConfig() {
  if (!session) return;
  session = await api(`/api/sessions/${session.session_id}/config`, {
    method: "POST",
    body: JSON.stringify(currentConfig()),
  });
  render(session, null);
}

async function forceEvict() {
  if (!session || busy) return;
  const out = await api(`/api/sessions/${session.session_id}/evict`, { method: "POST" });
  session = out.state;
  render(session, null);
  if (out.evicted) {
    qs("#stats-strip").innerHTML =
      `forced eviction: <b>${out.evicted_tokens}</b> tokens dropped` +
      (out.recompute ? " (recompute: fresh cache next turn)"
                     : " — <span class='rot'>rotation plan ships with the next turn</span>");
  }
}

/* ── turns ───────────────────────────────────────────────────────── */

async function sendTurn(text) {
  if (!session || busy) return;
  busy = true;
  qs("#send-btn").disabled = true;
  appendPending(text);
  try {
    const out = await api(`/api/sessions/${session.session_id}/turns`, {
      method: "POST",
      body: JSON.stringify({ text }),
    });
    session = out.state;
    render(session, out.stats);
  } catch (e) {
    qs("#stats-strip").innerHTML = `<span style="color:var(--red)">${e.message}</span>`;
    removePending();
  } finally {
    busy = false;
    qs("#send-btn").disabled = false;
  }
}

function appendPending(text) {
  const scroll = qs("#chat-scroll");
  const el = document.createElement("div");
  el.className = "msg user";
  el.id = "pending-msg";
  el.innerHTML = `<div class="who">you</div><div class="body"></div>`;
  el.querySelector(".body").textContent = text;
  scroll.appendChild(el);
  const wait = document.createElement("div");
  wait.className = "msg model";
  wait.id = "pending-wait";
  wait.innerHTML = `<div class="who">…</div><div class="body">thinking…</div>`;
  scroll.appendChild(wait);
  scroll.scrollTop = scroll.scrollHeight;
}
function removePending() {
  for (const id of ["pending-msg", "pending-wait"]) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }
}

/* ── rendering ───────────────────────────────────────────────────── */

function render(state, stats) {
  removePending();
  qs("#policy-tag").textContent = state.config.policy;

  // chat
  const scroll = qs("#chat-scroll");
  scroll.innerHTML = "";
  if (state.turns.length <= 1) {
    scroll.innerHTML = `<div class="chat-empty">start a conversation below</div>`;
  }
  for (const t of state.turns) {
    const el = document.createElement("div");
    el.className = `msg ${t.role}${t.evicted ? " evicted" : ""}`;
    const who = t.role === "user" ? "you" : t.role === "model" ? state.bot_name : "scene";
    const toks = t.evicted ? t.original_tokens : t.live_tokens;
    el.innerHTML = `<div class="who">${who} <span class="toks">${toks} tok</span></div>` +
                   `<div class="body"></div>`;
    // collapse huge seeded preambles: show head + tail around a fold note
    const body = el.querySelector(".body");
    if (t.text.length > 1200) {
      body.textContent =
        t.text.slice(0, 400) +
        `\n\n··· [${toks} tokens of seeded context collapsed] ···\n\n` +
        t.text.slice(-300);
      body.title = "seeded context (collapsed for display; fully in the model's cache)";
    } else {
      body.textContent = t.text;
    }
    scroll.appendChild(el);
  }
  scroll.scrollTop = scroll.scrollHeight;

  // meter
  const live = state.live_tokens, budget = state.config.budget;
  const pct = Math.min(100, (100 * live) / budget);
  const fill = qs("#meter-fill");
  fill.style.width = pct + "%";
  fill.className = "meter-fill" + (pct > 90 ? " hot" : pct > 70 ? " warm" : "");
  qs("#meter-live").textContent = `${live} tok`;
  qs("#meter-budget").textContent = `budget ${budget}`;

  // eviction events
  const ev = qs("#events-list");
  if (state.events.length === 0) {
    ev.innerHTML = `<div class="chat-empty">none yet</div>`;
  } else {
    ev.innerHTML = state.events
      .map(
        (e) =>
          `<div class="event-row"><b>${e.evicted_tokens}</b> tok, turns ` +
          `[${e.turn_indices.join(", ")}] <span class="mode">${
            e.recompute ? "recompute" : "rotate"
          }</span></div>`
      )
      .join("");
  }

  // per-turn stats
  const totalEvicted = state.events.reduce((a, e) => a + e.evicted_tokens, 0);
  qs("#stat-evicted").textContent = totalEvicted;
  if (stats) {
    qs("#stat-store").textContent = stats.store_tokens || "—";
    qs("#stat-claim").textContent = stats.claimed_tokens || "0";
    qs("#stat-turnms").textContent = stats.wall_s;
    const bits = [
      `turn <b>${stats.wall_s}s</b>`,
      `claimed <b>${stats.claimed_tokens}</b>/${stats.prompt_tokens} prompt tok`,
      `${stats.gen_tokens} generated`,
    ];
    if (stats.evicted_turns > 0) {
      bits.push(
        `<span class="rot">evicted ${stats.evicted_turns} turns ` +
        `(${stats.evicted_tokens} tok) + re-rotated survivors</span>`
      );
    }
    qs("#stats-strip").innerHTML = bits.join(" · ");
  }
}

/* ── wiring ──────────────────────────────────────────────────────── */

qs("#chat-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const box = qs("#chat-input");
  const text = box.value.trim();
  if (!text) return;
  box.value = "";
  sendTurn(text);
});
qs("#chat-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    qs("#chat-form").requestSubmit();
  }
});
async function loadSeedOptions() {
  try {
    const o = await api("/api/seed_options");
    qs("#seed-template").innerHTML = o.templates
      .map((t) => `<option value="${t.id}">${t.label}</option>`).join("");
    qs("#seed-doc").innerHTML =
      `<option value="">longest doc</option>` +
      o.docs.map((d) =>
        `<option value="${d.index}">#${d.index} ~${d.approx_tokens} tok — ${d.preview}</option>`
      ).join("");
  } catch {}
}

async function seedSession() {
  if (!session || busy) return;
  busy = true;
  qs("#seed-btn").disabled = true;
  qs("#stats-strip").innerHTML = "seeding… (tokenizing + building turns)";
  try {
    const docSel = qs("#seed-doc").value;
    const out = await api(`/api/sessions/${session.session_id}/seed`, {
      method: "POST",
      body: JSON.stringify({
        template: qs("#seed-template").value,
        doc_index: docSel === "" ? null : parseInt(docSel, 10),
        target_tokens: parseInt(qs("#seed-tokens").value, 10),
      }),
    });
    session = out.state;
    render(session, null);
    qs("#needle-list").innerHTML = out.needles
      .map((n) =>
        `<div class="event-row">d=${n.depth} <b>${n.code}</b> — ` +
        `<span class="mode">${n.probe}</span></div>`
      ).join("");
    qs("#stats-strip").innerHTML =
      `seeded <b>${session.live_tokens}</b> tokens as evictable turns — ` +
      `probe the planted facts as the session rolls`;
  } catch (e) {
    qs("#stats-strip").innerHTML = `<span style="color:var(--red)">${e.message}</span>`;
  } finally {
    busy = false;
    qs("#seed-btn").disabled = false;
  }
}

qs("#seed-btn").addEventListener("click", seedSession);
loadSeedOptions();

qs("#fork-btn").addEventListener("click", async () => {
  if (!session || busy) return;
  const f = await api(`/api/sessions/${session.session_id}/fork`, { method: "POST" });
  const u = new URL(location.href);
  u.searchParams.set("session", f.session_id);
  window.open(u.toString(), "_blank");
  qs("#stats-strip").innerHTML =
    `forked → <b>${f.session_id}</b> (opened in new tab; branches are independent)`;
});

qs("#reroll-btn").addEventListener("click", async () => {
  if (!session || busy) return;
  busy = true; qs("#reroll-btn").disabled = true;
  qs("#stats-strip").innerHTML = "rerolling last reply…";
  try {
    const out = await api(`/api/sessions/${session.session_id}/reroll`, { method: "POST" });
    session = out.state;
    render(session, out.stats);
  } catch (e) {
    qs("#stats-strip").innerHTML = `<span style="color:var(--red)">${e.message}</span>`;
  } finally { busy = false; qs("#reroll-btn").disabled = false; }
});

qs("#export-btn").addEventListener("click", () => {
  if (!session) return;
  const a = document.createElement("a");
  a.href = `/api/sessions/${session.session_id}/export` + (TOKEN ? `?token=${TOKEN}` : "");
  a.download = `kvrot-${session.session_id}.json`;
  a.click();
});

qs("#import-btn").addEventListener("click", () => qs("#import-file").click());
qs("#import-file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    const data = JSON.parse(await file.text());
    const s = await api("/api/sessions/import", {
      method: "POST", body: JSON.stringify({ export: data }),
    });
    const u = new URL(location.href);
    u.searchParams.set("session", s.session_id);
    location.href = u.toString();
  } catch (err) {
    qs("#stats-strip").innerHTML = `<span style="color:var(--red)">${err.message}</span>`;
  }
});

qs("#apply-btn").addEventListener("click", applyConfig);
qs("#evict-btn").addEventListener("click", forceEvict);
qs("#new-btn").addEventListener("click", newSession);

pollHealth();
setInterval(pollHealth, 10_000);

// ?session=<id> attaches to an existing (e.g. seeded) session instead of
// creating a fresh one
const ATTACH = new URLSearchParams(location.search).get("session");
(ATTACH
  ? api(`/api/sessions/${ATTACH}`).then((s) => {
      session = s;
      // reflect the seeded config in the controls
      qs("#ctl-policy").value = s.config.policy;
      qs("#ctl-budget").value = s.config.budget;
      qs("#ctl-sinks").value = s.config.num_sink_tokens;
      qs("#ctl-maxreply").value = s.config.max_reply_tokens;
      qs("#ctl-temp").value = s.config.temperature;
      render(s, null);
    })
  : newSession()
).catch((e) => {
  qs("#stats-strip").innerHTML = `<span style="color:var(--red)">${e.message}</span>`;
});
