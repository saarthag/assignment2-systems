"use strict";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  schema: { connection: [], benchmark: [], presets: {} },
  connection: {},
  params: {},
  currentRun: null,
  offset: 0,
  pollTimer: null,
  elapsedTimer: null,
  pollMs: 600,
};

const LS_KEY = "cs336-dashboard-connection-v1";
const REQUIRED = new Set(["d_model", "d_ff", "num_layers", "num_heads"]);
const ORDER = [
  "d_model", "d_ff", "num_layers", "num_heads", "vocab_size", "context_length", "theta",
  "batch_size", "compute", "warmup_steps", "bench_steps", "verbose",
];
const LONG_TEXT = new Set(["bench_cmd", "setup_cmd", "extra_args"]);
// Per-field minimums. warmup_steps=0 disables warmup; everything else must be >= 1.
const MIN_VALUE = { warmup_steps: 0 };
// Dashboard defaults that differ from the CLI defaults (the graph needs per-step data).
const BENCH_DEFAULTS = { per_step_timing: true };
const COMPUTE_LABELS = {
  "": "forward only (default)",
  b: "forward + backward",
  bo: "forward + backward + optimizer step",
};

const $ = (sel) => document.querySelector(sel);

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function shellQuote(value) {
  return "'" + String(value).replace(/'/g, "'\\''") + "'";
}

function fmtTime(epochSeconds) {
  if (!epochSeconds) return "";
  const d = new Date(epochSeconds * 1000);
  return d.toLocaleTimeString([], { hour12: false });
}

function fmtWhen(epochSeconds) {
  if (!epochSeconds) return "";
  return new Date(epochSeconds * 1000).toLocaleString();
}

function toast(message, kind) {
  const node = $("#toast");
  node.textContent = message;
  node.className = "toast" + (kind ? " " + kind : "");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => node.classList.add("hidden"), 4200);
}

// ---------------------------------------------------------------------------
// Schema-driven form
// ---------------------------------------------------------------------------
async function loadSchema() {
  const res = await fetch("/api/schema");
  state.schema = await res.json();

  const saved = JSON.parse(localStorage.getItem(LS_KEY) || "{}");
  for (const field of state.schema.connection) {
    const stored = saved[field.name];
    state.connection[field.name] = stored !== undefined ? stored : field.default;
  }
  for (const field of state.schema.benchmark) {
    if (field.name in BENCH_DEFAULTS) state.params[field.name] = BENCH_DEFAULTS[field.name];
    else state.params[field.name] = field.type === "bool" ? !!field.default : field.default;
  }

  renderConnectionFields();
  renderBenchmarkFields();
  renderPresets();
  updatePreview();
}

function renderConnectionFields() {
  const container = $("#connection-fields");
  container.innerHTML = "";
  for (const field of state.schema.connection) {
    const wide = LONG_TEXT.has(field.name);
    const wrap = el("div", "field" + (wide ? " wide" : ""));
    const label = el("label", null, field.label);
    wrap.appendChild(label);

    let input;
    if (field.type === "bool") {
      input = el("input");
      input.type = "checkbox";
      input.checked = !!state.connection[field.name];
      label.prepend(input);
      input.addEventListener("change", () => {
        state.connection[field.name] = input.checked;
        persistConnection();
        updatePreview();
      });
    } else if (LONG_TEXT.has(field.name)) {
      input = el("textarea");
      input.rows = 1;
      input.value = state.connection[field.name] ?? "";
      input.addEventListener("input", () => {
        state.connection[field.name] = input.value;
        persistConnection();
        updatePreview();
      });
    } else {
      input = el("input");
      input.type = field.type === "int" ? "number" : "text";
      if (field.placeholder) input.placeholder = field.placeholder;
      input.value = state.connection[field.name] ?? "";
      input.addEventListener("input", () => {
        state.connection[field.name] = field.type === "int" ? input.value : input.value;
        persistConnection();
        updatePreview();
      });
    }
    if (field.type !== "bool") wrap.appendChild(input);
    if (field.help) wrap.appendChild(el("div", "hint", field.help));
    container.appendChild(wrap);
  }
}

function renderBenchmarkFields() {
  const container = $("#benchmark-fields");
  container.innerHTML = "";
  const fields = [...state.schema.benchmark].sort(
    (a, b) => (ORDER.indexOf(a.name) + 1 || 99) - (ORDER.indexOf(b.name) + 1 || 99)
  );

  for (const field of fields) {
    const wide = field.name === "compute";
    const wrap = el("div", "field" + (wide ? " wide" : ""));
    const label = el("label");
    label.appendChild(document.createTextNode(field.name.replace(/_/g, " ")));
    if (REQUIRED.has(field.name)) label.appendChild(el("span", "req", "*"));
    wrap.appendChild(label);

    if (field.type === "bool") {
      const input = el("input");
      input.type = "checkbox";
      input.checked = !!state.params[field.name];
      label.prepend(input);
      input.addEventListener("change", () => {
        state.params[field.name] = input.checked;
        updatePreview();
      });
    } else if (field.choices) {
      const select = el("select");
      const empty = el("option", null, COMPUTE_LABELS[""] || "default");
      empty.value = "";
      select.appendChild(empty);
      for (const choice of field.choices) {
        const opt = el("option", null, COMPUTE_LABELS[choice] || choice);
        opt.value = choice;
        select.appendChild(opt);
      }
      select.value = state.params[field.name] ?? "";
      select.addEventListener("change", () => {
        state.params[field.name] = select.value;
        updatePreview();
      });
      wrap.appendChild(select);
    } else {
      const input = el("input");
      input.type = field.type === "int" || field.type === "float" ? "number" : "text";
      if (field.type === "float") input.step = "any";
      input.min = String(field.name in MIN_VALUE ? MIN_VALUE[field.name] : 1);
      if (REQUIRED.has(field.name)) input.required = true;
      if (field.default !== null && field.default !== undefined) input.placeholder = String(field.default);
      input.value = state.params[field.name] ?? "";
      input.addEventListener("input", () => {
        state.params[field.name] = input.value;
        updatePreview();
      });
      wrap.appendChild(input);
    }
    if (field.help) wrap.appendChild(el("div", "hint", field.help));
    container.appendChild(wrap);
  }
}

function renderPresets() {
  const container = $("#presets");
  container.innerHTML = "";
  for (const [name, values] of Object.entries(state.schema.presets || {})) {
    const btn = el("button", null, name);
    btn.type = "button";
    btn.title = Object.entries(values).map(([k, v]) => `${k}=${v}`).join(", ");
    btn.addEventListener("click", () => {
      Object.assign(state.params, values);
      renderBenchmarkFields();
      updatePreview();
    });
    container.appendChild(btn);
  }
}

function persistConnection() {
  localStorage.setItem(LS_KEY, JSON.stringify(state.connection));
}

// ---------------------------------------------------------------------------
// Command preview
// ---------------------------------------------------------------------------
function collectParams() {
  const out = {};
  for (const field of state.schema.benchmark) {
    let value = state.params[field.name];
    if (value === "" || value === undefined || value === null) continue;
    out[field.name] = value;
  }
  return out;
}

function buildFlagString() {
  const parts = [];
  for (const field of state.schema.benchmark) {
    const value = state.params[field.name];
    if (field.type === "bool") {
      if (value) parts.push(field.flag);
      continue;
    }
    if (value === "" || value === undefined || value === null) continue;
    parts.push(field.flag, shellQuote(value));
  }
  const extra = (state.connection.extra_args || "").trim();
  if (extra) parts.push(extra);
  return parts.join(" ");
}

function buildPreview() {
  const env = state.connection.env_path || "/venv/main";
  const bench = (state.connection.bench_cmd || "").replaceAll("{env}", env);
  const setup = (state.connection.setup_cmd || "").replaceAll("{env}", env);
  const remoteDir = state.connection.remote_dir || "/workspace/assignment2";
  const flags = buildFlagString();
  const command = [bench, flags].filter(Boolean).join(" ");

  let remote = `cd ${shellQuote(remoteDir)} && export UV_PROJECT_ENVIRONMENT=${shellQuote(env)}`;
  if (setup.trim()) remote += ` && ${setup.trim()}`;
  remote += ` && exec ${command}`;

  const sshParts = ["ssh", "-T", "-p", String(state.connection.port || 22)];
  if (state.connection.batch_mode) sshParts.push("-o", "BatchMode=yes");
  if (state.connection.ssh_key) sshParts.push("-i", state.connection.ssh_key);
  sshParts.push(`${state.connection.user || "root"}@${state.connection.host || "<host>"}`, `\\\n    ${remote}`);

  return sshParts.join(" ") + "\n\n# benchmark command\n" + command;
}

function updatePreview() {
  $("#command-preview").textContent = buildPreview();
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------
async function testConnection() {
  const btn = $("#test-connection");
  const out = $("#connection-output");
  btn.disabled = true;
  btn.textContent = "Testing…";
  out.classList.remove("hidden");
  out.textContent = "running ssh probe…";
  try {
    const res = await fetch("/api/test-connection", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ connection: state.connection }),
    });
    const data = await res.json();
    out.textContent = (data.ok ? "✓ " : "✗ ") + (data.error || "connected") + "\n\n" + (data.output || "");
    setConnStatus(data.ok ? "connected" : "failed");
    toast(data.ok ? "Connection OK" : data.error || "Connection failed", data.ok ? "ok" : "error");
  } catch (err) {
    out.textContent = "probe failed: " + err;
    toast("Probe failed", "error");
  } finally {
    btn.disabled = false;
    btn.textContent = "Test connection";
  }
}

function setConnStatus(status) {
  const pill = $("#conn-status");
  pill.textContent = status;
  pill.className = "pill " + (status === "connected" ? "succeeded" : status === "failed" ? "failed" : "muted");
}

async function startRun(event) {
  event.preventDefault();
  const params = collectParams();
  const missing = [...REQUIRED].filter((name) => !(name in params));
  if (missing.length) {
    toast("Missing required parameters: " + missing.join(", "), "error");
    return;
  }
  if (!state.connection.host) {
    toast("Hostname is required", "error");
    return;
  }
  $("#run-btn").disabled = true;
  try {
    const res = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ connection: state.connection, params }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "failed to start run");
    await refreshHistory();
    selectRun(data.id);
  } catch (err) {
    toast(String(err.message || err), "error");
    $("#run-btn").disabled = false;
  }
}

async function stopRun() {
  if (!state.currentRun) return;
  await fetch(`/api/runs/${state.currentRun}/stop`, { method: "POST" });
  toast("Stopping run…");
}

function selectRun(runId) {
  state.currentRun = runId;
  state.offset = 0;
  state.lastPlotCount = -1;
  if (window.Plotly) Plotly.purge($("#plot"));
  $("#plot").classList.add("hidden");
  $("#plot-empty").classList.remove("hidden");
  $("#plot-summary").textContent = "";
  $("#result-cards").innerHTML = "";
  $("#log").innerHTML = "";
  $("#log-count").textContent = "";
  stopPolling();
  poll();
}

function stopPolling() {
  clearTimeout(state.pollTimer);
  clearInterval(state.elapsedTimer);
  state.pollTimer = null;
  state.elapsedTimer = null;
}

const TERMINAL = new Set(["succeeded", "failed", "stopped", "interrupted"]);

async function poll() {
  if (!state.currentRun) return;
  try {
    const res = await fetch(`/api/runs/${state.currentRun}?offset=${state.offset}`);
    if (res.status === 404) {
      toast("Run disappeared", "error");
      state.currentRun = null;
      return;
    }
    const run = await res.json();
    appendLines(run.lines || []);
    state.offset = run.offset;
    renderRunState(run);

    if (TERMINAL.has(run.status)) {
      stopPolling();
      $("#run-btn").disabled = false;
      refreshHistory();
      return;
    }
    state.pollTimer = setTimeout(poll, state.pollMs);
  } catch (err) {
    state.pollTimer = setTimeout(poll, 1500);
  }
}

function appendLines(lines) {
  const log = $("#log");
  const follow = $("#follow-log").checked;
  const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
  const fragment = document.createDocumentFragment();
  for (const line of lines) {
    fragment.appendChild(renderLine(line));
  }
  log.appendChild(fragment);
  if (follow && atBottom) log.scrollTop = log.scrollHeight;
  $("#log-count").textContent = `${log.childElementCount} line(s)`;
}

function renderLine(line) {
  const text = line.text ?? "";
  const node = el("div", "log-line");
  const ts = el("span", "ts", fmtTime(line.t) + "  ");
  node.appendChild(ts);
  node.appendChild(document.createTextNode(text));

  if (text.startsWith("[dashboard]")) node.classList.add("dashboard");
  else if (/summary:|Per-step time|finished successfully|Constructed TransformerLM/.test(text)) node.classList.add("result");
  else if (/^step \d+:/.test(text)) node.classList.add("step");
  else {
    const match = text.match(/\b(DEBUG|INFO|WARNING|ERROR|CRITICAL)\b/);
    if (match) node.classList.add(match[1].toLowerCase());
  }
  return node;
}

function renderRunState(run) {
  const pill = $("#run-status");
  pill.textContent = run.status;
  pill.className = "pill " + run.status;

  const title = run.params?.d_model
    ? `d_model=${run.params.d_model} · layers=${run.params.num_layers ?? "?"} · heads=${run.params.num_heads ?? "?"} · batch=${run.params.batch_size ?? "?"} · ctx=${run.params.context_length ?? 512}`
    : run.command;
  $("#run-title").textContent = title;

  $("#stop-btn").disabled = TERMINAL.has(run.status);
  $("#run-btn").disabled = !TERMINAL.has(run.status);

  renderResults(run);
  renderPlot(run);
  renderElapsed(run);
  markActiveHistory(run.id);
}

function renderPlot(run) {
  const metrics = run.metrics || [];
  const plotEl = $("#plot");
  const emptyEl = $("#plot-empty");

  if (!window.Plotly) {
    emptyEl.textContent = "Plotly failed to load (no local bundle and no network).";
    emptyEl.classList.remove("hidden");
    plotEl.classList.add("hidden");
    return;
  }
  if (!metrics.length) {
    emptyEl.classList.remove("hidden");
    plotEl.classList.add("hidden");
    return;
  }

  emptyEl.classList.add("hidden");
  plotEl.classList.remove("hidden");

  const x = metrics.map((m) => m.step);
  const y = metrics.map((m) => m.ms);
  const result = run.result || {};
  const mean = typeof result.mean_ms === "number" ? result.mean_ms : null;

  const trace = {
    x,
    y,
    type: "scatter",
    mode: metrics.length > 400 ? "lines" : "lines+markers",
    name: "step time",
    line: { color: "#5b9dff", width: 1.6 },
    marker: { size: 5, color: "#7fb4ff" },
    hovertemplate: "step %{x}<br>%{y:.3f} ms<extra></extra>",
  };

  const shapes = [];
  const annotations = [];
  if (mean !== null) {
    shapes.push({ type: "line", xref: "paper", x0: 0, x1: 1, y0: mean, y1: mean, line: { color: "#3fb950", width: 1.4, dash: "dash" } });
    annotations.push({
      xref: "paper", x: 1, y: mean, xanchor: "right", yanchor: "bottom",
      text: `mean ${mean.toFixed(2)} ms`, showarrow: false, font: { color: "#3fb950", size: 11 },
    });
  }

  const layout = {
    margin: { l: 56, r: 18, t: 8, b: 42 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#8896a8", size: 11 },
    xaxis: { title: { text: "step", font: { size: 11 } }, gridcolor: "#1b2532", zerolinecolor: "#1b2532" },
    yaxis: { title: { text: "time (ms)", font: { size: 11 } }, gridcolor: "#1b2532", zerolinecolor: "#1b2532" },
    shapes,
    annotations,
    showlegend: false,
    hovermode: "closest",
  };

  Plotly.react(plotEl, [trace], layout, { responsive: true, displayModeBar: false });

  const bits = [`n=${metrics.length}`];
  if (mean !== null) bits.push(`mean ${mean.toFixed(2)} ms`);
  if (typeof result.std_ms === "number") bits.push(`std ${result.std_ms.toFixed(2)} ms`);
  $("#plot-summary").textContent = bits.join("  ·  ");
}

function renderResults(run) {
  const cards = [];
  const result = run.result || {};
  const ms = (v) => v.toFixed(3) + " ms";

  // Aggregate per-step statistics (preferred when the per-step driver ran).
  if (typeof result.mean_ms === "number") cards.push(["mean", ms(result.mean_ms), "good"]);
  if (typeof result.std_ms === "number") cards.push(["std dev", ms(result.std_ms), ""]);
  if (typeof result.min_ms === "number") cards.push(["min", ms(result.min_ms), ""]);
  if (typeof result.max_ms === "number") cards.push(["max", ms(result.max_ms), ""]);
  if (typeof result.median_ms === "number") cards.push(["median", ms(result.median_ms), ""]);
  if (typeof result.steps === "number") cards.push(["steps", String(result.steps), ""]);
  // Fallback for the stock benchmark, which only reports an aggregate.
  if (typeof result.per_step_ms === "number" && typeof result.mean_ms !== "number") {
    cards.push(["per-step", ms(result.per_step_ms), "good"]);
  }
  if (result.parameters) cards.push(["params", result.parameters, ""]);
  if (run.started_at && run.ended_at) cards.push(["duration", (run.ended_at - run.started_at).toFixed(1) + " s", ""]);
  if (run.exit_code !== null && run.exit_code !== undefined) cards.push(["exit code", String(run.exit_code), run.exit_code === 0 ? "good" : "bad"]);

  const container = $("#result-cards");
  container.innerHTML = "";
  for (const [k, v, cls] of cards) {
    const card = el("div", "result-card");
    card.appendChild(el("div", "k", k));
    card.appendChild(el("div", "v" + (cls ? " " + cls : ""), v));
    container.appendChild(card);
  }
}

function renderElapsed(run) {
  clearInterval(state.elapsedTimer);
  if (TERMINAL.has(run.status) || !run.started_at) return;
  const update = () => {
    const seconds = (Date.now() / 1000 - run.started_at).toFixed(1);
    let card = document.querySelector('[data-live="elapsed"]');
    if (!card) {
      card = el("div", "result-card");
      card.dataset.live = "elapsed";
      card.appendChild(el("div", "k", "elapsed"));
      card.appendChild(el("div", "v"));
      $("#result-cards").appendChild(card);
    }
    card.querySelector(".v").textContent = seconds + " s";
  };
  update();
  state.elapsedTimer = setInterval(update, 500);
}

// ---------------------------------------------------------------------------
// History
// ---------------------------------------------------------------------------
async function refreshHistory() {
  const res = await fetch("/api/runs");
  const data = await res.json();
  const list = $("#history-list");
  list.innerHTML = "";
  $("#history-count").textContent = `${data.runs.length} run(s)`;
  for (const run of data.runs) {
    list.appendChild(renderHistoryItem(run));
  }
  markActiveHistory(state.currentRun);
}

function renderHistoryItem(run) {
  const item = el("div", "history-item");
  item.dataset.runId = run.id;
  item.appendChild(el("span", "dot " + run.status));

  const meta = el("div", "meta");
  const model = run.params?.d_model
    ? `d_model=${run.params.d_model} L=${run.params.num_layers} H=${run.params.num_heads} b=${run.params.batch_size} c=${run.params.context_length ?? 512}`
    : run.command;
  meta.appendChild(el("div", "name", model));
  const timing = run.result?.per_step_ms !== undefined ? ` · ${run.result.per_step_ms.toFixed(2)} ms` : "";
  meta.appendChild(el("div", "time", `${fmtWhen(run.created_at)} · ${run.status}${timing}`));
  item.appendChild(meta);

  const del = el("button", "del", "✕");
  del.title = "Delete run";
  del.addEventListener("click", async (event) => {
    event.stopPropagation();
    if (!confirm("Delete this run and its log?")) return;
    await fetch(`/api/runs/${run.id}/delete`, { method: "POST" });
    if (state.currentRun === run.id) {
      stopPolling();
      state.currentRun = null;
      state.offset = 0;
      $("#log").innerHTML = "";
      $("#result-cards").innerHTML = "";
      $("#run-title").textContent = "No run selected";
      $("#run-status").textContent = "idle";
      $("#run-status").className = "pill muted";
    }
    refreshHistory();
  });
  item.appendChild(del);

  item.addEventListener("click", () => selectRun(run.id));
  return item;
}

function markActiveHistory(runId) {
  for (const item of document.querySelectorAll(".history-item")) {
    item.classList.toggle("active", item.dataset.runId === runId);
  }
}

// ---------------------------------------------------------------------------
// Log tools
// ---------------------------------------------------------------------------
async function downloadLog() {
  if (!state.currentRun) return;
  const res = await fetch(`/api/runs/${state.currentRun}?offset=0`);
  const run = await res.json();
  const text = (run.lines || []).map((l) => l.text).join("\n");
  const blob = new Blob([text], { type: "text/plain" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `benchmark-${state.currentRun}.log`;
  a.click();
  URL.revokeObjectURL(a.href);
}

function clearView() {
  $("#log").innerHTML = "";
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------
function wire() {
  $("#config-form").addEventListener("submit", startRun);
  $("#test-connection").addEventListener("click", testConnection);
  $("#stop-btn").addEventListener("click", stopRun);
  $("#download-log").addEventListener("click", downloadLog);
  $("#clear-view").addEventListener("click", clearView);
  $("#reset-connection").addEventListener("click", () => {
    localStorage.removeItem(LS_KEY);
    for (const field of state.schema.connection) state.connection[field.name] = field.default;
    renderConnectionFields();
    updatePreview();
    toast("Connection settings reset");
  });
}

async function init() {
  wire();
  try {
    await loadSchema();
    await refreshHistory();
    updatePreview();
  } catch (err) {
    toast("Failed to load schema: " + err, "error");
  }
}

init();
