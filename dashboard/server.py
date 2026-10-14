#!/usr/bin/env python3
"""FastAPI dashboard that drives the CS336 benchmark on a remote GPU box.

The server shells out to the system ``ssh`` client (the same dependency the
``scripts/deploy_*.sh`` helpers use) to connect to a Vast.ai instance and run
``dashboard/bench_driver.py`` (which emits per-step timings) inside the remote
project directory.

Run it with the dashboard dependency group::

    uv run --group dashboard python dashboard/server.py
    uv run --group dashboard python dashboard/server.py --port 9000 --host 0.0.0.0

HTTP API (see ``dashboard/README.md`` for details):

* ``GET  /``                        -> dashboard UI
* ``GET  /api/schema``              -> connection + benchmark option schema
* ``GET  /api/runs``                -> run summaries, newest first
* ``POST /api/runs``                -> start a run
* ``GET  /api/runs/<id>?offset=N``  -> run state, per-step metrics + new logs
* ``POST /api/runs/<id>/stop``      -> stop a running run
* ``POST /api/runs/<id>/delete``    -> remove a run
* ``GET  /api/runs/<id>/report``    -> fetch + download the profiler report
* ``GET  /api/runs/<id>/memory``    -> fetch + download the CUDA memory snapshot
* ``GET  /api/runs/<id>/memory_viz``-> PyTorch MemoryViz viewer for the snapshot
* ``POST /api/runs/<id>/memory/fetch`` -> pre-fetch the snapshot over SSH
* ``POST /api/test-connection``     -> probe ssh connectivity (blocking)
* ``POST /api/sync``                -> push the local working tree to the remote dir
* ``POST /api/runs/<id>/export``    -> zip the log, report and graph for download
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, create_model
from schema import extract_cli_schema

STATIC_DIR = Path(__file__).resolve().parent / "static"
RUNS_DIR = Path(__file__).resolve().parent / ".runs"
# The dashboard lives inside the assignment2 repo, so its parent is the tree
# that ``Sync to remote`` ships (mirrors scripts/deploy_code.sh).
LOCAL_REPO_ROOT = Path(__file__).resolve().parent.parent
SYNC_LOCK = threading.Lock()
DEFAULT_REMOTE_DIR = "/workspace/assignment2"
# Vast.ai's PyTorch image ships a working torch/CUDA stack in /venv/main. We
# reuse it instead of letting uv download the multi-GB torch wheels:
#   * setup  -> scripts/uv_sync_no_torch.sh with `--inexact`, which installs
#               everything *except* torch/triton/cuda-*/nvidia-* and never prunes
#               the image's own libraries (see the script header).
#   * run    -> `uv run --no-sync`, so uv never re-syncs/reinstalls anything.
DEFAULT_ENV_PATH = "/venv/main"
DEFAULT_BENCH_CMD = "uv run --no-sync python -m cs336_systems.benchmark"
DEFAULT_SETUP_CMD = "bash scripts/uv_sync_no_torch.sh"
SSH_MAX_TIMEOUT = 30

# Optional profiler wrapper (Nsight Systems by default). When enabled the
# benchmark command is wrapped as ``<profiler> ... -o <prefix> <command>`` and
# the resulting report is pulled back over SSH on demand.
DEFAULT_PROFILER_CMD = "nsys profile"
# Assignment-suggested Nsight Systems flags (NVTX/PyTorch annotations, CUDA and
# Python backtraces). We additionally force overwrite (so a re-run never blocks
# on nsys' interactive prompt over BatchMode SSH) and disable the host-side perf
# collectors (`--sample=none --cpuctxsw=none`), which cannot be enabled inside an
# unprivileged Vast.ai/Docker container and only add noise.
#
# ``--gpu-metrics-devices=0`` is intentionally omitted. On hosts whose driver
# restricts profiling to admin users (`NVreg_RestrictProfilingToAdminUsers`, e.g.
# many Vast.ai RTX 50xx instances) nsys exits 1 with ``ERR_NVGPUCTRPERM`` and
# aborts the whole run. Add it back per-run only where GPU metrics are allowed.
DEFAULT_PROFILER_ARGS = (
    "--force-overwrite true "
    "--trace=cuda,cudnn,cublas,osrt,nvtx "
    "--pytorch=functions-trace,autograd-shapes-nvtx "
    "--cudabacktrace=all "
    "--python-backtrace=cuda "
    "--sample=none "
    "--cpuctxsw=none"
)
DEFAULT_PROFILER_DIR = "profiles"

# Optional CUDA memory snapshot. When enabled the benchmark CLI is passed
# ``--memory-profile <prefix>.pickle`` and the snapshot is pulled back over SSH
# on demand for the in-dashboard MemoryViz viewer (pytorch.org/memory_viz).
DEFAULT_MEMORY_DIR = "profiles"
DEFAULT_MEMORY_MAX_ENTRIES = 1_000_000

# Official PyTorch memory-viewer bootstrap. The snapshot is inlined as base64 so
# the bundled viewer can decode it without a second request. MemoryViz.js and its
# d3 imports are loaded from jsDelivr (the same modules docs.pytorch.org/memory_viz
# uses); the vendored copy in dashboard/static/memviz pins the viewer itself.
MEMORY_VIZ_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>CUDA memory snapshot</title>
  <style>
    :root { color-scheme: dark; }
    html, body {
      margin: 0;
      height: 100%;
      background: #0d1117;
      color: #c9d1d9;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      font-size: 13px;
    }
    /* Toolbar (snapshot/view/device selects). */
    body > div:first-of-type {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
      padding: 12px 18px;
      background: #131a23;
      border-bottom: 1px solid #243040;
    }
    select, input {
      background: #0a0f16;
      color: #d6e0ec;
      border: 1px solid #243040;
      border-radius: 7px;
      padding: 5px 9px;
      font-size: 12px;
      font-family: inherit;
      outline: none;
    }
    select:hover { border-color: #33414f; }
    select:focus, input:focus { border-color: #5b9dff; box-shadow: 0 0 0 3px rgba(91, 157, 255, 0.15); }
    input[type="checkbox"] { accent-color: #5b9dff; width: auto; }
    input[type="range"] { accent-color: #5b9dff; background: transparent; padding: 0; }
    input[type="text"] { font-family: ui-monospace, Menlo, Consolas, monospace; }
    /* The snapshot is already loaded, so the native file picker is redundant. */
    body > input[type="file"] { display: none; }
    label { color: #8896a8; font-size: 12px; }
    body > label { padding: 12px 18px 0; }
    /* Plot + legend. */
    svg { display: block; }
    svg text { fill: #8b949e; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    svg rect[fill="white"] { fill: #0b1117; }
    .tick line, .domain { stroke: #2a3644; }
    /* Detail slider + search row. */
    .mv-tools {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
      padding: 6px 0 14px;
    }
    .mv-tools input[type="range"] { flex: 1 1 160px; min-width: 120px; }
    .mv-tools label { white-space: nowrap; }
    /* Legend, lifted out of the SVG so it no longer covers the plot. */
    .mv-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 22px;
      padding: 11px 15px;
      margin: 0 0 14px;
      background: #0a0f16;
      border: 1px solid #1b2532;
      border-radius: 9px;
    }
    .mv-legend-item { display: inline-flex; align-items: center; gap: 8px; color: #c9d1d9; font-size: 12px; }
    .mv-legend-item i { width: 11px; height: 11px; border-radius: 3px; box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.14); }
    .mv-viewer { padding: 16px 18px 18px; }
    .mv-plot { border: 1px solid #1b2532; border-radius: 10px; overflow: hidden; }
    .mv-minimap { border: 1px solid #1b2532; border-radius: 10px; overflow: hidden; background: #0b1117; }
    .mv-context { min-height: 0; }
    /* Stack trace / allocator settings panes. */
    pre {
      background: #0a0f16;
      color: #b8c6d8;
      border: 1px solid #1b2532;
      border-radius: 8px;
      padding: 10px 12px;
      font-family: ui-monospace, Menlo, Consolas, monospace;
      font-size: 11.5px;
      line-height: 1.5;
      white-space: pre-wrap;
    }
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #26313f; border-radius: 6px; border: 2px solid #0d1117; }
    ::-webkit-scrollbar-thumb:hover { background: #33414f; }
  </style>
</head>
<body>
<script type="module">
import {add_local_files} from "/static/memviz/MemoryViz.js"
const local_files = $SNAPSHOT
add_local_files(local_files, $VIZ_KIND)

// PyTorch's viewer draws the legend inside the plot SVG at fixed coordinates,
// which overlaps the timeline and reads as cramped. Lift it into an HTML bar
// and give the surrounding controls room to breathe.
const PALETTE = ["#4e79a7", "#f28e2c", "#e15759", "#76b7b2", "#59a14f", "#edc949", "#af7aa1", "#ff9da7", "#9c755f", "#bab0ab"];

function enhance() {
  const grid = [...document.querySelectorAll("div")].find((d) => (d.getAttribute("style") || "").includes("grid-template-rows: 10fr"));
  if (!grid) return false;
  const viewer = grid.parentElement;
  viewer.classList.add("mv-viewer");
  grid.style.gridTemplateRows = "14fr 0.9fr 4fr";
  const tools = grid.previousElementSibling;
  if (tools) {
    tools.classList.add("mv-tools");
    [...tools.querySelectorAll("span")].forEach((s) => {
      if (s.textContent.trim() === "|") s.style.display = "none";
    });
  }
  if (grid.children[0]) grid.children[0].classList.add("mv-plot");
  if (grid.children[1]) grid.children[1].classList.add("mv-minimap");
  if (grid.children[2]) grid.children[2].classList.add("mv-context");

  const textGs = [...document.querySelectorAll("svg g")].filter((g) => g.querySelector('text[font-family="helvetica"]'));
  const rectGs = [...document.querySelectorAll("svg g")].filter((g) => g.querySelector('rect[width="10"][height="10"][x="100"]'));
  if (textGs.length && rectGs.length) {
    const textG = textGs[textGs.length - 1];
    const rectG = rectGs[rectGs.length - 1];
    const labels = [...textG.querySelectorAll("text")].map((t) => t.textContent);
    const fills = [...rectG.querySelectorAll("rect")].map((r) => r.getAttribute("fill"));
    textG.style.display = "none";
    rectG.style.display = "none";
    if (labels.length && !viewer.querySelector(":scope > .mv-legend")) {
      const bar = document.createElement("div");
      bar.className = "mv-legend";
      labels.forEach((label, i) => {
        const item = document.createElement("span");
        item.className = "mv-legend-item";
        const swatch = document.createElement("i");
        swatch.style.background = fills[i] || PALETTE[i % PALETTE.length];
        item.appendChild(swatch);
        item.append(label);
        bar.appendChild(item);
      });
      viewer.insertBefore(bar, grid);
    }
  }
  return true;
}

enhance();
let attempts = 0;
const timer = setInterval(() => {
  if (enhance() || ++attempts > 25) clearInterval(timer);
}, 200);
new MutationObserver(() => queueMicrotask(enhance)).observe(document.body, { childList: true, subtree: true });
</script>
</body>
</html>
"""

BENCH_FALLBACK = "uv run --no-sync python -m cs336_systems.benchmark"

# Connection fields surfaced in the UI. The pydantic request model is generated
# from this list so there is a single source of truth.
CONNECTION_FIELDS: list[dict[str, Any]] = [
    {"name": "host", "label": "Hostname", "type": "str", "default": "", "placeholder": "ssh4.vast.ai", "help": "Vast.ai instance host (the part before :port)."},
    {"name": "port", "label": "SSH port", "type": "int", "default": 22, "placeholder": "12345", "help": "Mapped SSH port for the instance."},
    {"name": "user", "label": "User", "type": "str", "default": "root", "placeholder": "root", "help": "Remote user."},
    {
        "name": "ssh_key",
        "label": "SSH key",
        "type": "str",
        "default": "",
        "placeholder": "~/.ssh/id_ed25519",
        "help": "Optional identity file. Blank uses your ssh config / agent.",
    },
    {
        "name": "remote_dir",
        "label": "Remote dir",
        "type": "str",
        "default": DEFAULT_REMOTE_DIR,
        "placeholder": DEFAULT_REMOTE_DIR,
        "help": "Directory containing the assignment2 tree on the instance.",
    },
    {
        "name": "env_path",
        "label": "Remote uv env",
        "type": "str",
        "default": DEFAULT_ENV_PATH,
        "placeholder": DEFAULT_ENV_PATH,
        "help": (
            "Exported as UV_PROJECT_ENVIRONMENT on the instance (before setup + benchmark). Vast.ai PyTorch images use /venv/main, which keeps the preinstalled torch/CUDA stack."
        ),
    },
    {
        "name": "bench_cmd",
        "label": "Benchmark command",
        "type": "str",
        "default": DEFAULT_BENCH_CMD,
        "placeholder": DEFAULT_BENCH_CMD,
        "help": "Runs inside the remote dir. `--no-sync` guarantees uv will not reinstall dependencies. The per-step timing checkbox appends --per-step-timing to this command.",
    },
    {
        "name": "setup_cmd",
        "label": "Setup command",
        "type": "str",
        "default": DEFAULT_SETUP_CMD,
        "placeholder": DEFAULT_SETUP_CMD,
        "help": "Run before every benchmark. Defaults to uv_sync_no_torch.sh, which skips torch/triton/cuda/nvidia and never prunes the image env.",
    },
    {"name": "extra_args", "label": "Extra CLI args", "type": "str", "default": "", "placeholder": "--batch-size 8", "help": "Raw extra flags appended to the benchmark command."},
    {"name": "batch_mode", "label": "Non-interactive SSH", "type": "bool", "default": True, "help": "Adds -o BatchMode=yes so SSH never hangs waiting for a password."},
]

# Profiler settings. These are not part of the benchmark CLI; they control how
# the remote command is wrapped and where the report is written.
PROFILING_FIELDS: list[dict[str, Any]] = [
    {
        "name": "enabled",
        "label": "Profile with Nsight Systems (nsys)",
        "type": "bool",
        "default": False,
        "help": "Wrap the benchmark command in the profiler and write a report on the instance for download.",
    },
    {
        "name": "profiler_cmd",
        "label": "Profiler command",
        "type": "str",
        "default": DEFAULT_PROFILER_CMD,
        "placeholder": DEFAULT_PROFILER_CMD,
        "help": "Wrapper prepended to the benchmark command. `nsys profile` for Nsight Systems, or `ncu` for Nsight Compute.",
    },
    {
        "name": "profiler_args",
        "label": "Profiler args",
        "type": "str",
        "default": DEFAULT_PROFILER_ARGS,
        "placeholder": DEFAULT_PROFILER_ARGS,
        "help": (
            "Extra profiler flags. Defaults to the assignment-suggested nsys flags plus --sample=none "
            "--cpuctxsw=none (the host-side perf collectors can't run in an unprivileged container). "
            "`-o <report>` is appended automatically."
        ),
    },
    {
        "name": "report_dir",
        "label": "Remote report dir",
        "type": "str",
        "default": DEFAULT_PROFILER_DIR,
        "placeholder": DEFAULT_PROFILER_DIR,
        "help": "Where the report is written on the instance (relative to the remote dir, or absolute).",
    },
]

# CUDA memory-snapshot settings. These map onto the benchmark CLI's
# ``--memory-profile`` flag; the dashboard owns the output path so it can pull
# the snapshot back for the MemoryViz panel.
MEMORY_FIELDS: list[dict[str, Any]] = [
    {
        "name": "enabled",
        "label": "Record CUDA memory snapshot",
        "type": "bool",
        "default": False,
        "help": "Run the benchmark with --memory-profile and view the timeline with PyTorch's memory_viz viewer.",
    },
    {
        "name": "snapshot_dir",
        "label": "Remote snapshot dir",
        "type": "str",
        "default": DEFAULT_MEMORY_DIR,
        "placeholder": DEFAULT_MEMORY_DIR,
        "help": "Where the snapshot is written on the instance (relative to the remote dir, or absolute).",
    },
    {
        "name": "max_entries",
        "label": "Max history entries",
        "type": "int",
        "default": DEFAULT_MEMORY_MAX_ENTRIES,
        "placeholder": str(DEFAULT_MEMORY_MAX_ENTRIES),
        "help": "Allocator history ring-buffer size. Larger captures more steps but makes the snapshot (and SSH transfer) bigger.",
    },
]

MODEL_FIELDS = {"d_model", "d_ff", "num_layers", "num_heads"}
RUN_FIELDS = {"warmup_steps", "bench_steps", "verbose"}
PARAM_PRESETS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10B": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}

_TYPE_MAP = {"str": str, "int": int, "bool": bool}
ConnectionConfig = create_model(
    "ConnectionConfig",
    **{field["name"]: (_TYPE_MAP[field["type"]], field["default"]) for field in CONNECTION_FIELDS},
)
ProfilingConfig = create_model(
    "ProfilingConfig",
    **{field["name"]: (_TYPE_MAP[field["type"]], field["default"]) for field in PROFILING_FIELDS},
)
MemoryConfig = create_model(
    "MemoryConfig",
    **{field["name"]: (_TYPE_MAP[field["type"]], field["default"]) for field in MEMORY_FIELDS},
)


class ConnectionRequest(BaseModel):
    connection: ConnectionConfig


class RunRequest(BaseModel):
    connection: ConnectionConfig
    params: dict[str, Any] = Field(default_factory=dict)
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)


class ExportRequest(BaseModel):
    # Optional base64 data URL of the graphed PNG, rendered client-side by Plotly.
    graph_png: str | None = None


def _coerce(field: dict[str, Any], value: Any) -> Any:
    if value is None or value == "":
        return None
    kind = field.get("type")
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    if kind == "bool":
        return bool(value)
    return str(value)


def build_remote_command(bench_cmd: str, params: dict[str, Any], schema: list[dict[str, Any]], extra_args: str = "") -> str:
    """Turn the form values into the shell command executed on the remote."""
    parts = [bench_cmd.strip() or BENCH_FALLBACK]
    for field in schema:
        value = params.get(field["name"])
        if field["type"] == "bool":
            if value:
                parts.append(field["flag"])
            continue
        if value is None or value == "":
            continue
        parts += [field["flag"], shlex.quote(str(value))]
    if extra_args.strip():
        parts.append(extra_args.strip())
    return " ".join(parts)


def expand_env(template: str, env_path: str) -> str:
    """Substitute the legacy ``{env}`` placeholder with the configured env."""
    return template.replace("{env}", env_path or DEFAULT_ENV_PATH)


def build_full_remote(remote_dir: str, env_path: str, setup_cmd: str, command: str) -> str:
    """Assemble the remote shell pipeline.

    The env var is exported up-front (``export`` accepts assignment words, whereas
    ``exec VAR=value cmd`` would be parsed as running a file named ``VAR=value``).
    """
    remote = f"cd {shlex.quote(remote_dir)} && export UV_PROJECT_ENVIRONMENT={shlex.quote(env_path or DEFAULT_ENV_PATH)}"
    if setup_cmd.strip():
        remote += f" && {setup_cmd.strip()}"
    remote += f" && exec {command}"
    return remote


def build_ssh_argv(connection: dict[str, Any], remote_command: str) -> list[str]:
    host = str(connection.get("host") or "").strip()
    user = str(connection.get("user") or "root").strip() or "root"
    port = int(connection.get("port") or 22)
    ssh_key = str(connection.get("ssh_key") or "").strip()

    argv = [
        "ssh",
        "-T",
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "LogLevel=ERROR",
    ]
    if connection.get("batch_mode", True):
        argv += ["-o", "BatchMode=yes"]
    if ssh_key:
        argv += ["-i", os.path.expanduser(ssh_key)]
    argv += [f"{user}@{host}", remote_command]
    return argv


def _display_ssh(argv: list[str]) -> str:
    return " ".join(shlex.quote(a) for a in argv)


def build_profiler_prefix(profiling: dict[str, Any], output_prefix: str) -> str:
    """Build the profiler wrapper, e.g. ``nsys profile --force-overwrite true -o <prefix>``."""
    profiler = str(profiling.get("profiler_cmd") or DEFAULT_PROFILER_CMD).strip() or DEFAULT_PROFILER_CMD
    args = str(profiling.get("profiler_args") or "").strip()
    parts = [profiler]
    if args:
        parts.append(args)
    parts += ["-o", shlex.quote(output_prefix)]
    return " ".join(parts)


def report_extension(profiler_cmd: str) -> str:
    """Nsight Compute writes ``.ncu-rep``; Nsight Systems writes ``.nsys-rep``."""
    return ".ncu-rep" if "ncu" in (profiler_cmd or "") else ".nsys-rep"


_STEP_DISPLAY = re.compile(r"^step (\d+): ([\d.]+) ms$")


class Run:
    """A single benchmark invocation, its logs and its per-step metrics."""

    def __init__(
        self,
        run_id: str,
        connection: dict[str, Any],
        params: dict[str, Any],
        command: str,
        remote_command: str,
        argv: list[str],
        profiling: dict[str, Any] | None = None,
        report_path: str = "",
        memory: dict[str, Any] | None = None,
        memory_path: str = "",
    ):
        self.id = run_id
        self.created_at = time.time()
        self.started_at: float | None = None
        self.ended_at: float | None = None
        self.status = "starting"
        self.exit_code: int | None = None
        self.error: str | None = None
        self.connection = connection
        self.params = params
        self.command = command
        self.remote_command = remote_command
        self.argv = argv
        self.profiling = dict(profiling or {})
        self.report_path = report_path or ""
        self.memory = dict(memory or {})
        self.memory_path = memory_path or ""
        self.logs: list[dict[str, Any]] = []
        self.metrics: list[dict[str, Any]] = []
        self.result: dict[str, Any] = {}
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._dir = RUNS_DIR / run_id
        self._dir.mkdir(parents=True, exist_ok=True)
        self._log_fh = (self._dir / "output.jsonl").open("a", encoding="utf-8")

    # -- log handling ---------------------------------------------------
    @staticmethod
    def _ingest(raw: str) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
        """Map a raw stdout line to (display text, metric point, summary)."""
        if raw.startswith("{"):
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                event = None
            if isinstance(event, dict) and event.get("event") == "step":
                step_no = int(event.get("step", 0))
                ms = float(event.get("ms", 0.0))
                return f"step {step_no}: {ms:.3f} ms", {"step": step_no, "ms": ms}, None
            if isinstance(event, dict) and event.get("event") == "summary":
                summary = {k: v for k, v in event.items() if k != "event"}
                display = f"summary: mean={summary.get('mean_ms', 0.0):.3f} ms  std={summary.get('std_ms', 0.0):.3f} ms  (n={summary.get('steps', 0)})"
                return display, None, summary
        match = _STEP_DISPLAY.match(raw)
        if match:
            return raw, {"step": int(match.group(1)), "ms": float(match.group(2))}, None
        return raw, None, None

    @staticmethod
    def _parse_text_metric(text: str, result: dict[str, Any]) -> None:
        m = re.search(r"Per-step time:\s*([\d.]+)\s*ms", text)
        if m:
            result["per_step_ms"] = float(m.group(1))
        m = re.search(r"Constructed TransformerLM with ([\d.]+) (\w+) parameters", text)
        if m:
            result["parameters"] = f"{m.group(1)} {m.group(2)}"
        m = re.search(r"Benchmarking (\d+) step", text)
        if m:
            result["bench_steps"] = int(m.group(1))

    def ingest(self, raw: str) -> None:
        """Process one raw line from the remote process (live or replayed)."""
        raw = raw.rstrip("\n")
        display, metric, summary = self._ingest(raw)
        with self._lock:
            if metric is not None:
                self.metrics.append(metric)
            if summary is not None:
                self.result.update(summary)
            self._parse_text_metric(raw, self.result)
            entry = {"t": time.time(), "text": display}
            self.logs.append(entry)
            self._log_fh.write(json.dumps(entry) + "\n")
            self._log_fh.flush()

    # -- serialization --------------------------------------------------
    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status,
            "exit_code": self.exit_code,
            "error": self.error,
            "command": self.command,
            "remote_command": self.remote_command,
            "params": self.params,
            "result": self.result,
            "profiling": self.profiling,
            "report": self.report_info(),
            "memory": self.memory_info(),
            "log_count": len(self.logs),
            "metric_count": len(self.metrics),
        }

    def to_dict(self, offset: int = 0) -> dict[str, Any]:
        with self._lock:
            lines = self.logs[offset:]
            total = len(self.logs)
            metrics = list(self.metrics)
        data = self.summary()
        data["lines"] = lines
        data["offset"] = total
        data["metrics"] = metrics
        return data

    def save_meta(self) -> None:
        meta = self.summary()
        meta["argv"] = self.argv
        meta["connection"] = self.connection
        meta["profiling"] = self.profiling
        meta["report_path"] = self.report_path
        meta["memory"] = self.memory
        meta["memory_path"] = self.memory_path
        (self._dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self._log_fh.close()

    # -- profiler report -----------------------------------------------
    def report_local_path(self) -> Path:
        return self._dir / (Path(self.report_path).name or "report.nsys-rep")

    def report_info(self) -> dict[str, Any]:
        enabled = bool(self.profiling.get("enabled")) and bool(self.report_path)
        return {
            "enabled": enabled,
            "remote_path": self.report_path or None,
            "name": Path(self.report_path).name if self.report_path else None,
            "downloaded": enabled and self.report_local_path().exists(),
        }

    def fetch_report(self) -> tuple[bool, str | None]:
        """Copy the profiler report from the instance over SSH (cached locally)."""
        if not (self.profiling.get("enabled") and self.report_path):
            return False, "profiling was not enabled for this run"
        local = self.report_local_path()
        if local.exists() and local.stat().st_size > 0:
            return True, None
        remote_dir = str(self.connection.get("remote_dir") or DEFAULT_REMOTE_DIR)
        remote = f"cd {shlex.quote(remote_dir)} && test -f {shlex.quote(self.report_path)} && cat {shlex.quote(self.report_path)}"
        argv = build_ssh_argv(self.connection, remote)
        self.ingest(f"[dashboard] fetching profiler report: {self.report_path}")
        try:
            with local.open("wb") as fh:
                proc = subprocess.run(argv, stdout=fh, stderr=subprocess.PIPE, check=False)
        except FileNotFoundError:
            return False, "ssh executable not found on PATH"
        if proc.returncode != 0:
            local.unlink(missing_ok=True)
            detail = (proc.stderr or b"").decode(errors="replace").strip()
            return False, detail or f"ssh exited with code {proc.returncode}"
        if not local.exists() or local.stat().st_size == 0:
            local.unlink(missing_ok=True)
            return False, "remote report is empty or missing"
        return True, None

    # -- CUDA memory snapshot ------------------------------------------
    def memory_local_path(self) -> Path:
        return self._dir / (Path(self.memory_path).name or "memory_snapshot.pickle")

    def memory_info(self) -> dict[str, Any]:
        enabled = bool(self.memory.get("enabled")) and bool(self.memory_path)
        return {
            "enabled": enabled,
            "remote_path": self.memory_path or None,
            "name": Path(self.memory_path).name if self.memory_path else None,
            "downloaded": enabled and self.memory_local_path().exists(),
        }

    def fetch_memory(self) -> tuple[bool, str | None]:
        """Copy the CUDA memory snapshot from the instance over SSH (cached)."""
        if not (self.memory.get("enabled") and self.memory_path):
            return False, "memory profiling was not enabled for this run"
        local = self.memory_local_path()
        if local.exists() and local.stat().st_size > 0:
            return True, None
        remote_dir = str(self.connection.get("remote_dir") or DEFAULT_REMOTE_DIR)
        remote = f"cd {shlex.quote(remote_dir)} && test -f {shlex.quote(self.memory_path)} && cat {shlex.quote(self.memory_path)}"
        argv = build_ssh_argv(self.connection, remote)
        self.ingest(f"[dashboard] fetching memory snapshot: {self.memory_path}")
        try:
            with local.open("wb") as fh:
                proc = subprocess.run(argv, stdout=fh, stderr=subprocess.PIPE, check=False)
        except FileNotFoundError:
            return False, "ssh executable not found on PATH"
        if proc.returncode != 0:
            local.unlink(missing_ok=True)
            detail = (proc.stderr or b"").decode(errors="replace").strip()
            return False, detail or f"ssh exited with code {proc.returncode}"
        if not local.exists() or local.stat().st_size == 0:
            local.unlink(missing_ok=True)
            return False, "remote memory snapshot is empty or missing"
        return True, None

    @classmethod
    def load(cls, run_dir: Path) -> Run | None:
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        run = cls(
            run_dir.name,
            meta.get("connection", {}),
            meta.get("params", {}),
            meta.get("command", ""),
            meta.get("remote_command", ""),
            meta.get("argv", []),
            meta.get("profiling", {}),
            meta.get("report_path", ""),
            meta.get("memory", {}),
            meta.get("memory_path", ""),
        )
        run.created_at = meta.get("created_at", run.created_at)
        run.started_at = meta.get("started_at")
        run.ended_at = meta.get("ended_at")
        run.status = meta.get("status", "unknown")
        run.exit_code = meta.get("exit_code")
        run.error = meta.get("error")
        run.result = meta.get("result", {})
        if run.status in {"starting", "running"}:
            run.status = "interrupted"
        # Replay persisted logs to rebuild per-step metrics.
        log_path = run_dir / "output.jsonl"
        if log_path.exists():
            for line in log_path.read_text(encoding="utf-8").splitlines():
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    entry = {"t": 0, "text": line}
                raw = entry.get("text", "")
                _, metric, _ = run._ingest(raw)
                if metric is not None:
                    run.metrics.append(metric)
                run.logs.append({"t": entry.get("t", 0), "text": raw})
        return run


class RunStore:
    def __init__(self) -> None:
        self._runs: dict[str, Run] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        for run_dir in sorted(RUNS_DIR.iterdir()):
            if run_dir.is_dir():
                run = Run.load(run_dir)
                if run:
                    self._runs[run.id] = run

    def add(self, run: Run) -> None:
        with self._lock:
            self._runs[run.id] = run
        run.save_meta()

    def get(self, run_id: str) -> Run | None:
        with self._lock:
            return self._runs.get(run_id)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            runs = list(self._runs.values())
        runs.sort(key=lambda r: r.created_at, reverse=True)
        return [r.summary() for r in runs]

    def delete(self, run_id: str) -> bool:
        with self._lock:
            run = self._runs.pop(run_id, None)
        if not run:
            return False
        import shutil

        run.close()
        shutil.rmtree(run._dir, ignore_errors=True)
        return True


class Runner:
    """Spawns ssh and pumps the merged stdout/stderr into a :class:`Run`."""

    @staticmethod
    def start(store: RunStore, connection: dict[str, Any], params: dict[str, Any], profiling: dict[str, Any] | None = None, memory: dict[str, Any] | None = None) -> Run:
        profiling = dict(profiling or {})
        memory = dict(memory or {})
        schema = extract_cli_schema()
        env_path = str(connection.get("env_path") or DEFAULT_ENV_PATH)
        command = build_remote_command(
            expand_env(str(connection.get("bench_cmd") or DEFAULT_BENCH_CMD), env_path),
            params,
            schema,
            str(connection.get("extra_args") or ""),
        )
        remote_dir = str(connection.get("remote_dir") or DEFAULT_REMOTE_DIR)
        setup = expand_env(str(connection.get("setup_cmd") or ""), env_path)

        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        report_path = ""
        memory_path = ""
        setup_prepend: list[str] = []
        if memory.get("enabled"):
            snapshot_dir = str(memory.get("snapshot_dir") or DEFAULT_MEMORY_DIR).strip() or DEFAULT_MEMORY_DIR
            setup_prepend.append(f"mkdir -p {shlex.quote(snapshot_dir)}")
            snapshot_prefix = f"{snapshot_dir.rstrip('/')}/{run_id}"
            memory_path = f"{snapshot_prefix}.pickle"
            max_entries = int(memory.get("max_entries") or DEFAULT_MEMORY_MAX_ENTRIES)
            command = f"{command} --memory-profile {shlex.quote(memory_path)} --memory-profile-max-entries {max_entries}"
        if profiling.get("enabled"):
            report_dir = str(profiling.get("report_dir") or DEFAULT_PROFILER_DIR).strip() or DEFAULT_PROFILER_DIR
            setup_prepend.append(f"mkdir -p {shlex.quote(report_dir)}")
            report_prefix = f"{report_dir.rstrip('/')}/{run_id}"
            command = f"{build_profiler_prefix(profiling, report_prefix)} {command}"
            report_path = f"{report_prefix}{report_extension(str(profiling.get('profiler_cmd') or ''))}"
        if setup_prepend:
            prefix = " && ".join(dict.fromkeys(setup_prepend))
            setup = f"{prefix} && {setup}" if setup.strip() else prefix

        remote_command = build_full_remote(remote_dir, env_path, setup, command)
        argv = build_ssh_argv(connection, remote_command)

        run = Run(run_id, dict(connection), dict(params), command, remote_command, argv, profiling, report_path, memory, memory_path)
        store.add(run)
        threading.Thread(target=Runner._run, args=(run,), daemon=True).start()
        return run

    @staticmethod
    def _run(run: Run) -> None:
        run.ingest("[dashboard] " + _display_ssh(run.argv))
        try:
            run._proc = subprocess.Popen(
                run.argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            run.status = "failed"
            run.error = "ssh executable not found on PATH"
            run.ingest("[dashboard] error: " + run.error)
            run.ended_at = time.time()
            run.save_meta()
            return

        run.status = "running"
        run.started_at = time.time()
        run.save_meta()

        assert run._proc.stdout is not None
        for line in run._proc.stdout:
            run.ingest(line)

        code = run._proc.wait()
        run.exit_code = code
        run.ended_at = time.time()
        if run.status == "stopping":
            run.status = "stopped"
            run.ingest("[dashboard] run stopped")
        elif code == 0:
            run.status = "succeeded"
            run.ingest("[dashboard] finished successfully")
        else:
            run.status = "failed"
            run.error = f"process exited with code {code}"
            run.ingest(f"[dashboard] failed (exit code {code})")
        run.save_meta()

    @staticmethod
    def stop(run: Run) -> None:
        proc = run._proc
        if not proc or proc.poll() is not None:
            return
        run.status = "stopping"
        run.ingest("[dashboard] stopping...")
        with contextlib.suppress(OSError):
            proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


STORE = RunStore()
SCHEMA = extract_cli_schema()

try:
    import plotly

    PLOTLY_JS = Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"
except (ImportError, AttributeError, TypeError):  # pragma: no cover - plotly is an optional dashboard dep
    PLOTLY_JS = None


def probe_connection(connection: dict[str, Any]) -> dict[str, Any]:
    if not str(connection.get("host") or "").strip():
        return {"ok": False, "error": "hostname is required"}
    remote_dir = str(connection.get("remote_dir") or DEFAULT_REMOTE_DIR)
    env_path = str(connection.get("env_path") or DEFAULT_ENV_PATH)
    probe = (
        "echo DASHBOARD_OK"
        " && echo host=$(hostname)"
        " && (uv --version 2>/dev/null || echo 'uv: missing')"
        " && (nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi: unavailable')"
        f" && (test -x {shlex.quote(env_path)}/bin/python && echo 'env: {env_path}' || echo 'env: {env_path} MISSING')"
        f" && ({shlex.quote(env_path)}/bin/python -c \"import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())\""
        f" 2>/dev/null || echo 'torch: not importable from {env_path}')"
        f" && (cd {shlex.quote(remote_dir)} 2>/dev/null && test -f scripts/uv_sync_no_torch.sh && echo 'no-torch sync script: present' || echo 'no-torch sync script: MISSING')"
        f" && (cd {shlex.quote(remote_dir)} 2>/dev/null && grep -q 'per-step-timing' cs336_systems/benchmark.py"
        f" && echo 'per-step timing: present' || echo 'per-step timing: MISSING (redeploy assignment2)')"
        f" && (test -d {shlex.quote(remote_dir)} && echo 'dir: present' || echo 'dir: MISSING ({remote_dir})')"
    )
    argv = build_ssh_argv(connection, probe)
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=SSH_MAX_TIMEOUT, check=False)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timed out after {SSH_MAX_TIMEOUT}s", "command": _display_ssh(argv)}
    except FileNotFoundError:
        return {"ok": False, "error": "ssh executable not found on PATH"}
    output = (proc.stdout or "") + (proc.stderr or "")
    ok = proc.returncode == 0 and "DASHBOARD_OK" in output
    return {
        "ok": ok,
        "exit_code": proc.returncode,
        "output": output.strip(),
        "error": None if ok else "ssh probe failed (check host, port, key and network)",
        "command": _display_ssh(argv),
    }


def sync_to_remote(connection: dict[str, Any]) -> dict[str, Any]:
    """Ship the local working tree to the remote dir over SSH.

    Mirrors ``scripts/deploy_code.sh``: the file list comes from ``git ls-files
    -co --exclude-standard`` (tracked plus untracked-but-not-ignored, so
    uncommitted edits and new files are included) and is piped through tar into
    ``ssh`` where the archive is extracted under the remote dir. The SSH options
    are the same ones the benchmark and probe use, including BatchMode.
    """
    host = str(connection.get("host") or "").strip()
    if not host:
        return {"ok": False, "error": "hostname is required"}
    remote_dir = str(connection.get("remote_dir") or DEFAULT_REMOTE_DIR)

    try:
        listing = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard"],
            cwd=LOCAL_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return {"ok": False, "error": "git executable not found on PATH (server host)"}
    if listing.returncode != 0:
        detail = (listing.stderr or "").strip()
        return {"ok": False, "error": detail or "git ls-files failed"}
    files = [line for line in listing.stdout.splitlines() if line]
    if not files:
        return {"ok": False, "error": "nothing to sync (git ls-files returned no files)"}

    remote_cmd = f"mkdir -p {shlex.quote(remote_dir)} && tar -xzf - -C {shlex.quote(remote_dir)}"
    argv = build_ssh_argv(connection, remote_cmd)
    # Pipe the packed tree straight into ssh; bash handles the pipeline so the
    # archive is never buffered in memory. Exit status is the remote tar's.
    pipeline = "git ls-files -co --exclude-standard -z | tar --null -T - -czf - | " + shlex.join(argv)
    started = time.time()
    with SYNC_LOCK:
        try:
            proc = subprocess.run(
                ["bash", "-c", pipeline],
                cwd=LOCAL_REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return {"ok": False, "error": "bash executable not found on PATH"}
    duration = time.time() - started
    output = ((proc.stdout or "") + (proc.stderr or "")).strip()
    ok = proc.returncode == 0
    return {
        "ok": ok,
        "files": len(files),
        "remote_dir": remote_dir,
        "duration": round(duration, 2),
        "exit_code": proc.returncode,
        "output": output,
        "error": None if ok else f"sync failed (exit code {proc.returncode})",
        "command": _display_ssh(argv),
    }


app = FastAPI(title="CS336 Benchmark Dashboard", docs_url=None, redoc_url=None)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html", headers={"Cache-Control": "no-store"})


@app.get("/plotly.min.js")
def plotly_bundle() -> FileResponse:
    if not PLOTLY_JS or not PLOTLY_JS.exists():
        raise HTTPException(status_code=404, detail="plotly bundle not installed")
    return FileResponse(PLOTLY_JS, media_type="application/javascript", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/schema")
def api_schema() -> dict[str, Any]:
    return {"connection": CONNECTION_FIELDS, "benchmark": SCHEMA, "profiling": PROFILING_FIELDS, "memory": MEMORY_FIELDS, "presets": PARAM_PRESETS}


@app.get("/api/runs")
def api_runs() -> dict[str, Any]:
    return {"runs": STORE.list()}


@app.post("/api/runs", status_code=201)
def api_start_run(req: RunRequest) -> dict[str, Any]:
    connection = req.connection.model_dump()
    if not str(connection.get("host") or "").strip():
        raise HTTPException(status_code=400, detail="hostname is required")
    params = dict(req.params)
    missing = [name for name in ("d_model", "d_ff", "num_layers", "num_heads") if params.get(name) in (None, "")]
    if missing:
        raise HTTPException(status_code=400, detail="missing required parameters: " + ", ".join(missing))
    clean_params = {key: _coerce({"type": next((s["type"] for s in SCHEMA if s["name"] == key), "str")}, value) for key, value in params.items()}
    raw_profiling = req.profiling.model_dump()
    profiling = {field["name"]: _coerce(field, raw_profiling.get(field["name"])) for field in PROFILING_FIELDS}
    raw_memory = req.memory.model_dump()
    memory = {field["name"]: _coerce(field, raw_memory.get(field["name"])) for field in MEMORY_FIELDS}
    run = Runner.start(STORE, connection, clean_params, profiling, memory)
    return run.summary()


@app.get("/api/runs/{run_id}")
def api_get_run(run_id: str, offset: int = 0) -> dict[str, Any]:
    run = STORE.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    return run.to_dict(offset)


@app.post("/api/runs/{run_id}/stop")
def api_stop_run(run_id: str) -> dict[str, Any]:
    run = STORE.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    Runner.stop(run)
    return run.summary()


@app.post("/api/runs/{run_id}/delete")
def api_delete_run(run_id: str) -> dict[str, Any]:
    return {"deleted": STORE.delete(run_id)}


@app.get("/api/runs/{run_id}/report")
def api_download_report(run_id: str) -> FileResponse:
    run = STORE.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    if not (run.profiling.get("enabled") and run.report_path):
        raise HTTPException(status_code=400, detail="profiling was not enabled for this run")
    ok, error = run.fetch_report()
    if not ok:
        raise HTTPException(status_code=502, detail=f"could not fetch report: {error}")
    local = run.report_local_path()
    return FileResponse(local, filename=local.name, media_type="application/octet-stream")


@app.post("/api/runs/{run_id}/memory/fetch")
def api_fetch_memory(run_id: str) -> dict[str, Any]:
    """Pull the CUDA memory snapshot from the instance over SSH (cached)."""
    run = STORE.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    if not (run.memory.get("enabled") and run.memory_path):
        raise HTTPException(status_code=400, detail="memory profiling was not enabled for this run")
    ok, error = run.fetch_memory()
    if not ok:
        raise HTTPException(status_code=502, detail=f"could not fetch memory snapshot: {error}")
    return {"ok": True, **run.memory_info()}


@app.get("/api/runs/{run_id}/memory")
def api_download_memory(run_id: str) -> FileResponse:
    run = STORE.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    if not (run.memory.get("enabled") and run.memory_path):
        raise HTTPException(status_code=400, detail="memory profiling was not enabled for this run")
    ok, error = run.fetch_memory()
    if not ok:
        raise HTTPException(status_code=502, detail=f"could not fetch memory snapshot: {error}")
    local = run.memory_local_path()
    return FileResponse(local, filename=local.name, media_type="application/octet-stream")


@app.get("/api/runs/{run_id}/memory_viz")
def api_memory_viz(run_id: str) -> Response:
    """Serve PyTorch's MemoryViz viewer with this run's snapshot inlined."""
    run = STORE.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    if not (run.memory.get("enabled") and run.memory_path):
        raise HTTPException(status_code=400, detail="memory profiling was not enabled for this run")
    ok, error = run.fetch_memory()
    if not ok:
        raise HTTPException(status_code=502, detail=f"could not fetch memory snapshot: {error}")
    local = run.memory_local_path()
    encoded = base64.b64encode(local.read_bytes()).decode("ascii")
    files = json.dumps([{"name": local.name, "base64": encoded}])
    html = MEMORY_VIZ_TEMPLATE.replace("$VIZ_KIND", repr("Active Memory Timeline")).replace("$SNAPSHOT", files)
    return Response(content=html, media_type="text/html", headers={"Cache-Control": "no-store"})


@app.post("/api/test-connection")
def api_test_connection(req: ConnectionRequest) -> dict[str, Any]:
    return probe_connection(req.connection.model_dump())


@app.post("/api/sync")
def api_sync(req: ConnectionRequest) -> dict[str, Any]:
    return sync_to_remote(req.connection.model_dump())


@app.post("/api/runs/{run_id}/export")
def api_export_run(run_id: str, req: ExportRequest | None = None) -> Response:
    """Bundle a run's log, profiler report and (optional) graph into a zip."""
    run = STORE.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    # Fetch/reuse the report first so the fetch progress line is captured in the
    # log written below. A failed fetch just omits the report from the archive.
    if run.profiling.get("enabled") and run.report_path:
        run.fetch_report()
    if run.memory.get("enabled") and run.memory_path:
        run.fetch_memory()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        meta = {**run.summary(), "argv": run.argv, "connection": run.connection}
        archive.writestr("run.json", json.dumps(meta, indent=2))
        archive.writestr("benchmark.log", "\n".join(line["text"] for line in run.logs))
        local_report = run.report_local_path()
        if run.report_info().get("downloaded") and local_report.exists():
            archive.write(local_report, arcname=local_report.name)
        local_memory = run.memory_local_path()
        if run.memory_info().get("downloaded") and local_memory.exists():
            archive.write(local_memory, arcname=local_memory.name)
        graph = req.graph_png if req else None
        if isinstance(graph, str) and graph.startswith("data:image/png;base64,"):
            try:
                archive.writestr("per-step.png", base64.b64decode(graph.split(",", 1)[1]))
            except (ValueError, TypeError):
                pass
    buf.seek(0)
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{run.id}.zip"'},
    )


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="port (default: 8765)")
    parser.add_argument("--reload", action="store_true", help="auto-reload on code changes (dev only)")
    args = parser.parse_args(argv)

    print(f"CS336 benchmark dashboard -> http://{args.host}:{args.port}")
    print(f"  benchmark CLI options discovered: {len(SCHEMA)}")
    if args.reload:
        uvicorn.run("server:app", host=args.host, port=args.port, reload=True)
    else:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
