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
* ``POST /api/test-connection``     -> probe ssh connectivity (blocking)
"""

from __future__ import annotations

import argparse
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
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field, create_model  # noqa: E402
from schema import extract_cli_schema  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
RUNS_DIR = Path(__file__).resolve().parent / ".runs"
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
# Python backtraces, GPU metrics), plus --force-overwrite true so a re-run never
# blocks on nsys' interactive overwrite prompt over BatchMode SSH.
DEFAULT_PROFILER_ARGS = (
    "--force-overwrite true "
    "--trace=cuda,cudnn,cublas,osrt,nvtx "
    "--pytorch=functions-trace,autograd-shapes-nvtx "
    "--cudabacktrace=all "
    "--python-backtrace=cuda "
    "--gpu-metrics-devices=0"
)
DEFAULT_PROFILER_DIR = "profiles"

BENCH_FALLBACK = "uv run --no-sync python -m cs336_systems.benchmark"

# Connection fields surfaced in the UI. The pydantic request model is generated
# from this list so there is a single source of truth.
CONNECTION_FIELDS: list[dict[str, Any]] = [
    {"name": "host", "label": "Hostname", "type": "str", "default": "", "placeholder": "ssh4.vast.ai", "help": "Vast.ai instance host (the part before :port)."},
    {"name": "port", "label": "SSH port", "type": "int", "default": 22, "placeholder": "12345", "help": "Mapped SSH port for the instance."},
    {"name": "user", "label": "User", "type": "str", "default": "root", "placeholder": "root", "help": "Remote user."},
    {"name": "ssh_key", "label": "SSH key", "type": "str", "default": "", "placeholder": "~/.ssh/id_ed25519", "help": "Optional identity file. Blank uses your ssh config / agent."},
    {"name": "remote_dir", "label": "Remote dir", "type": "str", "default": DEFAULT_REMOTE_DIR, "placeholder": DEFAULT_REMOTE_DIR, "help": "Directory containing the assignment2 tree on the instance."},
    {"name": "env_path", "label": "Remote uv env", "type": "str", "default": DEFAULT_ENV_PATH, "placeholder": DEFAULT_ENV_PATH, "help": "Exported as UV_PROJECT_ENVIRONMENT on the instance (before setup + benchmark). Vast.ai PyTorch images use /venv/main, which keeps the preinstalled torch/CUDA stack."},
    {"name": "bench_cmd", "label": "Benchmark command", "type": "str", "default": DEFAULT_BENCH_CMD, "placeholder": DEFAULT_BENCH_CMD, "help": "Runs inside the remote dir. `--no-sync` guarantees uv will not reinstall dependencies. The per-step timing checkbox appends --per-step-timing to this command."},
    {"name": "setup_cmd", "label": "Setup command", "type": "str", "default": DEFAULT_SETUP_CMD, "placeholder": DEFAULT_SETUP_CMD, "help": "Run before every benchmark. Defaults to uv_sync_no_torch.sh, which skips torch/triton/cuda/nvidia and never prunes the image env."},
    {"name": "extra_args", "label": "Extra CLI args", "type": "str", "default": "", "placeholder": "--batch-size 8", "help": "Raw extra flags appended to the benchmark command."},
    {"name": "batch_mode", "label": "Non-interactive SSH", "type": "bool", "default": True, "help": "Adds -o BatchMode=yes so SSH never hangs waiting for a password."},
]

# Profiler settings. These are not part of the benchmark CLI; they control how
# the remote command is wrapped and where the report is written.
PROFILING_FIELDS: list[dict[str, Any]] = [
    {"name": "enabled", "label": "Profile with Nsight Systems (nsys)", "type": "bool", "default": False, "help": "Wrap the benchmark command in the profiler and write a report on the instance for download."},
    {"name": "profiler_cmd", "label": "Profiler command", "type": "str", "default": DEFAULT_PROFILER_CMD, "placeholder": DEFAULT_PROFILER_CMD, "help": "Wrapper prepended to the benchmark command. `nsys profile` for Nsight Systems, or `ncu` for Nsight Compute."},
    {"name": "profiler_args", "label": "Profiler args", "type": "str", "default": DEFAULT_PROFILER_ARGS, "placeholder": DEFAULT_PROFILER_ARGS, "help": "Extra profiler flags. Defaults to the assignment-suggested nsys flags (NVTX/PyTorch annotations, CUDA + Python backtraces, GPU metrics). `-o <report>` is appended automatically."},
    {"name": "report_dir", "label": "Remote report dir", "type": "str", "default": DEFAULT_PROFILER_DIR, "placeholder": DEFAULT_PROFILER_DIR, "help": "Where the report is written on the instance (relative to the remote dir, or absolute)."},
]

MODEL_FIELDS = {"d_model", "d_ff", "num_layers", "num_heads"}
RUN_FIELDS = {"warmup_steps", "bench_steps", "verbose"}
PARAM_PRESETS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
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


class ConnectionRequest(BaseModel):
    connection: ConnectionConfig


class RunRequest(BaseModel):
    connection: ConnectionConfig
    params: dict[str, Any] = Field(default_factory=dict)
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)


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
            except Exception:
                event = None
            if isinstance(event, dict) and event.get("event") == "step":
                step_no = int(event.get("step", 0))
                ms = float(event.get("ms", 0.0))
                return f"step {step_no}: {ms:.3f} ms", {"step": step_no, "ms": ms}, None
            if isinstance(event, dict) and event.get("event") == "summary":
                summary = {k: v for k, v in event.items() if k != "event"}
                display = (
                    f"summary: mean={summary.get('mean_ms', 0.0):.3f} ms  "
                    f"std={summary.get('std_ms', 0.0):.3f} ms  (n={summary.get('steps', 0)})"
                )
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
        (self._dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def close(self) -> None:
        try:
            self._log_fh.close()
        except Exception:
            pass

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
        remote = (
            f"cd {shlex.quote(remote_dir)}"
            f" && test -f {shlex.quote(self.report_path)}"
            f" && cat {shlex.quote(self.report_path)}"
        )
        argv = build_ssh_argv(self.connection, remote)
        self.ingest(f"[dashboard] fetching profiler report: {self.report_path}")
        try:
            with local.open("wb") as fh:
                proc = subprocess.run(argv, stdout=fh, stderr=subprocess.PIPE)
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

    @classmethod
    def load(cls, run_dir: Path) -> Run | None:
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
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
                except Exception:
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
    def start(store: RunStore, connection: dict[str, Any], params: dict[str, Any], profiling: dict[str, Any] | None = None) -> Run:
        profiling = dict(profiling or {})
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
        if profiling.get("enabled"):
            report_dir = str(profiling.get("report_dir") or DEFAULT_PROFILER_DIR).strip() or DEFAULT_PROFILER_DIR
            mkdir = f"mkdir -p {shlex.quote(report_dir)}"
            setup = f"{mkdir} && {setup}" if setup.strip() else mkdir
            report_prefix = f"{report_dir.rstrip('/')}/{run_id}"
            command = f"{build_profiler_prefix(profiling, report_prefix)} {command}"
            report_path = f"{report_prefix}{report_extension(str(profiling.get('profiler_cmd') or ''))}"

        remote_command = build_full_remote(remote_dir, env_path, setup, command)
        argv = build_ssh_argv(connection, remote_command)

        run = Run(run_id, dict(connection), dict(params), command, remote_command, argv, profiling, report_path)
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
        try:
            proc.send_signal(signal.SIGINT)
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


STORE = RunStore()
SCHEMA = extract_cli_schema()

try:
    import plotly

    PLOTLY_JS = Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"
except Exception:  # pragma: no cover - plotly is an optional dashboard dep
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
        f" && ({shlex.quote(env_path)}/bin/python -c \"import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())\" 2>/dev/null || echo 'torch: not importable from {env_path}')"
        f" && (cd {shlex.quote(remote_dir)} 2>/dev/null && test -f scripts/uv_sync_no_torch.sh && echo 'no-torch sync script: present' || echo 'no-torch sync script: MISSING')"
        f" && (cd {shlex.quote(remote_dir)} 2>/dev/null && grep -q 'per-step-timing' cs336_systems/benchmark.py && echo 'per-step timing: present' || echo 'per-step timing: MISSING (redeploy assignment2)')"
        f" && (test -d {shlex.quote(remote_dir)} && echo 'dir: present' || echo 'dir: MISSING ({remote_dir})')"
    )
    argv = build_ssh_argv(connection, probe)
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=SSH_MAX_TIMEOUT)
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
    return {"connection": CONNECTION_FIELDS, "benchmark": SCHEMA, "profiling": PROFILING_FIELDS, "presets": PARAM_PRESETS}


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
    run = Runner.start(STORE, connection, clean_params, profiling)
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


@app.post("/api/test-connection")
def api_test_connection(req: ConnectionRequest) -> dict[str, Any]:
    return probe_connection(req.connection.model_dump())


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
