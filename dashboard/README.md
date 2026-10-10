# CS336 Benchmark Dashboard

A local web dashboard for running `cs336_systems.benchmark` on a remote GPU
instance (e.g. a Vast.ai box) over SSH. Pick your connection settings and
benchmark parameters, watch the logs stream back, and see a live Plotly graph of
step number vs per-step time with aggregate mean / std-dev metrics.

The backend is [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn, and it shells
out to the system `ssh` client — the same approach as `scripts/deploy_code.sh`.
Dashboard dependencies live in their own `dashboard` uv dependency group, so the
assignment environment is unaffected.

## Quick start

```sh
# from the repo root
uv run --group dashboard python dashboard/server.py            # http://127.0.0.1:8765
uv run --group dashboard python dashboard/server.py --port 9000 --host 0.0.0.0
```

Then open <http://127.0.0.1:8765>, fill in the connection details and click
**Test connection** followed by **Run benchmark**.

> Tip: the browser never talks to the instance directly. All SSH happens on the
> machine running `server.py`, so it works behind a firewall/VPN just like your
> terminal `ssh` does.

## Dependencies

Added to a dedicated uv dependency group:

```sh
uv add --group dashboard fastapi "uvicorn[standard]" plotly
```

`plotly` is only used to serve its bundled `plotly.min.js` locally at
`/plotly.min.js`, so the graph works without a CDN (there is a CDN fallback in
`index.html`). Everything PyTorch-related stays in the main project
dependencies.

## Per-step timings

`cs336_systems.benchmark` gained one minimal, opt-in flag:

```
--per-step-timing   emit per-step timings as JSON lines (used by the dashboard)
```

When set, `bench()` times each step (synchronizing per step), prints one JSON
event per step plus a summary:

```json
{"event": "step", "step": 1, "ms": 42.31}
{"event": "summary", "steps": 20, "mean_ms": 42.1, "std_ms": 0.9, "min_ms": 40.0, "max_ms": 46.2, "median_ms": 41.7, "total_s": 0.84}
```

Without the flag the benchmark behaves exactly as before. The dashboard enables
the checkbox by default and parses these events into the graph and metric cards;
it still understands the stock `Per-step time: …` aggregate line if you point it
at another command.

### Dependencies are never reinstalled

Vast.ai's PyTorch image already ships a working torch/CUDA stack in
`/venv/main`, so the dashboard deliberately avoids `uv sync`/`uv run` doing a
full resolve-and-install. The generated remote command is:

```sh
cd "$REMOTE_DIR" \
  && export UV_PROJECT_ENVIRONMENT=/venv/main \
  && bash scripts/uv_sync_no_torch.sh \
  && exec uv run --no-sync python -m cs336_systems.benchmark --per-step-timing <args...>
```

* `scripts/uv_sync_no_torch.sh` installs every dependency **except**
  `torch`/`triton`/`cuda-*`/`nvidia-*`, uses `--inexact` so it never prunes the
  image's own libraries, and refuses to recreate an existing env if its Python
  doesn't match `requires-python`. See that script's header for details.
* `uv run --no-sync` guarantees uv does not re-resolve or reinstall anything
  when the benchmark starts.

Change **Remote uv env** if your image uses a different environment path, and
blank/edit the setup command if you don't need it.

## Benchmark parameters

The parameter form is generated at startup by parsing the `argparse` definitions
in `cs336_systems/benchmark.py` with `ast` (see `dashboard/schema.py`). It does
**not** import torch, and it stays in sync automatically: add a
`parser.add_argument(...)` and the field appears in the UI.

Fields left blank are omitted from the command, so the model's pydantic defaults
apply. `d_model`, `d_ff`, `num_layers` and `num_heads` are required.
`warmup_steps=0` disables warmup. Quick presets for small/medium/large model
shapes are provided.

## Features

* Live Plotly graph of step number vs per-step time, with a dashed mean line.
* Aggregate metric cards: mean, std dev, min, max, median, steps, params,
  duration and exit code.
* Live logs (stdout + stderr merged) in a secondary panel under the graph, with
  per-level colouring.
* Run history persisted to `dashboard/.runs/` and reloaded on restart.
* Test-connection probe that reports the GPU, `uv`, the remote env, torch/CUDA
  availability, the no-torch sync script and whether the per-step flag is
  deployed.
* Stop, download-log, delete-run and command preview.

## HTTP API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/schema` | Connection fields, benchmark option schema, presets. |
| `GET` | `/api/runs` | Run summaries, newest first. |
| `POST` | `/api/runs` | Start a run: `{"connection": {...}, "params": {...}}`. |
| `GET` | `/api/runs/<id>?offset=N` | Run state, per-step metrics + log lines from `N`. |
| `POST` | `/api/runs/<id>/stop` | Stop a running benchmark. |
| `POST` | `/api/runs/<id>/delete` | Delete a run and its log. |
| `POST` | `/api/test-connection` | Probe SSH connectivity (blocking). |

## Files

```
dashboard/
├── server.py          # FastAPI app + SSH orchestration + metric parsing
├── schema.py          # AST parser for the benchmark CLI
├── static/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── README.md
└── .runs/             # persisted run metadata + logs (gitignored)
```

The per-step flag lives in `cs336_systems/benchmark.py`; no other assignment
code is touched.
