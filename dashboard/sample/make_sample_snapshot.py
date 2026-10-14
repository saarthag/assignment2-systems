#!/usr/bin/env python3
"""Generate a small, synthetic CUDA memory snapshot for local dashboard testing.

Real snapshots come from ``torch.cuda.memory._dump_snapshot`` on a GPU box. This
script fabricates one with the same structure (``segments`` + ``device_traces``)
so the dashboard's memory-viz panel can be exercised on a machine without CUDA.
The timeline imitates a single XL-ish training step: persistent weights, a
forward pass that allocates activations, a backward pass that frees activations
and allocates gradients, then an optimizer step that allocates optimizer state.

Usage::

    python dashboard/sample/make_sample_snapshot.py [output.pickle]
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

MiB = 1024 * 1024
BASE = 0x00007F00_00000000

WEIGHT_FRAMES = [
    {"filename": "cs336_basics/transformer.py", "line": 212, "name": "TransformerLM.__init__"},
    {"filename": "torch/nn/modules/sparse.py", "line": 157, "name": "Embedding.__init__"},
]
ACT_FRAMES = [
    {"filename": "cs336_systems/benchmark.py", "line": 131, "name": "step"},
    {"filename": "cs336_basics/transformer.py", "line": 98, "name": "TransformerBlock.forward"},
    {"filename": "torch/nn/modules/linear.py", "line": 125, "name": "Linear.forward"},
]
GRAD_FRAMES = [
    {"filename": "cs336_systems/benchmark.py", "line": 153, "name": "step"},
    {"filename": "torch/autograd/engine.py", "line": 286, "name": "backward"},
    {"filename": "cs336_basics/transformer.py", "line": 104, "name": "TransformerBlock.backward"},
]
OPT_FRAMES = [
    {"filename": "cs336_systems/benchmark.py", "line": 164, "name": "step"},
    {"filename": "cs336_basics/transformer.py", "line": 61, "name": "AdamW.step"},
]


def _frames(frames):
    return [dict(f) for f in frames]


class Allocator:
    """Hands out non-overlapping addresses and tracks live blocks."""

    def __init__(self) -> None:
        self.cursor = BASE
        # addr -> (size, frames, category); presence means the block is live.
        self.live: dict[int, tuple[int, list[dict], str]] = {}

    def alloc(self, size: int, frames, category: str) -> int:
        addr = self.cursor
        self.cursor += size
        self.live[addr] = (size, frames, category)
        return addr

    def free(self, addr: int) -> None:
        self.live.pop(addr, None)


def build_snapshot() -> dict:
    allocator = Allocator()
    trace: list[dict] = []

    def emit(action: str, addr: int, size: int, frames, category: str | None = None) -> None:
        event = {"action": action, "addr": addr, "size": size, "stream": 0, "frames": _frames(frames)}
        if category is not None:
            event["category"] = category
        trace.append(event)

    # --- Reserved segment from the caching allocator -----------------------
    segment_size = 1024 * MiB
    segment_addr = BASE
    emit("segment_alloc", segment_addr, segment_size, [], "segment")

    # --- Persistent weights -------------------------------------------------
    for _name, size in (
        ("embedding", 128 * MiB),
        ("attention", 96 * MiB),
        ("mlp", 256 * MiB),
        ("lm_head", 128 * MiB),
    ):
        addr = allocator.alloc(size, WEIGHT_FRAMES, "weights")
        emit("alloc", addr, size, WEIGHT_FRAMES, "weights")

    # --- Forward pass: activations -----------------------------------------
    activation_addrs: list[int] = []
    for _layer in range(4):
        for _kind, size in (("attn", 24 * MiB), ("mlp", 40 * MiB), ("residual", 8 * MiB)):
            addr = allocator.alloc(size, ACT_FRAMES, "activations")
            emit("alloc", addr, size, ACT_FRAMES, "activations")
            activation_addrs.append(addr)

    # --- Backward pass: frees activations, allocates gradients --------------
    for idx, addr in enumerate(activation_addrs):
        size = allocator.live[addr][0]
        emit("free_requested", addr, size, ACT_FRAMES)
        emit("free_completed", addr, size, ACT_FRAMES)
        allocator.free(addr)
        if idx % 3 == 0:
            grad_addr = allocator.alloc(size, GRAD_FRAMES, "gradients")
            emit("alloc", grad_addr, size, GRAD_FRAMES, "gradients")

    # --- Optimizer step: persistent optimizer state -------------------------
    for addr, (size, _fr, category) in list(allocator.live.items()):
        if category != "weights":
            continue
        opt_size = size // 2
        opt_addr = allocator.alloc(opt_size, OPT_FRAMES, "optimizer_state")
        emit("alloc", opt_addr, opt_size, OPT_FRAMES, "optimizer_state")

    # --- Snapshot: list every still-live allocation as a segment block ------
    frames_for = {
        "weights": WEIGHT_FRAMES,
        "activations": ACT_FRAMES,
        "gradients": GRAD_FRAMES,
        "optimizer_state": OPT_FRAMES,
    }
    blocks = []
    for _addr, (size, _fr, category) in sorted(allocator.live.items()):
        blocks.append(
            {
                "size": size,
                "requested_size": size,
                "state": "active_allocated",
                "frames": _frames(frames_for.get(category, WEIGHT_FRAMES)),
            }
        )
    allocated_size = sum(b["size"] for b in blocks)
    segments = [
        {
            "device": 0,
            "address": segment_addr,
            "total_size": segment_size,
            "stream": 0,
            "segment_type": "large",
            "allocated_size": allocated_size,
            "active_size": allocated_size,
            "blocks": blocks,
        }
    ]

    return {"segments": segments, "device_traces": [trace, []], "categories": []}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    out = Path(argv[0]) if argv else Path(__file__).resolve().parent / "memory_snapshot.pickle"
    snapshot = build_snapshot()
    with out.open("wb") as fh:
        pickle.dump(snapshot, fh)
    print(f"wrote {out} ({out.stat().st_size} bytes, {len(snapshot['device_traces'][0])} trace events)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
