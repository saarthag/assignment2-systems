import argparse
import json
import logging
import math
import shlex
import statistics
import timeit
from functools import partial
from typing import Any

import cs336_basics.transformer
import humanize
import torch
import torch.nn.functional as F
from cs336_basics.transformer import AdamW, TransformerLM, my_softmax
from einops import einsum
from jaxtyping import Bool, Float
from pydantic import BaseModel
from torch import Tensor
from torch.cuda import nvtx

logger = logging.getLogger(__name__)


@nvtx.range("sdpa")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    with nvtx.range("computing attention scores"):
        sdp = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)

    with nvtx.range("applying causal mask"):
        pre_softmax = sdp.masked_fill(~mask, -torch.inf) if mask is not None else sdp

    with nvtx.range("computing softmax"):
        out = einsum(my_softmax(pre_softmax, dim=-1), V, "... queries keys, ... keys d_v -> ... queries d_v")

    return out


class TransformerLMConfig(BaseModel):
    """Validated configuration for :class:`TransformerLM`."""

    vocab_size: int = 10_000
    context_length: int = 512
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    theta: float = 10_000.0

    def build_kwargs(self) -> dict[str, Any]:
        """Keyword arguments accepted by ``TransformerLM.__init__``."""
        kwargs = self.model_dump()
        kwargs["rope_theta"] = kwargs.pop("theta")
        return kwargs

    def build(self, **overrides: Any) -> TransformerLM:
        return TransformerLM(**self.build_kwargs(), **overrides)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a TransformerLM forward/backward pass.",
        fromfile_prefix_chars="@",
    )
    parser.convert_arg_line_to_args = partial(shlex.split, comments=True)
    parser.add_argument("--vocab-size", type=int, help="size of the token vocabulary (default: 10000)")
    parser.add_argument("--context-length", type=int, help="maximum sequence length (default: 512)")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size (default: 4)")
    parser.add_argument("--d-model", type=int, help="model embedding dimension (required)")
    parser.add_argument("--d-ff", type=int, help="feed-forward hidden dimension (required)")
    parser.add_argument("--num-layers", type=int, help="number of transformer blocks (required)")
    parser.add_argument("--num-heads", type=int, help="number of attention heads (required)")
    parser.add_argument("--theta", type=float, help="RoPE theta (default: 10000.0)")
    parser.add_argument(
        "--compute",
        choices=["b", "bo"],
        default=None,
        help="unset = forward only, 'b' = forward+backward, 'bo' = forward+backward+optimizer step",
    )

    parser.add_argument("-w", "--warmup-steps", type=int, default=10, help="number of warmup steps (default: 10)")
    parser.add_argument("-n", "--bench-steps", type=int, default=100, help="number of timed steps (default: 100)")

    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    parser.add_argument("--per-step-timing", action="store_true", help="emit per-step timings as JSON lines (used by the dashboard)")
    return parser.parse_args(argv)


def step(model: TransformerLM, x: torch.Tensor, y: torch.Tensor, compute: str | None = None, optimizer: AdamW | None = None):
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("step: x=%s y=%s compute=%s optimizer=%s", tuple(x.shape), tuple(y.shape), compute, type(optimizer).__name__ if optimizer else None)

    with nvtx.range("forward"):
        logits = model(x)
        torch.accelerator.synchronize()

    if compute is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        if logger.isEnabledFor(logging.DEBUG):
            # NOTE: .item() forces a device sync, which perturbs timing; only evaluated with -v.
            logger.debug("loss=%.4f", loss.item())
        if optimizer is not None:
            optimizer.zero_grad()
        # need to sync before the backward pass to avoid
        # inflating its time with cross_entropy and zero_grad
        torch.accelerator.synchronize()

        with nvtx.range("backward"):
            loss.backward()
            torch.accelerator.synchronize()

        if compute == "bo":
            assert optimizer is not None, "Optimizer should be an AdamW instance if compute == bo"
            with nvtx.range("optimizer"):
                optimizer.step()
                torch.accelerator.synchronize()


def bench(
    model: TransformerLM,
    config: TransformerLMConfig,
    batch_size: int,
    compute: str | None = None,
    optimizer: AdamW | None = None,
    warmup_steps: int = 10,
    bench_steps: int = 100,
    per_step_timing: bool = False,
) -> float:
    device = torch.accelerator.current_accelerator() or torch.device("cpu")
    logger.info("Using device: %s", device)
    model = model.to(device)

    x = torch.randint(high=config.vocab_size, size=(batch_size, config.context_length), device=device)
    y = torch.randint(high=config.vocab_size, size=(batch_size, config.context_length), device=device)
    logger.debug("Benchmark inputs: x=%s y=%s", tuple(x.shape), tuple(y.shape))

    logger.info("Warming up for %d step(s)", warmup_steps)
    with nvtx.range("warmup"):
        for _ in range(warmup_steps):
            step(model, x, y, compute, optimizer)

    torch.accelerator.synchronize()

    logger.info("Benchmarking %d step(s) (compute=%s)", bench_steps, compute)
    step_times: list[float] = []
    start_time = timeit.default_timer()
    for index in range(bench_steps):
        if per_step_timing:
            step_start = timeit.default_timer()
        step(model, x, y, compute, optimizer)
        if per_step_timing:
            torch.accelerator.synchronize()
            step_ms = (timeit.default_timer() - step_start) * 1e3
            step_times.append(step_ms)
            print(json.dumps({"event": "step", "step": index + 1, "ms": step_ms}), flush=True)

    torch.accelerator.synchronize()
    end_time = timeit.default_timer()

    elapsed = end_time - start_time
    per_step = elapsed / bench_steps
    logger.info("Per-step time: %.3f ms (%d steps in %.3f s)", per_step * 1e3, bench_steps, elapsed)
    if per_step_timing:
        print(
            json.dumps(
                {
                    "event": "summary",
                    "steps": len(step_times),
                    "mean_ms": statistics.fmean(step_times),
                    "std_ms": statistics.pstdev(step_times) if len(step_times) > 1 else 0.0,
                    "min_ms": min(step_times),
                    "max_ms": max(step_times),
                    "median_ms": statistics.median(step_times),
                    "total_s": elapsed,
                }
            ),
            flush=True,
        )
    return per_step


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Swap in the NVTX-annotated SDPA implementation when NVTX is available. The annotated version
    # wraps the attention computation in NVTX ranges; torch gates that support on torch._C._nvtx
    # (which raises on first use when missing), so key off the same symbol the wrapper does.
    if hasattr(torch._C, "_nvtx"):
        cs336_basics.transformer.scaled_dot_product_attention = annotated_scaled_dot_product_attention
        logger.info(
            "Swapped cs336_basics.transformer.scaled_dot_product_attention for annotated_scaled_dot_product_attention (CUDA %s, NVTX ranges enabled).",
            torch.version.cuda,
        )
    else:
        logger.warning(
            "PyTorch was built without NVTX (torch._C._nvtx is missing); skipping swap to "
            "annotated_scaled_dot_product_attention. NVTX ranges will not be emitted and the "
            "original cs336_basics.transformer.scaled_dot_product_attention will be benchmarked."
        )

    # Merge CLI values on top of the pydantic defaults (unset args are None).
    values = {name: value for name in TransformerLMConfig.model_fields if (value := getattr(args, name, None)) is not None}
    config = TransformerLMConfig(**values)
    logger.debug("Using TransformerLM config: %s", config)
    logger.debug("Running compute: %s", args.compute)
    logger.debug("Batch size: %d", args.batch_size)

    model = config.build()
    num_params = sum(p.numel() for p in model.parameters())
    logger.info("Constructed TransformerLM with %s parameters", humanize.intword(num_params, format="%.2f"))

    compute = getattr(args, "compute", None)
    optimizer = AdamW(params=model.parameters()) if compute == "bo" else None

    bench(
        model=model,
        config=config,
        batch_size=args.batch_size,
        compute=compute,
        optimizer=optimizer,
        warmup_steps=args.warmup_steps,
        bench_steps=args.bench_steps,
        per_step_timing=args.per_step_timing,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
