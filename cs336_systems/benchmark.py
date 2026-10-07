import argparse
import logging
import shlex
from functools import partial
from typing import Any
import timeit

import humanize
import torch
import torch.nn.functional as F
from pydantic import BaseModel

from cs336_basics.transformer import TransformerLM, AdamW

logger = logging.getLogger(__name__)


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
    return parser.parse_args(argv)


def step(model: TransformerLM, x: torch.Tensor, y: torch.Tensor, compute: str | None = None, optimizer: AdamW | None = None):
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("step: x=%s y=%s compute=%s optimizer=%s", tuple(x.shape), tuple(y.shape), compute, type(optimizer).__name__ if optimizer else None)

    logits = model(x)

    if compute is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        if logger.isEnabledFor(logging.DEBUG):
            # NOTE: .item() forces a device sync, which perturbs timing; only evaluated with -v.
            logger.debug("loss=%.4f", loss.item())
        if optimizer is not None:
            optimizer.zero_grad()
        loss.backward()

        if compute == "bo":
            assert optimizer is not None, "Optimizer should be an AdamW instance if compute == bo"
            optimizer.step()


def bench(
    model: TransformerLM,
    config: TransformerLMConfig,
    batch_size: int,
    compute: str | None = None,
    optimizer: AdamW | None = None,
    warmup_steps: int = 10,
    bench_steps: int = 100,
) -> float:
    device = torch.accelerator.current_accelerator() or torch.device("cpu")
    logger.info("Using device: %s", device)
    model = model.to(device)

    x = torch.randint(high=config.vocab_size, size=(batch_size, config.context_length), device=device)
    y = torch.randint(high=config.vocab_size, size=(batch_size, config.context_length), device=device)
    logger.debug("Benchmark inputs: x=%s y=%s", tuple(x.shape), tuple(y.shape))

    logger.info("Warming up for %d step(s)", warmup_steps)
    for _ in range(warmup_steps):
        step(model, x, y, compute, optimizer)

    torch.accelerator.synchronize()

    logger.info("Benchmarking %d step(s) (compute=%s)", bench_steps, compute)
    start_time = timeit.default_timer()
    for _ in range(bench_steps):
        step(model, x, y, compute, optimizer)

    torch.accelerator.synchronize()
    end_time = timeit.default_timer()

    elapsed = end_time - start_time
    per_step = elapsed / bench_steps
    logger.info("Per-step time: %.3f ms (%d steps in %.3f s)", per_step * 1e3, bench_steps, elapsed)
    return per_step


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
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
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
