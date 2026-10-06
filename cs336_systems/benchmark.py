import argparse
import logging
from pathlib import Path
from typing import Any
import timeit

import humanize
from pydantic import BaseModel

from cs336_basics.transformer import TransformerLM

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
    parser = argparse.ArgumentParser(description="Benchmark a TransformerLM forward/backward pass.")
    parser.add_argument("--vocab-size", type=int, help="size of the token vocabulary (default: 10000)")
    parser.add_argument("--context-length", type=int, help="maximum sequence length (default: 512)")
    parser.add_argument("--d-model", type=int, help="model embedding dimension (required)")
    parser.add_argument("--d-ff", type=int, help="feed-forward hidden dimension (required)")
    parser.add_argument("--num-layers", type=int, help="number of transformer blocks (required)")
    parser.add_argument("--num-heads", type=int, help="number of attention heads (required)")
    parser.add_argument("--theta", type=float, help="RoPE theta (default: 10000.0)")

    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    return parser.parse_args(argv)


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

    model = config.build()
    num_params = sum(p.numel() for p in model.parameters())
    logger.info("Constructed TransformerLM with %s parameters", humanize.intword(num_params, format="%.2f"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
