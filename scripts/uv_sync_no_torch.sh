#!/usr/bin/env bash
# Sync the project with uv while skipping PyTorch and its GPU runtime stack.
#
# Why: a plain `uv sync` downloads torch (~500 MB) plus the huge CUDA wheels it
# depends on (nvidia-cublas / cudnn / nccl / ... several GB). When the machine
# already ships a working GPU environment - e.g. Vast.ai's PyTorch image with
# torch at /venv/main - those wheels are redundant and can even clobber the
# image's own CUDA libraries with mismatched versions.
#
# The skip list is derived from uv.lock, so it tracks CUDA version bumps
# (cu13 -> cu14) and renames automatically. It intentionally excludes only the
# torch/CUDA/triton stack; all other dependencies install normally.
#
# Usage:
#   scripts/uv_sync_no_torch.sh [extra uv sync args...]
#
# Env vars:
#   UV_PROJECT_ENVIRONMENT  uv target env, e.g. /venv/main (inherited by uv)
#   UV_SYNC_EXTRA_SKIP      space-separated extra package names to skip
#
# Examples:
#   # preview against the image env
#   UV_PROJECT_ENVIRONMENT=/venv/main scripts/uv_sync_no_torch.sh --dry-run
#   # sync for real
#   UV_PROJECT_ENVIRONMENT=/venv/main scripts/uv_sync_no_torch.sh
set -euo pipefail

# Resolve the project root from this script's location (scripts/..) rather than
# `git rev-parse`: deployed trees are shipped without .git, so git isn't
# available on the remote instance.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_root"

[[ -f uv.lock ]] || {
    echo "error: uv.lock not found in $repo_root" >&2
    exit 1
}

# Packages that provide torch and its GPU runtime. The host image supplies
# these, so tell uv to resolve (lock) but not install them.
skip=()
while IFS= read -r pkg; do
    [[ -n "$pkg" ]] && skip+=("$pkg")
done < <(
    grep -oE '^name = "(torch|triton|cuda-[^"]+|nvidia-[^"]+)"' uv.lock \
        | cut -d'"' -f2 | sort -u
)

if [[ -n "${UV_SYNC_EXTRA_SKIP:-}" ]]; then
    read -r -a _extra <<< "$UV_SYNC_EXTRA_SKIP"
    skip+=("${_extra[@]}")
fi

if ((${#skip[@]} == 0)); then
    echo "warning: no torch/cuda packages found in uv.lock; running a normal sync" >&2
else
    echo "uv sync: skipping ${#skip[@]} package(s): ${skip[*]}" >&2
fi

args=()
for pkg in "${skip[@]}"; do
    args+=(--no-install-package "$pkg")
done

# Safety: when targeting an existing environment (e.g. the image's /venv/main),
# uv will RECREATE it if its Python doesn't satisfy requires-python - wiping the
# very libraries we're trying to preserve. Detect that and refuse by default.
if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" && -e "$UV_PROJECT_ENVIRONMENT/bin/python" ]]; then
    dry="$(uv sync --inexact --dry-run "${args[@]}" "$@" 2>&1 || true)"
    if grep -q 'Would replace existing virtual environment' <<< "$dry"; then
        echo "error: uv would REPLACE $UV_PROJECT_ENVIRONMENT (its Python does not match" >&2
        echo "       requires-python in pyproject.toml). Refusing to destroy it." >&2
        if [[ "${UV_SYNC_ALLOW_RECREATE:-0}" != "1" ]]; then
            echo "       Re-run with UV_SYNC_ALLOW_RECREATE=1 to allow this." >&2
            exit 1
        fi
        echo "       UV_SYNC_ALLOW_RECREATE=1 set - proceeding anyway." >&2
    fi
fi

# --inexact: never prune libraries the image already ships (without it, uv would
# uninstall the preinstalled torch/NCCL because they aren't "owned" by the lock).
exec uv sync --inexact "${args[@]}" "$@"
