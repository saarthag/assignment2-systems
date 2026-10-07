#!/usr/bin/env bash
# shellcheck disable=SC2029  # $REMOTE_DIR is intentionally expanded client-side
# Ship the working tree to the remote instance.
#
# Uses `git ls-files -co --exclude-standard`, which lists tracked files plus
# untracked-but-not-ignored files. This respects .gitignore (so ignored paths
# like .venv/, __pycache__/, .env and *.log never leave the local machine)
# while still including uncommitted edits and new, not-yet-committed files.
#
# Env vars (set by the Makefile or the shell):
#   REMOTE      user@host of the instance           (required)
#   PORT        ssh port                             (default 22)
#   REMOTE_DIR  destination directory on the remote  (default /workspace/assignment2)
#   SSH_KEY     optional identity file
set -euo pipefail

REMOTE="${REMOTE:?set REMOTE=user@host}"
PORT="${PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/assignment2}"
SSH_KEY="${SSH_KEY:-}"

ssh_opts=(-p "$PORT" -o StrictHostKeyChecking=accept-new)
[[ -n "$SSH_KEY" ]] && ssh_opts+=(-i "$SSH_KEY")

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

echo "code: $(git rev-parse --short HEAD) (+working tree) -> $REMOTE:$REMOTE_DIR"
git ls-files -co --exclude-standard -z \
    | tar --null -T - -czf - \
    | ssh "${ssh_opts[@]}" "$REMOTE" \
        "mkdir -p '$REMOTE_DIR' && tar -xzf - -C '$REMOTE_DIR'"
echo "code deployed"
