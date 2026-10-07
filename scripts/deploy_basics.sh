#!/usr/bin/env bash
# shellcheck disable=SC2029  # remote paths are intentionally expanded client-side
# Ship the assignment1-basics (cs336-basics package) source tree next to the
# assignment2 tree so the editable path dependency resolves on the remote.
#
# assignment2's pyproject.toml declares:
#     cs336-basics = { path = "../assignment1-basics/", editable = true }
# so for REMOTE_DIR=/workspace/assignment2 the package must land at
# /workspace/assignment1-basics.
#
# Like deploy_code.sh this uses `git ls-files -co --exclude-standard`, so it
# respects assignment1's .gitignore (data/, experiments/, tokenizers/, .venv/,
# __pycache__/ ... never leave the local machine) while still including
# uncommitted edits.
#
# Env vars (set by the Makefile or the shell):
#   REMOTE       user@host of the instance            (required)
#   PORT         ssh port                              (default 22)
#   REMOTE_DIR   assignment2 dir on the remote         (default /workspace/assignment2)
#   BASICS_DIR   local assignment1-basics repo         (default ../assignment1-basics)
#   BASICS_NAME  remote folder name                    (default assignment1-basics)
#   SSH_KEY      optional identity file
set -euo pipefail

REMOTE="${REMOTE:?set REMOTE=user@host}"
PORT="${PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/assignment2}"
BASICS_DIR="${BASICS_DIR:-../assignment1-basics}"
BASICS_NAME="${BASICS_NAME:-assignment1-basics}"
SSH_KEY="${SSH_KEY:-}"

ssh_opts=(-p "$PORT" -o StrictHostKeyChecking=accept-new)
[[ -n "$SSH_KEY" ]] && ssh_opts+=(-i "$SSH_KEY")

[[ -d "$BASICS_DIR" ]] || { echo "error: assignment1-basics dir not found: $BASICS_DIR" >&2; exit 1; }
basics_root="$(git -C "$BASICS_DIR" rev-parse --show-toplevel)"

# Strip a trailing slash so dirname gives the true parent of the assignment2 dir.
remote_dir="${REMOTE_DIR%/}"
remote_basics="$(dirname "$remote_dir")/$BASICS_NAME"

echo "basics: $(git -C "$basics_root" rev-parse --short HEAD) (+working tree) $basics_root -> $REMOTE:$remote_basics"
cd "$basics_root"
git ls-files -co --exclude-standard -z \
    | tar --null -T - -czf - \
    | ssh "${ssh_opts[@]}" "$REMOTE" \
        "mkdir -p '$remote_basics' && tar -xzf - -C '$remote_basics'"
echo "basics deployed"
