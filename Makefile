SHELL := bash

# Remote target. Override on the command line, e.g.:
#   make deploy REMOTE=root@1.2.3.4 PORT=40022 REMOTE_DIR=/workspace/assignment2
REMOTE     ?= root@HOST
PORT       ?= 22
REMOTE_DIR ?= /workspace/assignment2
SSH_KEY    ?=

# Local dashboard bind address.
DASHBOARD_HOST ?= 127.0.0.1
DASHBOARD_PORT ?= 8765

export REMOTE PORT REMOTE_DIR SSH_KEY

.PHONY: help dashboard deploy deploy-code

help:
	@printf '%s\n' \
		'targets:' \
		'  dashboard      launch the local benchmark dashboard' \
		'  deploy         push assignment2 to the remote' \
		"  deploy-code    assignment2 -> remote:$$REMOTE_DIR" \
		'' \
		'notes:' \
		'  dashboard deps live in the `dashboard` uv group; the target syncs' \
		'  them on demand, leaving the assignment environment untouched.' \
		'  cs336-basics is resolved from its git remote by `uv sync` on the' \
		'  remote, so only the assignment2 tree is transferred.' \
		'  sends tracked + untracked-not-ignored files (respects .gitignore,' \
		'  includes uncommitted edits). Override vars, e.g.:' \
		'    make deploy REMOTE=root@1.2.3.4 PORT=40022' \
		'    make dashboard DASHBOARD_PORT=9000'

dashboard:
	uv run --group dashboard python dashboard/server.py --host "$(DASHBOARD_HOST)" --port "$(DASHBOARD_PORT)"

deploy: deploy-code

deploy-code:
	scripts/deploy_code.sh
