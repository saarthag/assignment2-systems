SHELL := bash

# Remote target. Override on the command line, e.g.:
#   make deploy REMOTE=root@1.2.3.4 PORT=40022 REMOTE_DIR=/workspace/assignment2
REMOTE     ?= root@HOST
PORT       ?= 22
REMOTE_DIR ?= /workspace/assignment2
SSH_KEY    ?=

export REMOTE PORT REMOTE_DIR SSH_KEY

.PHONY: help deploy deploy-code

help:
	@printf '%s\n' \
		'targets:' \
		'  deploy         push assignment2 to the remote' \
		"  deploy-code    assignment2 -> remote:$$REMOTE_DIR" \
		'' \
		'notes:' \
		'  cs336-basics is resolved from its git remote by `uv sync` on the' \
		'  remote, so only the assignment2 tree is transferred.' \
		'  sends tracked + untracked-not-ignored files (respects .gitignore,' \
		'  includes uncommitted edits). Override vars, e.g.:' \
		'    make deploy REMOTE=root@1.2.3.4 PORT=40022'

deploy: deploy-code

deploy-code:
	scripts/deploy_code.sh
