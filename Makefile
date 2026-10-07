SHELL := bash

# Remote target. Override on the command line, e.g.:
#   make deploy REMOTE=root@1.2.3.4 PORT=40022 REMOTE_DIR=/workspace/assignment2
REMOTE     ?= root@HOST
PORT       ?= 22
REMOTE_DIR ?= /workspace/assignment2
# Local assignment1-basics repo, pushed to <parent-of-REMOTE_DIR>/assignment1-basics
# so the editable path dep `../assignment1-basics` resolves on the remote.
BASICS_DIR ?= ../assignment1-basics
SSH_KEY    ?=

export REMOTE PORT REMOTE_DIR BASICS_DIR SSH_KEY

.PHONY: help deploy deploy-code deploy-basics

help:
	@printf '%s\n' \
		'targets:' \
		'  deploy         push assignment2 + assignment1-basics' \
		"  deploy-code    assignment2 -> remote:$$REMOTE_DIR" \
		"  deploy-basics  assignment1-basics -> remote:<parent>/assignment1-basics" \
		'' \
		'notes:' \
		'  sends tracked + untracked-not-ignored files (respects .gitignore,' \
		'  includes uncommitted edits). Override vars, e.g.:' \
		'    make deploy REMOTE=root@1.2.3.4 PORT=40022'

deploy: deploy-code deploy-basics

deploy-code:
	scripts/deploy_code.sh

deploy-basics:
	scripts/deploy_basics.sh
