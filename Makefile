# Follow Goblin Makefile
#
# Usage: make help

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ─── Observability ───────────────────────────────────────────────────────

DAYS ?= 7
PROJECT ?= follow_goblin_4

.PHONY: cost

cost:  ## LLM spend for this project (DAYS=7)
	@python -m llm_client cost --group-by model --days $(DAYS) --project $(PROJECT)

# ─── Development ─────────────────────────────────────────────────────────

.PHONY: lint install

lint:  ## Run ruff linter
	@ruff check . 2>/dev/null || true

install:  ## Install in editable mode
	@pip install -e . 2>/dev/null || true

# ─── Help ────────────────────────────────────────────────────────────────

.PHONY: help

help:  ## Show available targets
	@echo "Follow Goblin"
	@echo ""
	@grep -E '^[a-z][-a-zA-Z0-9_]*:.*## ' $(MAKEFILE_LIST) | \
		awk -F ':.*## ' '{printf "  make %-20s %s\n", $$1, $$2}'
