# Use bash so RANDOM, brace-groups, and `set -o pipefail` work reliably.
SHELL := /bin/bash

STACK         ?= swarm
COMPOSE_FILE  ?= docker-compose.$(STACK).yml
PROJECT_NAME  ?= dvd-$(STACK)
INSTANCES     ?=
RAM_GB        ?= 16
# Prefer the project venv so sudo -E picks up the right Python version.
_VENV_PY      := $(abspath ../.venv/bin/python3)
PYTHON        := $(if $(wildcard $(_VENV_PY)),$(_VENV_PY),python3)
START_INDEX   ?= auto
COMPOSE       := docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE)

.PHONY: help generate generate-ram pull up down status logs sysctl-check sim sim-multi sim-random sim-benign follow

help:
	@echo "DVD Swarm — headless litemode instances"
	@echo ""
	@echo "  make generate                  Auto-size for $(RAM_GB) GB RAM (~9 instances, with GCS)"
	@echo "  make generate NO_GCS=1         Omit QGroundControl (~14 instances)"
	@echo "  make generate INSTANCES=5      Override instance count explicitly"
	@echo "  make generate RAM_GB=32        Auto-size for a different host"
	@echo "  make generate STACK=survey     Use docker-compose.survey.yml"
	@echo "  make generate START_INDEX=12   Start at a fixed instance index"
	@echo ""
	@echo "  make pull                      Pull all lite images (run once before 'up')"
	@echo "  make up                        Start all instances detached"
	@echo "  make down                      Stop and remove all instances + volumes"
	@echo "  make status                    Show running DVD container count"
	@echo "  make logs                      Stream logs from all instances"
	@echo "  make sysctl-check              Warn if inotify limits may be too low"
	@echo "  make sim N=100 START=10 END=1200 ATTACK=attitude-spoof    Full attack-cycle run"
	@echo ""
	@echo "  make sim-multi N=15 START=40 END=120 ATTACK=attitude_spoof RUNS=10"
	@echo "      Run 'sim' RUNS times with identical parameters."
	@echo ""
	@echo "  make sim-random N=15 ATTACK=attitude_spoof RUNS=10 \\"
	@echo "       [START_MIN=40 START_MAX=70 DUR_MIN=30 DUR_MAX=80 END=150]"
	@echo "      Run RUNS sims with random attack-window timing per iteration."
	@echo ""
	@echo "  make sim-benign N=50 END=300 RUNS=5"
	@echo "      Run RUNS benign-only captures (no attack, label=null on every row)."
	@echo ""
	@echo "  make follow"
	@echo "      tail -f the most-recent logs/*.log file."
	@echo ""
	@echo "Detached usage (so your terminal stays free):"
	@echo "  nohup make sim-benign N=50 END=300 RUNS=5 > /dev/null 2>&1 &"
	@echo "  make follow                  # watch the log in this or another terminal"
	@echo ""
	@echo "Memory per instance: ~1.5 GB with GCS (default) | ~1 GB no GCS"
	@echo "Recommended for 16 GB host: 9 instances (with GCS)  |  14 instances (no GCS)"
	@echo ""
	@echo "Two-terminal example: 50 instances each, collision-free"
	@echo "  Terminal 1:"
	@echo "    make generate INSTANCES=50 STACK=swarm-a START_INDEX=1"
	@echo "    make up STACK=swarm-a"
	@echo "  Terminal 2:"
	@echo "    make generate INSTANCES=50 STACK=swarm-b START_INDEX=51"
	@echo "    make up STACK=swarm-b"
	@echo ""
	@echo "  swarm-a: instances 1-50, companion ports 3001-3050,"
	@echo "           simulator ports 8001-8050,"
	@echo "           subnets 10.13.1.0/24 through 10.13.50.0/24"
	@echo "  swarm-b: instances 51-100, companion ports 3051-3100,"
	@echo "           simulator ports 8051-8100,"
	@echo "           subnets 10.13.51.0/24 through 10.13.100.0/24"

# ── Generate ──────────────────────────────────────────────────────────────────

# GCS is on by default; NO_GCS=1 turns it off
_NO_GCS_FLAG := $(if $(NO_GCS),--no-gcs,)
_START_FLAG  := $(if $(filter auto,$(START_INDEX)),--auto-start-index,--start-index $(START_INDEX))

generate:
ifdef INSTANCES
	$(PYTHON) ./generate_swarm.py --instances $(INSTANCES) $(_NO_GCS_FLAG) $(_START_FLAG) --out $(COMPOSE_FILE)
else
	$(PYTHON) ./generate_swarm.py --ram-gb $(RAM_GB) $(_NO_GCS_FLAG) $(_START_FLAG) --out $(COMPOSE_FILE)
endif

# ── Image management ──────────────────────────────────────────────────────────

pull:
	@[ -f $(COMPOSE_FILE) ] || $(MAKE) generate
	$(COMPOSE) pull

# ── Lifecycle ─────────────────────────────────────────────────────────────────

up: $(COMPOSE_FILE)
	$(COMPOSE) up -d

down: $(COMPOSE_FILE)
	$(COMPOSE) down -v --remove-orphans

# ── Observability ─────────────────────────────────────────────────────────────

status:
	@echo "Running DVD containers:"
	@echo "  flight-controllers    : $$(docker ps -q --filter 'name=flight-controller-lite-' | wc -l)"
	@echo "  companion-computers   : $$(docker ps -q --filter 'name=companion-computer-lite-' | wc -l)"
	@echo "  ground-control-stns   : $$(docker ps -q --filter 'name=ground-control-station-lite-' | wc -l)"
	@echo "  simulators            : $$(docker ps -q --filter 'name=simulator-lite-' | wc -l)"

logs: $(COMPOSE_FILE)
	$(COMPOSE) logs -f

# ── Host prerequisites ────────────────────────────────────────────────────────

sysctl-check:
	@echo "=== inotify limits ==="
	@echo "  max_user_instances : $$(cat /proc/sys/fs/inotify/max_user_instances)  (need ≥1024 for 14 instances)"
	@echo "  max_user_watches   : $$(cat /proc/sys/fs/inotify/max_user_watches)"
	@echo ""
	@echo "To raise (temporary):"
	@echo "  sudo sysctl -w fs.inotify.max_user_instances=8192"
	@echo "  sudo sysctl -w fs.inotify.max_user_watches=524288"

# ── Ensure compose file exists before using it ───────────────────────────────

$(COMPOSE_FILE):
	@echo "$(COMPOSE_FILE) not found — run 'make generate' first"
	@exit 1

# ── Full attack-cycle simulation ──────────────────────────────────────────────

SIZE   ?= $(N)
START  ?= 10
END    ?= 1200
ATTACK ?= attitude-spoof

# Every sim* recipe tees its output to a UTC-timestamped file under
# <project_root>/logs/ via this snippet:
#   set -o pipefail
#   $(PYTHON) -u -m swarm.sim run ... 2>&1 | tee "$$LOG"
# `-u` keeps stdout unbuffered so tee shows live progress, and pipefail
# bubbles up python failures (otherwise tee's success would hide them).

.PHONY: sim
sim:
	@if [ -z "$(SIZE)" ]; then echo "ERROR: pass N=<swarm size>"; exit 1; fi
	@cd .. && mkdir -p logs && \
	LOG="logs/sim_$$(date -u +%Y%m%dT%H%M%SZ).log"; \
	echo "[make] logging to $$LOG"; \
	set -o pipefail; \
	$(PYTHON) -u -m swarm.sim run \
	    --size $(SIZE) --start $(START) --end $(END) \
	    --attack $(ATTACK) 2>&1 | tee "$$LOG"

# ── Multi-run loops ───────────────────────────────────────────────────────────
# Knobs:
#   RUNS                  How many iterations (default 5)
#   N / SIZE              Swarm size per run
#   START / END           Attack window (sim-multi uses these literally;
#                         sim-random treats END as the cap for the random draw)
#   START_MIN/START_MAX   Random START range for sim-random
#   DUR_MIN/DUR_MAX       Random attack-duration range for sim-random
#   ATTACK                Attack type (default: $(ATTACK))
#
# Each multi-run target produces ONE aggregate log per invocation that captures
# every iteration's full output (including the `=== run X/Y ===` headers).
# Iterations call swarm.sim directly (no recursive $(MAKE)) so everything lands
# in a single coherent log file.

RUNS       ?= 5
START_MIN  ?= 40
START_MAX  ?= 70
DUR_MIN    ?= 30
DUR_MAX    ?= 80

# Run the same `sim` config RUNS times.
sim-multi:
	@if [ -z "$(SIZE)" ]; then echo "ERROR: pass N=<swarm size>"; exit 1; fi
	@cd .. && mkdir -p logs && \
	LOG="logs/sim-multi_$$(date -u +%Y%m%dT%H%M%SZ).log"; \
	echo "[make] logging entire loop to $$LOG"; \
	set -o pipefail; \
	{ \
	  for i in $$(seq 1 $(RUNS)); do \
	    echo ""; \
	    echo "=== run $$i/$(RUNS)  N=$(SIZE) START=$(START) END=$(END) ATTACK=$(ATTACK) ==="; \
	    $(PYTHON) -u -m swarm.sim run \
	        --size $(SIZE) --start $(START) --end $(END) \
	        --attack $(ATTACK) || exit 1; \
	  done; \
	} 2>&1 | tee "$$LOG"

# Run RUNS sims, randomising attack window per iteration.
# Each iteration draws START ∈ [START_MIN, START_MAX], duration ∈ [DUR_MIN, DUR_MAX],
# computes END = START + duration (capped at $(END) so the mission generator sizes correctly).
sim-random:
	@if [ -z "$(SIZE)" ]; then echo "ERROR: pass N=<swarm size>"; exit 1; fi
	@cd .. && mkdir -p logs && \
	LOG="logs/sim-random_$$(date -u +%Y%m%dT%H%M%SZ).log"; \
	echo "[make] logging entire loop to $$LOG"; \
	set -o pipefail; \
	SPAN=$$(($(START_MAX) - $(START_MIN) + 1)); \
	DSPAN=$$(($(DUR_MAX) - $(DUR_MIN) + 1)); \
	{ \
	  for i in $$(seq 1 $(RUNS)); do \
	    S=$$(( $(START_MIN) + RANDOM % $$SPAN )); \
	    D=$$(( $(DUR_MIN)   + RANDOM % $$DSPAN )); \
	    E=$$(( S + D )); \
	    if [ $$E -gt $(END) ]; then E=$(END); fi; \
	    echo ""; \
	    echo "=== run $$i/$(RUNS)  N=$(SIZE) START=$$S END=$$E ATTACK=$(ATTACK) ==="; \
	    $(PYTHON) -u -m swarm.sim run \
	        --size $(SIZE) --start $$S --end $$E \
	        --attack $(ATTACK) || exit 1; \
	  done; \
	} 2>&1 | tee "$$LOG"

# Tail the most-recent log file under <project_root>/logs/. Handy when
# you've launched a sim with `nohup make sim-X ... > /dev/null 2>&1 &` and
# want to monitor progress without occupying the shell that launched it.
follow:
	@cd .. && LATEST=$$(ls -t logs/*.log 2>/dev/null | head -1); \
	if [ -z "$$LATEST" ]; then \
	    echo "no logs in logs/. Run 'make sim*' first."; exit 1; \
	fi; \
	echo "==> tail -f $$LATEST   (Ctrl+C to stop)"; \
	tail -f "$$LATEST"

# Run RUNS benign-only captures. Every row in every drone's CSV gets
# attack_type=null — useful for building a clean baseline before mixing
# attack runs in. START is ignored under --attack none.
sim-benign:
	@if [ -z "$(SIZE)" ]; then echo "ERROR: pass N=<swarm size>"; exit 1; fi
	@cd .. && mkdir -p logs && \
	LOG="logs/sim-benign_$$(date -u +%Y%m%dT%H%M%SZ).log"; \
	echo "[make] logging entire loop to $$LOG"; \
	set -o pipefail; \
	{ \
	  for i in $$(seq 1 $(RUNS)); do \
	    echo ""; \
	    echo "=== benign run $$i/$(RUNS)  N=$(SIZE) END=$(END) ==="; \
	    $(PYTHON) -u -m swarm.sim run \
	        --size $(SIZE) --start 0 --end $(END) \
	        --attack none || exit 1; \
	  done; \
	} 2>&1 | tee "$$LOG"
