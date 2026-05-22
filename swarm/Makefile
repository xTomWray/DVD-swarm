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

.PHONY: help generate generate-ram pull up down status logs sysctl-check sim sim-multi sim-random sim-benign follow archive restore list-archives analyze

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
	@echo "  make analyze                         Canned report on the latest run."
	@echo "  make analyze RUN=output/run_...      Report for a specific run."
	@echo "  make analyze SQL='SELECT ...'        Ad-hoc query against view 'flight'."
	@echo "  make analyze FORCE=1                 Rebuild flight.parquet."
	@echo ""
	@echo "  Tuning knobs (any sim* target):"
	@echo "    GCS_CONCURRENCY=8           (default 12) — max parallel /stage2 + /stage3 POSTs"
	@echo "    STAGE1_CONCURRENCY=16       (default 16) — lower to 8 if docker exec contention at N=100+"
	@echo "    MISSION_REQUEST_TIMEOUT=60  (default 30) — bump if uploads time out under host load"
	@echo "    MISSION_ACK_TIMEOUT=60      (default 30) — bump if MISSION_ACK is missing post-upload"
	@echo "    MISSION_UPLOAD_RETRIES=5    (default 3)  — bump if all retries exhausted"
	@echo "    STAGE_HTTP_TIMEOUT=900      (default 600)— upper bound for one stage script per drone"
	@echo "    STAGE1_SETTLE=30            (default 0 = auto: max(15, N/3))"
	@echo ""
	@echo "  make follow"
	@echo "      tail -f the most-recent logs/*.log file."
	@echo ""
	@echo "  make archive [NAME=label] [KEEP_RAW=0] [ZSTD_LEVEL=10]"
	@echo "      tar+zstd all output/run_* (and matching logs/) into"
	@echo "      archives/<UTC>[_<NAME>].tar.zst, then delete the originals."
	@echo "    KEEP_RAW=0            strip per-run raw/ dirs before archiving (smaller archive)"
	@echo "    ZSTD_LEVEL=10         compression level (1=fastest, 19=smallest; default 10)"
	@echo ""
	@echo "  make restore ARCHIVE=archives/<file>.tar.zst"
	@echo "      Extract a past archive back into output/ for re-training."
	@echo "      (Also accepts legacy .tar.gz archives — tar auto-detects.)"
	@echo ""
	@echo "  make list-archives"
	@echo "      Show all archives with sizes."
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

SIZE                     ?= $(N)
START                    ?= 10
END                      ?= 1200
ATTACK                   ?= attitude-spoof
GCS_CONCURRENCY          ?= 12
STAGE1_CONCURRENCY       ?= 16
MISSION_REQUEST_TIMEOUT  ?= 30
MISSION_ACK_TIMEOUT      ?= 30
MISSION_UPLOAD_RETRIES   ?= 3
STAGE_HTTP_TIMEOUT       ?= 600
STAGE1_SETTLE            ?= 0

# Common knob bundle passed to every swarm.sim invocation. Avoids the
# four sim* recipes drifting from each other when a new knob is added.
_SIM_FLAGS = \
    --gcs-concurrency $(GCS_CONCURRENCY) \
    --stage1-concurrency $(STAGE1_CONCURRENCY) \
    --mission-request-timeout $(MISSION_REQUEST_TIMEOUT) \
    --mission-ack-timeout $(MISSION_ACK_TIMEOUT) \
    --mission-upload-retries $(MISSION_UPLOAD_RETRIES) \
    --stage-http-timeout $(STAGE_HTTP_TIMEOUT) \
    --stage1-settle $(STAGE1_SETTLE)

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
	    --attack $(ATTACK) \
	    $(_SIM_FLAGS) 2>&1 | tee "$$LOG"

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
	        --attack $(ATTACK) \
	        $(_SIM_FLAGS) || exit 1; \
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
	        --attack $(ATTACK) \
	        $(_SIM_FLAGS) || exit 1; \
	  done; \
	} 2>&1 | tee "$$LOG"

# ── Archival ──────────────────────────────────────────────────────────────────
# Workflow:
#   make sim-benign N=50 ...    # generates runs in output/
#   ... data accumulates ...
#   make archive NAME=benign-baseline   # rolls it all into archives/, clears output/
#   make sim-random N=15 ...    # next session starts with a clean output/
#
# To re-train on a past session:
#   make restore ARCHIVE=archives/20260520T020415Z_benign-baseline.tar.zst
#   python swarm/biLSTM-2Layer-Protocol.py --data-dir output/ ...

# Tar+zstd every output/run_* (plus matching logs/sim-*.log) into
# archives/<UTC>[_<NAME>].tar.zst then remove the originals.
#
# zstd -T0 -<level> saturates all host cores; level 10 is a good archival
# balance (smaller than gzip, ~3-5x faster on multi-GB datasets). Override
# with ZSTD_LEVEL=19 for max compression or ZSTD_LEVEL=3 for fast packing.
NAME       ?=
KEEP_RAW   ?= 1
ZSTD_LEVEL ?= 10
archive:
	@cd .. && \
	if [ -z "$$(ls -d output/run_* 2>/dev/null)" ]; then \
	    echo "Nothing to archive — output/ has no run_* dirs."; \
	    exit 0; \
	fi; \
	mkdir -p archives && \
	STAMP=$$(date -u +%Y%m%dT%H%M%SZ); \
	NAME_SUFFIX=""; \
	if [ -n "$(NAME)" ]; then NAME_SUFFIX="_$(NAME)"; fi; \
	ARCHIVE="archives/$${STAMP}$${NAME_SUFFIX}.tar.zst"; \
	N=$$(ls -d output/run_* | wc -l); \
	TOTAL_KB=$$(du -ks output/run_* 2>/dev/null | awk '{s+=$$1} END {print s+0}'); \
	EXCLUDE=""; \
	if [ "$(KEEP_RAW)" = "0" ]; then EXCLUDE="--exclude=output/run_*/raw"; fi; \
	echo "[make] archiving $$N run dir(s) (~$$((TOTAL_KB/1024)) MB raw, KEEP_RAW=$(KEEP_RAW), zstd -T0 -$(ZSTD_LEVEL)) -> $$ARCHIVE"; \
	set -o pipefail; \
	tar $$EXCLUDE \
	    --checkpoint=2000 \
	    --checkpoint-action=ttyout='  [%{%H:%M:%S}T] cp %u  %{w}T\n' \
	    -I 'zstd -T0 -$(ZSTD_LEVEL)' \
	    -cf "$$ARCHIVE" \
	    output/run_* $$(ls logs/sim-*.log logs/sim_*.log 2>/dev/null); \
	echo "[make] removing originals"; \
	rm -rf output/run_*; \
	echo "[make] done. $$ARCHIVE ($$(du -h "$$ARCHIVE" | cut -f1))"

# Extract a past archive back into output/. Existing run_* dirs aren't
# disturbed (timestamps in dir names prevent collisions). `tar -xf` auto-
# detects compression, so both legacy .tar.gz and new .tar.zst work.
restore:
	@if [ -z "$(ARCHIVE)" ]; then \
	    echo "ERROR: pass ARCHIVE=archives/<file>.tar.{zst,gz}"; \
	    echo "Available:"; \
	    ls -1 ../archives/*.tar.zst ../archives/*.tar.gz 2>/dev/null | sed 's/^/  /'; \
	    exit 1; \
	fi
	@cd .. && \
	if [ ! -f "$(ARCHIVE)" ]; then \
	    echo "ERROR: archive not found: $(ARCHIVE)"; exit 1; \
	fi; \
	echo "[make] extracting $(ARCHIVE)…"; \
	tar -xf "$(ARCHIVE)" && \
	echo "[make] done. Run dirs now in output/:"; \
	ls -d output/run_* 2>/dev/null | tail -10

# Fast post-run analysis. First call per run converts csv/drone_*.csv into
# output/<run>/flight.parquet (ZSTD); subsequent calls are sub-second. The
# default report covers attack-label coverage, row volume, message-type mix,
# and VFR_HUD/ATTITUDE dynamics split by attack_type. Pass SQL='...' for an
# ad-hoc query, RUN=output/run_... for a specific run, FORCE=1 to rebuild.
analyze:
	@cd .. && $(PYTHON) -m swarm.dataset.analyze \
	    $(if $(RUN),--run $(RUN)) \
	    $(if $(FORCE),--force) \
	    $(if $(SQL),--sql "$(SQL)")

# List archives newest first with file sizes. Shows both legacy .tar.gz
# and new .tar.zst.
list-archives:
	@cd .. && \
	if [ -z "$$(ls archives/*.tar.zst archives/*.tar.gz 2>/dev/null)" ]; then \
	    echo "No archives yet. Use 'make archive' to create one."; \
	else \
	    ls -lhS archives/*.tar.zst archives/*.tar.gz 2>/dev/null | awk '{print $$5, $$9}'; \
	fi

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
	        --attack none \
	        $(_SIM_FLAGS) || exit 1; \
	  done; \
	} 2>&1 | tee "$$LOG"
