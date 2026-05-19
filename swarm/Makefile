STACK         ?= swarm
COMPOSE_FILE  ?= docker-compose.$(STACK).yml
PROJECT_NAME  ?= dvd-$(STACK)
INSTANCES     ?=
RAM_GB        ?= 16
PYTHON        := python3
START_INDEX   ?= auto
COMPOSE       := docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE)

.PHONY: help generate generate-ram pull up down status logs sysctl-check sim

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

pull: $(COMPOSE_FILE)
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

.PHONY: sim
sim:
	cd .. && $(PYTHON) -m swarm.sim run \
	    --size $(SIZE) --start $(START) --end $(END) \
	    --attack $(ATTACK)
