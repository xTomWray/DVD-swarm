# Root passthrough: all targets live in swarm/Makefile
.DEFAULT_GOAL := help

%:
	@$(MAKE) -C swarm $(MAKECMDGOALS)

.PHONY: help
help:
	@$(MAKE) -C swarm help
