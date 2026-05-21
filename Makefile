# Root passthrough: all targets live in swarm/Makefile.
# Forward each cmd-line goal individually via $@ rather than $(MAKECMDGOALS).
# Some make versions leak command-line variable assignments (KEEP_RAW=0,
# NAME=foo, etc.) into MAKECMDGOALS, which then causes sub-make to try to
# build them as targets. Variable assignments still propagate to sub-make
# automatically via MAKEFLAGS, so $@ is both safer and equivalent.
.DEFAULT_GOAL := help

%:
	@$(MAKE) -C swarm $@

.PHONY: help
help:
	@$(MAKE) -C swarm help
