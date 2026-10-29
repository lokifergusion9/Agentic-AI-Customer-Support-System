# Developer commands (W5.8). Without make: `uv run python -m scripts.dev <command>`.
DEV = uv run python -m scripts.dev

.PHONY: setup migrate seed test smoke evaluate report

setup:
	$(DEV) setup

migrate:
	$(DEV) migrate

seed:
	$(DEV) seed

test:
	$(DEV) test

smoke:
	$(DEV) smoke

# make evaluate PLAN=pilot
evaluate:
	$(DEV) evaluate --plan $(PLAN)

# make report EXPERIMENT=exp_...
report:
	$(DEV) report $(EXPERIMENT)
