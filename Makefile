.PHONY: setup run test test-unit test-smoke waiver-alerts

VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
STREAMLIT := $(VENV)/bin/streamlit
PYTEST := $(VENV)/bin/pytest

# Create venv, install deps, copy .env, and enable the pre-push test hook
setup:
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt
	test -f .env || cp .env.example .env
	git config core.hooksPath .githooks

# Launch the Streamlit app
run:
	$(STREAMLIT) run main.py

# Run the full test suite
test:
	$(PYTEST)

# Run unit tests only
test-unit:
	$(PYTEST) tests/common/

# Run smoke tests only
test-smoke:
	$(PYTEST) tests/draft/ tests/classic/ tests/fpl/

# Run Discord waiver alerts (same entrypoint used by GitHub Actions)
waiver-alerts:
	$(PYTHON) -m scripts.common.waiver_alerts
