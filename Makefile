.PHONY: help install dev test test-cov lint format eval clean docker-build docker-up

VENV ?= .venv
PYTHON ?= $(VENV)/bin/python3
PYTEST ?= $(VENV)/bin/pytest
UVICORN ?= $(VENV)/bin/uvicorn
RUFF ?= $(VENV)/bin/ruff

help: ## Show this help message
	@echo "Second Brain — Developer Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Set up virtualenv and install all dependencies via uv
	uv venv --python 3.12
	uv pip install -e ".[dev]"

dev: ## Run the development server with live reload on port 8000
	$(UVICORN) backend.app.main:app --host 0.0.0.0 --port 8000 --reload

test: ## Run the isolated test suite
	$(PYTEST) tests/ -v

test-cov: ## Run test suite with line coverage report
	$(PYTEST) tests/ --cov=backend --cov-report=term-missing

lint: ## Run code linter
	$(RUFF) check backend/ tests/

format: ## Automatically format python files
	$(RUFF) format backend/ tests/

eval: ## Run the 4-way RAG Triad benchmark evaluation CLI
	$(PYTHON) -m backend.app.evaluation.benchmark_cli

docker-build: ## Build the production Docker image
	docker build -t second-brain:latest .

docker-up: ## Launch with Docker Compose
	docker compose up --build -d

clean: ## Remove python cache, pytest cache, and temporary artifacts
	rm -rf .pytest_cache .coverage htmlcov __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
