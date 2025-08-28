# FraudLens Makefile
# Author: Yobie Benjamin
# Date: 2025-08-26 18:34:00 PDT

.PHONY: help install test lint format clean docker-build docker-run docs

# Default target
help:
	@echo "FraudLens Development Commands"
	@echo "=============================="
	@echo "make install      - Install dependencies"
	@echo "make install-dev  - Install with development dependencies"
	@echo "make test        - Run tests"
	@echo "make lint        - Run linting"
	@echo "make format      - Format code"
	@echo "make typecheck   - Run type checking"
	@echo "make clean       - Clean build artifacts"
	@echo "make docker-build - Build Docker image"
	@echo "make docker-run  - Run Docker container"
	@echo "make docs        - Build documentation"
	@echo "make demo        - Run demo"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

# Testing
test:
	pytest tests/ -v --cov=fraudlens --cov-report=html --cov-report=term

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

# Code quality
lint:
	ruff check fraudlens/
	mypy fraudlens/

format:
	black fraudlens/ tests/
	ruff check --fix fraudlens/

typecheck:
	mypy fraudlens/ --ignore-missing-imports

# Cleaning
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/

# Docker
docker-build:
	docker build -t fraudlens:latest .

docker-run:
	docker run -p 8000:8000 -v $(PWD)/models:/app/models fraudlens:latest

docker-shell:
	docker run -it --rm fraudlens:latest /bin/bash

# Documentation
docs:
	mkdocs build

docs-serve:
	mkdocs serve

# Development
demo:
	python -m fraudlens.demo

run:
	uvicorn fraudlens.api.main:app --reload --host 0.0.0.0 --port 8000

# Benchmarking
benchmark:
	python -m fraudlens.benchmarks.run_benchmarks

# Model management
download-models:
	python -m fraudlens.models.download_models

quantize-models:
	python -m fraudlens.models.quantize --bits 8

# Database
db-migrate:
	alembic upgrade head

db-rollback:
	alembic downgrade -1

# CI/CD
ci-test:
	pytest tests/ --cov=fraudlens --cov-report=xml

ci-lint:
	ruff check fraudlens/ --exit-non-zero-on-fix
	mypy fraudlens/ --ignore-missing-imports

# Version management
version:
	@python -c "import fraudlens; print(fraudlens.__version__)"

bump-patch:
	bump2version patch

bump-minor:
	bump2version minor

bump-major:
	bump2version major