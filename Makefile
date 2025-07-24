.PHONY: help install install-dev format lint test test-cov clean setup-pre-commit

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	pip install -r requirements.txt

install-dev:  ## Install development dependencies
	pip install -r requirements.txt
	pip install pre-commit
	pre-commit install

setup-pre-commit:  ## Set up pre-commit hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

format:  ## Format code with black and isort
	black src/ experiments/ tests/
	isort src/ experiments/ tests/

lint:  ## Run linting checks
	flake8 src/ experiments/ tests/
	mypy src/
	black --check src/ experiments/ tests/
	isort --check-only src/ experiments/ tests/

test:  ## Run tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-fast:  ## Run fast tests only (exclude slow tests)
	pytest tests/ -v -m "not slow"

clean:  ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

setup:  ## Initial project setup
	$(MAKE) install-dev
	$(MAKE) setup-pre-commit
	@echo "Project setup complete! Run 'make help' to see available commands."