.PHONY: all clean format lint type-check test install dev dist venv

# Default target executed when no arguments are given to make.
all: clean format lint type-check test

# Clean up build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete

# Format code with ruff
format:
	ruff format .
	ruff check --fix .

# Lint with ruff
lint:
	ruff check .

# Type checking with mypy
type-check:
	mypy spacy_glossbert

# Run tests with pytest
test:
	pytest

# Install the package in development mode
dev:
	pip install -e ".[dev]"

# Install the package
install:
	pip install .

# Build source distribution and wheel
dist:
	python -m build 

# Create virtualenv using uv
venv:
	@echo "Creating virtualenv using uv..."
	@if ! command -v uv &> /dev/null; then \
		echo "uv not found. Installing uv..."; \
		pip install uv; \
	fi
	@if [ ! -d ".venv" ]; then \
		uv venv .venv; \
		echo "Virtual environment created at .venv/"; \
	else \
		echo "Virtual environment already exists at .venv/"; \
	fi
	@echo "Generating requirements files..."
	@uv pip compile pyproject.toml -o requirements.txt
	@uv pip compile pyproject.toml --extra dev -o requirements-dev.txt
	@echo "Installing dependencies..."
	@. .venv/bin/activate && uv pip install -r requirements.txt -r requirements-dev.txt
	@echo "Virtual environment setup complete. Activate with 'source .venv/bin/activate'"

# Install in development mode
dev-install: venv
	@. .venv/bin/activate && pip install -e . 