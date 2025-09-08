.PHONY: help install dev-install clean test lint format build upload-test upload docs

help:
	@echo "Available commands:"
	@echo "  install      Install the package in production mode"
	@echo "  dev-install  Install the package in development mode"
	@echo "  clean        Remove build artifacts"
	@echo "  test         Run tests"
	@echo "  lint         Run code linting"
	@echo "  format       Format code with black"
	@echo "  build        Build the package"
	@echo "  upload-test  Upload to TestPyPI"
	@echo "  upload       Upload to PyPI"
	@echo "  docs         Generate documentation"

install:
	python -m pip install -r requirements.txt

dev-install:
	python -m pip install -r requirements.txt

clean:
	@if exist build\ rmdir /s /q build
	@if exist dist\ rmdir /s /q dist
	@if exist *.egg-info rmdir /s /q *.egg-info
	@if exist .pytest_cache rmdir /s /q .pytest_cache
	@if exist .coverage del /q .coverage
	@if exist htmlcov\ rmdir /s /q htmlcov
	@for /r %%i in (*.pyc) do del "%%i"
	@for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"

test:
	python -m pytest

lint:
	python -m flake8 weather/
	python -m black --check weather/

format:
	python -m black weather/

build: clean
	python -m build

upload-test: build
	poetry config repositories.testpypi https://test.pypi.org/legacy/
	poetry publish -r testpypi

upload: build
	poetry publish

# Development helpers
notebook:
	jupyter notebook

api-dev:
	python -m uvicorn app.main:app --reload

# Project-specific commands
download-data:
	python -c "from weather.dataset import download_weather_data; download_weather_data()"

train-models:
	@echo "Training rain prediction model..."
	python -m weather.modeling.train --model rain_or_not
	@echo "Training precipitation volume model..."
	python -m weather.modeling.train --model precipitation_fall

# Quality checks
check: lint test
	@echo "All checks passed!"

# Setup commands
setup-poetry:
	python -m pip install poetry
	poetry config virtualenvs.in-project true

init: setup-poetry dev-install
	@echo "Project initialized with Poetry!"