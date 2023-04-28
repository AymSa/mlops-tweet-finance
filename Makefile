# Makefile
SHELL = /bin/bash

.PHONY: help
help:
	@echo "Commands:"
	@echo "venv    : creates a virtual environment."
	@echo "style   : executes style formatting."
	@echo "clean   : cleans all unnecessary files."

# Environment
.PHONY: env
.ONESHELL:
env:
	python3 -m venv .venv
	source venv/bin/activate
	python3 -m pip install pip setuptools wheel
	python3 -m pip install -e .
	python3 -m pip install -e ".[dev]"
	pre-commit install
	pre-commit autoupdate

# Styling
.PHONY: style
style :
	black .
	flake8
	isort .

# Cleaning
.PHONY: clean
.ONESHELL:
clean: style
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . | grep -E ".trash" | xargs rm -rf
	rm -f .coverage

# Docs
.PHONY: doc
doc : style
	python3 -m mkdocs gh-deploy

# MLflow experiments
MLFLOW_DIR := "notebooks/experiments"
.PHONY: mlflow
mlflow:
	mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri ${MLFLOW_DIR}


.PHONY: rest-prod
rest-prod:
	gunicorn -c app/gunicorn.py -k uvicorn.workers.UvicornWorker app.api:app

.PHONY: rest-dev
rest-dev:
	uvicorn app.api:app \
	--host 0.0.0.0 \
	--port 8000 \
	--reload \
	--reload-dir src \
	--reload-dir app


.PHONY : test
test:
	python -m pytest -m "not training"
	cd tests
	great_expectations checkpoint run tweets
	great_expectations checkpoint run labeled_tweets
