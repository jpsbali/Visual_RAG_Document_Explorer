.PHONY: install install-prod docker-up docker-down run test lint format clean help

help:
	@echo "Visual RAG Document Explorer - Available Commands:"
	@echo "  make install      - Install project dependencies with dev extras"
	@echo "  make install-prod - Install project dependencies (production only)"
	@echo "  make docker-up    - Start Qdrant, Milvus, etcd, and MinIO services"
	@echo "  make docker-down  - Stop all Docker services"
	@echo "  make run          - Run the Streamlit application"
	@echo "  make test         - Run pytest test suite"
	@echo "  make lint         - Run ruff linter"
	@echo "  make format       - Format code with ruff"
	@echo "  make clean        - Remove build artifacts and caches"

install:
	python3 -m pip install -e ".[dev]"

install-prod:
	python3 -m pip install -e .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

run:
	streamlit run app.py

test:
	pytest tests/ -v

lint:
	ruff check .

format:
	ruff format .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/
