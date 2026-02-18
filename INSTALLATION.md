# Installation Guide

## Overview

Visual RAG Document Explorer uses **standard Python packaging (PEP 621)** with **setuptools** as the build backend. This is NOT a Poetry project.

## Prerequisites

- **Python 3.11+** (Python 3.12 recommended)
- **Docker and Docker Compose** (for vector databases)
- **Git** (for cloning the repository)

## Installation Methods

### Method 1: Using Make (Recommended)

The project includes a Makefile with convenient commands:

```bash
# Clone the repository
git clone <repository-url>
cd Visual_RAG_Document_Explore

# Install with development dependencies
make install

# Or install production dependencies only
make install-prod
```

### Method 2: Using pip Directly

```bash
# Clone the repository
git clone <repository-url>
cd Visual_RAG_Document_Explore

# Install with development dependencies
python3 -m pip install -e ".[dev]"

# Or install production dependencies only
python3 -m pip install -e .
```

### Method 3: Using Virtual Environment (Recommended for Development)

```bash
# Clone the repository
git clone <repository-url>
cd Visual_RAG_Document_Explore

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -e ".[dev]"
```

## Configuration

### 1. Environment Variables

Copy the example environment file and configure your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```bash
# Required
OPENAI_API_KEY=sk-...
VOYAGE_API_KEY=pa-...

# Optional
COHERE_API_KEY=...
OPENROUTER_API_KEY=...

# Vector Database URLs (defaults shown)
QDRANT_URL=http://localhost:6333
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

### 2. Start Vector Databases

The project requires either Qdrant or Milvus (or both) for vector storage:

```bash
# Start all services (Qdrant, Milvus, etcd, MinIO)
make docker-up

# Or using docker-compose directly
docker-compose up -d
```

Verify services are running:
- **Qdrant**: http://localhost:6333/dashboard
- **Milvus**: http://localhost:19530   Health Check http://localhost:9091/healthz
- **MinIO Console**: http://localhost:9001

## Running the Application

### Start the Streamlit UI

```bash
# Using Make
make run

# Or directly
streamlit run app.py
```

The application will be available at: **http://localhost:8501**

## Verification

### Test Installation

```bash
# Run tests
make test

# Or using pytest directly
pytest tests/ -v
```

### Check Dependencies

```bash
# List installed packages
pip list | grep -E "langchain|streamlit|qdrant|pymilvus"

# Verify Python version
python3 --version  # Should be 3.11 or higher
```

## Common Issues

### Issue: "pip: command not found"

**Solution:** Use `python3 -m pip` instead of `pip`:
```bash
python3 -m pip install -e ".[dev]"
```

### Issue: "poetry install" fails

**Solution:** This project does NOT use Poetry. Use pip instead:
```bash
# Correct
pip install -e ".[dev]"

# Incorrect
poetry install  # ❌ Don't use this
```

### Issue: Import errors after installation

**Solution:** Ensure you're in the project directory and using the correct Python environment:
```bash
# Check current directory
pwd  # Should be .../Visual_RAG_Document_Explore

# Check Python path
python3 -c "import sys; print(sys.path)"

# Reinstall in editable mode
pip install -e ".[dev]"
```

### Issue: Vector database connection errors

**Solution:** Ensure Docker services are running:
```bash
# Check Docker containers
docker ps

# Restart services
make docker-down
make docker-up

# Check logs
docker-compose logs qdrant
docker-compose logs milvus-standalone
```

### Issue: Missing API keys

**Solution:** Verify your `.env` file:
```bash
# Check if .env exists
ls -la .env

# Verify required keys are set
grep -E "OPENAI_API_KEY|VOYAGE_API_KEY" .env
```

## Development Setup

### Install Development Tools

Development dependencies include:
- `pytest` - Testing framework
- `pytest-asyncio` - Async test support
- `pytest-cov` - Coverage reporting
- `ruff` - Linting and formatting
- `black` - Code formatting
- `mypy` - Type checking
- `ipython` - Interactive shell

These are automatically installed with:
```bash
make install
# or
pip install -e ".[dev]"
```

### Code Quality Commands

```bash
# Run linter
make lint

# Format code
make format

# Run tests with coverage
pytest tests/ --cov=core --cov=agents --cov=ui --cov-report=html

# Type checking
mypy core/ agents/ ui/
```

## Updating Dependencies

### Add New Dependencies

Edit [`pyproject.toml`](pyproject.toml) and add to the `dependencies` list:

```toml
[project]
dependencies = [
    "existing-package>=1.0.0",
    "new-package>=2.0.0",  # Add here
]
```

Then reinstall:
```bash
pip install -e ".[dev]"
```

### Update Existing Dependencies

```bash
# Update all packages
pip install --upgrade -e ".[dev]"

# Update specific package
pip install --upgrade package-name
```

## Uninstallation

### Remove the Package

```bash
pip uninstall visual-rag-explorer
```

### Stop Docker Services

```bash
make docker-down
# or
docker-compose down -v  # -v removes volumes
```

### Clean Build Artifacts

```bash
make clean
```

## Additional Resources

- **Project Structure**: See [`README.md`](README.md)
- **Architecture**: See [`plans/architecture.md`](plans/architecture.md)
- **API Documentation**: See individual module docstrings
- **Phase Completion**: See `PHASE*_COMPLETION_SUMMARY.md` files

## Support

For issues or questions:
1. Check this installation guide
2. Review the [README.md](README.md)
3. Check existing GitHub issues
4. Open a new issue with:
   - Python version (`python3 --version`)
   - OS information
   - Error messages
   - Steps to reproduce
