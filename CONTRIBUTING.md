# Contributing to Memoria

Welcome! We're glad you're interested in contributing to Memoria. This document provides guidelines and instructions for contributing.

## Development Setup

1. **Clone the repository**

```bash
git clone https://github.com/Oxygen56/memoria.git
cd memoria
```

2. **Create a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install in development mode**

```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with short traceback
pytest tests/ -v --tb=short

# Run a specific test file
pytest tests/test_memoria.py -v
```

## Code Style

We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting:

- **Line length**: 100 characters
- **Linter**: `ruff check src/ tests/`
- **Formatter**: `ruff format src/ tests/`

Please ensure your code passes linting before submitting a PR:

```bash
ruff check src/ tests/
ruff format --check src/ tests/
```

## Pull Request Workflow

1. **Fork** the repository and create a new branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and ensure:
   - All tests pass (`pytest tests/ -v`)
   - Code passes linting (`ruff check src/ tests/`)
   - New features include appropriate tests

3. **Commit** with a clear, descriptive message:
   ```bash
   git commit -m "feat: add support for X"
   ```

4. **Push** to your fork and open a Pull Request against `main`.

5. **Describe** your changes in the PR description, including:
   - What the change does
   - Why the change is needed
   - Any breaking changes

## Issue Reporting

When reporting a bug, please include:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected vs actual behavior
- Your environment (Python version, OS, Memoria version)
- Any relevant logs or error messages

For feature requests, please describe:

- The problem you're trying to solve
- Your proposed solution
- Any alternatives you've considered

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

Thank you for contributing!
