# Contributing to puredata

Thank you for your interest in contributing to puredata.

## Getting started

1. Fork the repository on GitHub
2. Clone your fork: `git clone https://github.com/<your-username>/puredata.py.git`
3. Install in development mode: `pip install -e ".[dev]"`
4. Run the test suite to verify your setup: `pytest`

## Workflow

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make your changes with accompanying tests
3. Ensure all tests pass: `pytest --cov`
4. Ensure code is formatted: `black puredata/ tests/` and `ruff check puredata/ tests/`
5. Push your branch and open a pull request

## Test requirements

- All new code must have tests
- Coverage must not drop below 90%
- Tests must pass on Python 3.9, 3.10, 3.11, and 3.12
- Tests must pass on Windows, macOS, and Linux

## Code style

- Python 3.9+ compatible syntax
- Type hints on all public functions and methods
- Docstrings on all public functions and classes (NumPy style)
- Line length 100 characters
- No comments that explain WHAT code does — only WHY if non-obvious

## Adding a cleaning strategy

1. Add your logic to `puredata/core/clean.py` in the `AutoClean` class
2. Add a new `FixAction` enum value in `puredata/core/report.py` if needed
3. Write tests in `tests/test_clean.py`

## Adding a DataWatch check

1. Add your check method to `puredata/core/watch.py` in the `DataWatch` class
2. Call it from the `check()` method
3. Write tests in `tests/test_watch.py`

## Adding a plugin

See the [Plugin Development Guide](docs/plugins.md).

## Reporting bugs

Open an issue on GitHub with:
- Python version
- puredata version
- Minimal reproducible example
- Expected vs actual behaviour

## Questions

Open a GitHub Discussion.
