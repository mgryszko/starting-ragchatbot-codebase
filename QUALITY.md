# Code Quality Tools

This project uses several code quality tools to maintain consistent, high-quality code.

## Quick Start

### Auto-format Code
```bash
./format.sh
```

This will:
- Sort imports with `isort`
- Format code with `black`

### Run Quality Checks
```bash
./quality.sh
```

This will run:
1. Import sorting check (isort)
2. Code formatting check (black)
3. Linting (flake8)
4. Type checking (mypy)
5. Tests (pytest)

### Run Strict Quality Checks
```bash
./quality-strict.sh
```

Same as `quality.sh` but with stricter linting rules for code cleanup tasks.

## Tools Configured

### Black (Code Formatter)
- Line length: 88 characters
- Target: Python 3.13
- Automatic code formatting

**Configuration**: `pyproject.toml` → `[tool.black]`

### isort (Import Sorter)
- Profile: black (compatible with black)
- Automatic import organization

**Configuration**: `pyproject.toml` → `[tool.isort]`

### Flake8 (Linter)
- Max line length: 88 (compatible with black)
- Ignored errors:
  - E203, W503: Conflicts with black
  - E501: Line too long (black handles this)
  - E402: Module level import not at top
  - F401, F841: Unused imports/variables

**Configuration**: `.flake8`

### MyPy (Type Checker)
- Python version: 3.13
- Lenient mode for gradual typing adoption
- Disabled checks: var-annotated, arg-type, call-overload, no-any-return, override, return-value

**Configuration**: `pyproject.toml` → `[tool.mypy]`

## Development Workflow

### Before Committing
1. Format your code:
   ```bash
   ./format.sh
   ```

2. Run quality checks:
   ```bash
   ./quality.sh
   ```

### Manual Tool Usage

Format code:
```bash
uv run black backend/ main.py
uv run isort backend/ main.py
```

Check without formatting:
```bash
uv run black --check backend/ main.py
uv run isort --check-only backend/ main.py
```

Run linter:
```bash
uv run flake8 backend/ main.py
```

Run type checker:
```bash
uv run mypy backend/ main.py
```

Run tests:
```bash
uv run pytest backend/tests/ -v
```

## Installing Dev Dependencies

If you need to reinstall the development dependencies:

```bash
uv sync --extra dev
```

## Future Improvements

The current configuration is lenient to allow for gradual adoption. Consider:

1. **Stricter MyPy**: Enable more type checks as type hints are added
2. **Remove ignored flake8 rules**: Clean up unused imports and variables
3. **Add pre-commit hooks**: Automatically run checks before commits
4. **CI/CD integration**: Run quality checks in CI pipeline

## Files

- `format.sh` - Auto-format code
- `quality.sh` - Run all quality checks
- `quality-strict.sh` - Run strict quality checks
- `.flake8` - Flake8 configuration
- `pyproject.toml` - Black, isort, and mypy configuration
