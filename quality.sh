#!/bin/bash

# Run all code quality checks
set -e  # Exit on first error

echo "🔍 Running code quality checks..."
echo ""

echo "1️⃣ Checking import sorting with isort..."
uv run isort --check-only backend/ main.py
echo "✅ Import sorting check passed!"
echo ""

echo "2️⃣ Checking code formatting with black..."
uv run black --check backend/ main.py
echo "✅ Code formatting check passed!"
echo ""

echo "3️⃣ Running linter with flake8..."
uv run flake8 backend/ main.py
echo "✅ Linting check passed!"
echo ""

echo "4️⃣ Running type checker with mypy..."
uv run mypy backend/ main.py
echo "✅ Type checking passed!"
echo ""

echo "5️⃣ Running tests with pytest..."
uv run pytest backend/tests/ -v
echo "✅ Tests passed!"
echo ""

echo "🎉 All quality checks passed!"
