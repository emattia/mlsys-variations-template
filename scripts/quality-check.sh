#!/bin/bash

# Quality Check Script - Run all CI quality checks locally
# This helps catch issues before pushing to CI

set -e

echo "🔍 Running Quality Checks..."
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]] && [[ ! -d ".venv" ]]; then
    print_warning "No virtual environment detected. Consider activating one."
fi

echo
echo "1. Code Formatting (ruff format)"
echo "--------------------------------"
if ruff format --check src/ tests/; then
    print_status "Code formatting passed"
else
    print_error "Code formatting failed. Run: ruff format src/ tests/"
    exit 1
fi

echo
echo "2. Linting (ruff check)"
echo "------------------------"
if ruff check src/ tests/; then
    print_status "Linting passed"
else
    print_error "Linting failed. Check the output above."
    exit 1
fi

echo
echo "3. Security Checks (bandit)"
echo "---------------------------"
if command -v bandit &> /dev/null; then
    if bandit -r src/ --severity-level medium --quiet; then
        print_status "Security checks passed"
    else
        print_error "Security issues found. Check the output above."
        exit 1
    fi
else
    print_warning "Bandit not installed. Install with: uv pip install 'bandit[toml]'"
fi

echo
echo "4. Code Complexity (xenon)"
echo "--------------------------"
if command -v xenon &> /dev/null; then
    if xenon src/ --max-absolute B --max-modules B --quiet; then
        print_status "Code complexity checks passed"
    else
        print_warning "Code complexity issues found. Consider refactoring complex functions."
        # Don't exit on complexity issues, just warn
    fi
else
    print_warning "Xenon not installed. Install with: uv pip install xenon"
fi

echo
echo "5. Type Checking (mypy)"
echo "-----------------------"
if command -v mypy &> /dev/null; then
    if mypy src/ --ignore-missing-imports --no-error-summary; then
        print_status "Type checking passed"
    else
        print_warning "Type checking issues found. Consider fixing type hints."
        # Don't exit on type issues, just warn
    fi
else
    print_warning "Mypy not installed. Install with: uv pip install mypy"
fi

echo
echo "6. Testing"
echo "----------"
if command -v pytest &> /dev/null; then
    if pytest tests/ -v --tb=short -x; then
        print_status "Tests passed"
    else
        print_error "Tests failed. Check the output above."
        exit 1
    fi
else
    print_warning "Pytest not installed. Install with: uv pip install pytest"
fi

echo
echo "🎉 Quality checks completed!"
echo
echo "To run individual checks:"
echo "  Format:     ruff format src/ tests/"
echo "  Lint:       ruff check src/ tests/"
echo "  Security:   bandit -r src/ --severity-level medium"
echo "  Complexity: xenon src/ --max-absolute B --max-modules B"
echo "  Types:      mypy src/"
echo "  Tests:      pytest tests/"
echo
echo "To install missing tools:"
echo "  uv pip install 'bandit[toml]' xenon mypy pytest"
