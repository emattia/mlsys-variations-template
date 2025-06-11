#!/usr/bin/env python3
"""Verify that the ASCII tree in docs/user-guide/project-structure.md matches the real repository tree.

This script is intended to run in CI to prevent the documentation from drifting
away from reality. It walks the repository (excluding some common ignore
patterns) and compares the top-level directory layout against the tree embedded
in the markdown file.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_FILE = REPO_ROOT / "docs/user-guide/project-structure.md"

# Directories deliberately ignored when comparing against the doc.
_IGNORE = {
    ".git",
    ".venv",
    ".bootstrap_venv",
    "__pycache__",
    ".ruff_cache",
    ".mypy_cache",
    ".pytest_cache",
    "htmlcov",
    "logs",
    "documentation",
    "tools",
    "archive",
    "scripts",
    ".cursor",
}


def _get_repo_dirs() -> list[str]:
    """Return sorted list of first-level directories in the repo root."""
    dirs = [p.name for p in REPO_ROOT.iterdir() if p.is_dir() and p.name not in _IGNORE]
    return sorted(dirs)


def _extract_doc_dirs() -> list[str]:
    """Parse the markdown file and extract the directory names from the ASCII tree."""
    content = DOC_FILE.read_text()

    # Extract the fenced code block containing the tree (```).
    tree_match = re.search(r"```[\s\S]*?```", content)
    if not tree_match:
        print("❌  Could not find an ASCII tree fenced code block in the doc.")
        sys.exit(1)

    tree_block = tree_match.group(0)

    # Grab directory names from first level only (start of line).
    dirs = re.findall(r"^[├└]──\s+([^/]+)/", tree_block, flags=re.MULTILINE)
    return sorted(dirs)


def main() -> None:  # noqa: D401
    """CLI entry-point."""
    repo_dirs = _get_repo_dirs()
    doc_dirs = _extract_doc_dirs()

    if repo_dirs != doc_dirs:
        print(
            "❌  Project structure in docs/user-guide/project-structure.md is out of date."
        )
        print("Repo directories:", repo_dirs)
        print("Doc  directories:", doc_dirs)
        sys.exit(1)

    print("✅  Project structure documentation is up-to-date.")


if __name__ == "__main__":
    main()
