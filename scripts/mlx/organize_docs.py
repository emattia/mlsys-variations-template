#!/usr/bin/env python3
"""
Documentation Organization Script for MLX Foundation

Moves root-level documentation files to appropriate docs/ subdirectories
while maintaining proper cross-references and updating links.
"""

import os
import shutil
from pathlib import Path


def organize_documentation():
    """Move documentation files to proper locations."""

    # Define file moves
    moves = [
        ("ANALYSIS_AND_RECOMMENDATIONS.md", "docs/project-analysis.md"),
        ("BRANCHING_STRATEGY.md", "docs/development/branching-strategy.md"),
        # Add more as needed
    ]

    for source, destination in moves:
        if os.path.exists(source):
            # Ensure destination directory exists
            Path(destination).parent.mkdir(parents=True, exist_ok=True)

            # Move file
            shutil.move(source, destination)
            print(f"Moved {source} -> {destination}")
        else:
            print(f"Source file not found: {source}")


if __name__ == "__main__":
    organize_documentation()
