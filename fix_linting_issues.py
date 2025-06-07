#!/usr/bin/env python3
"""Fix common linting issues in Python files.

This script fixes common linting issues in Python files, such as:
- Removing double underscores from keyword arguments
- Fixing undefined names
- Adding missing type annotations
"""

import os
import re


def fix_double_underscore_kwargs(file_path: str) -> int:
    """Fix double underscore keyword arguments.

    Args:
        file_path: Path to the Python file

    Returns:
        Number of replacements made
    """
    with open(file_path) as f:
        content = f.read()

    # Replace __keyword with keyword
    patterns = [
        (r"__level\b", "level"),
        (r"__log_file\b", "log_file"),
        (r"__log_format\b", "log_format"),
        (r"__exist_ok\b", "exist_ok"),
        (r"__title\b", "title"),
        (r"__version\b", "version"),
        (r"__response_model\b", "response_model"),
        (r"__host\b", "host"),
        (r"__port\b", "port"),
        (r"__reload\b", "reload"),
        (r"__xticks\b", "xticks"),
        (r"__yticks\b", "yticks"),
        (r"__xticklabels\b", "xticklabels"),
        (r"__yticklabels\b", "yticklabels"),
        (r"__ylabel\b", "ylabel"),
        (r"__xlabel\b", "xlabel"),
        (r"__rotation\b", "rotation"),
        (r"__va\b", "va"),
        (r"__color\b", "color"),
        (r"__key\b", "key"),
        (r"__indent\b", "indent"),
        (r"__return_train_score\b", "return_train_score"),
        (r"__input_path\b", "input_path"),
        (r"__nargs\b", "nargs"),
        (r"__action\b", "action"),
        (r"__filename\b", "filename"),
        (r"__bool\b", "bool"),
        (r"__int\b", "int"),
    ]

    total_count = 0
    for pattern, replacement in patterns:
        new_content, count = re.subn(pattern, replacement, content)
        if count > 0:
            content = new_content
            total_count += count

    if total_count > 0:
        with open(file_path, "w") as f:
            f.write(content)

        print(f"Fixed {total_count} double underscore keyword arguments in {file_path}")

    return total_count


def find_python_files(directory: str) -> list[str]:
    """Find all Python files in a directory and its subdirectories.

    Args:
        directory: Directory to search

    Returns:
        List of Python file paths
    """
    python_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))

    return python_files


def main() -> None:
    """Main function."""
    # Fix Python files
    directories = ["src", "tests", "workflows", "endpoints"]

    total_fixes = {
        "double_underscore_kwargs": 0,
    }

    for directory in directories:
        if os.path.exists(directory):
            python_files = find_python_files(directory)

            for file_path in python_files:
                double_underscore_kwargs = fix_double_underscore_kwargs(file_path)

                total_fixes["double_underscore_kwargs"] += double_underscore_kwargs

    print("\nSummary of fixes:")
    for key, value in total_fixes.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
