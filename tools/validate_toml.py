#!/usr/bin/env python3
"""Validate TOML files for syntax errors.

This script validates TOML files for syntax errors and reports any issues found.
"""

import argparse
import sys
from pathlib import Path


def validate_toml_file(file_path: str) -> tuple[bool, str | None]:
    """Validate a TOML file for syntax errors.

    Args:
        file_path: Path to the TOML file

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Try to use tomli (Python 3.11+ includes tomllib in standard library)
    try:
        try:
            import tomli

            with open(file_path, "rb") as f:
                tomli.load(f)

            return True, None
        except ModuleNotFoundError:
            # Try tomllib (Python 3.11+)
            try:
                import tomllib

                with open(file_path, "rb") as f:
                    tomllib.load(f)

                return True, None
            except ModuleNotFoundError:
                # Fall back to toml
                try:
                    import toml

                    with open(file_path) as f:
                        toml.load(f)

                    return True, None
                except ModuleNotFoundError:
                    # Last resort: use a simple syntax check
                    return simple_toml_syntax_check(file_path)
    except Exception as err:  # noqa: BLE001
        # This will catch TOMLDecodeError from any of the parsers
        return False, str(err)


def simple_toml_syntax_check(file_path: str, loaded_data) -> tuple[bool, str | None]:
    """Perform a simple syntax check on a TOML file.

    This is a fallback when no TOML parser is available.
    It checks for basic syntax errors like unbalanced brackets and quotes.

    Args:
        file_path: Path to the TOML file

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        with open(file_path) as f:
            loaded_data = f.read()

        # Check for balanced brackets
        if loaded_data.count("[") != loaded_data.count("]"):
            return False, "Unbalanced square brackets"

        # Check for balanced quotes
        if loaded_data.count('"') % 2 != 0:
            return False, "Unbalanced double quotes"

        if loaded_data.count("'") % 2 != 0:
            return False, "Unbalanced single quotes"

        # Check for balanced braces
        if loaded_data.count("{") != loaded_data.count("}"):
            return False, "Unbalanced curly braces"

        # Check for common TOML syntax errors
        lines = loaded_data.split("\n")
        for i, line in enumerate(lines):
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            # Check for invalid table headers
            if line.startswith("[") and line.endswith("]"):
                table_name = line[1:-1]
                if not table_name or "[]" in table_name:
                    return False, f"Invalid table name on line {i + 1}: {line}"

            # Check for invalid key-value pairs
            if "=" in line and not line.startswith("["):
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                if not key:
                    return False, f"Empty key on line {i + 1}: {line}"

                # Check for unclosed inline tables
                if (
                    value.startswith("{")
                    and not value.endswith("}")
                    and "}" not in value
                ):
                    return False, f"Unclosed inline table on line {i + 1}: {line}"

        return True, None
    except Exception as err:  # noqa: BLE001
        return False, f"Error during syntax check: {str(err)}"


def find_toml_files(directory: str) -> list[str]:
    """Find all TOML files in a directory and its subdirectories.

    Args:
        directory: Directory to search

    Returns:
        List of TOML file paths
    """
    toml_files = []

    for path in Path(directory).rglob("*.toml"):
        toml_files.append(str(path))

    return toml_files


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Validate TOML files for syntax errors"
    )
    parser.add_argument("files", nargs="*", help="TOML files to validate")
    parser.add_argument("--directory", "-d", help="Directory to search for TOML files")
    parser.add_argument(
        "--recursive", "-r", action="store_true", help="Search recursively in directory"
    )

    args = parser.parse_args()

    # Collect files to validate
    files_to_validate = []

    if args.files:
        files_to_validate.extend(args.files)

    if args.directory:
        if args.recursive:
            files_to_validate.extend(find_toml_files(args.directory))
        else:
            for path in Path(args.directory).glob("*.toml"):
                files_to_validate.append(str(path))

    if not files_to_validate:
        # Default to pyproject.toml if no files specified
        if Path("pyproject.toml").exists():
            files_to_validate.append("pyproject.toml")
        else:
            print("No TOML files specified and no pyproject.toml found.")
            sys.exit(1)

    # Validate files
    all_valid = True

    for file_path in files_to_validate:
        print(f"Validating {file_path}...")
        is_valid, error_message = validate_toml_file(file_path)

        if is_valid:
            print(f"✅ {file_path} is valid")
        else:
            all_valid = False
            print(f"❌ {file_path} has syntax errors:")
            print(f"   {error_message}")

    if not all_valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
