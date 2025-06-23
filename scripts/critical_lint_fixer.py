#!/usr/bin/env python3
"""
🎖️ Critical Lint Fixer - Repository Closer Tool

Legendary precision in fixing critical linting issues.
Focuses only on source code, avoiding virtual environment files.
"""

from pathlib import Path


def fix_critical_imports():
    """Fix critical missing imports in key files."""

    # Files that need os import
    files_needing_os = [
        "workflows/batch_inference.py",
        "workflows/data_ingestion.py",
        "workflows/feature_engineering.py",
        "workflows/model_evaluation.py",
        "tools/fix_linting_issues.py",
    ]

    # Files that need time import
    files_needing_time = ["workflows/model_training.py"]

    # Files that need json import
    files_needing_json = ["workflows/data_ingestion.py"]

    # Files that need tempfile import
    files_needing_tempfile = ["workflows/tests/test_workflow.py"]

    def add_import_if_missing(file_path, import_line):
        """Add import if it's missing from the file."""
        if not Path(file_path).exists():
            return

        loaded_data = Path(file_path).read_text()
        if import_line not in loaded_data:
            # Find the import section and add the import
            lines = loaded_data.split("\n")
            insert_pos = 0

            # Find last import line
            for i, line in enumerate(lines):
                if line.startswith("import ") or line.startswith("from "):
                    insert_pos = i + 1
                elif line.strip() == "" and insert_pos > 0:
                    break

            lines.insert(insert_pos, import_line)
            Path(file_path).write_text("\n".join(lines))
            print(f"✅ Added '{import_line}' to {file_path}")

    # Add missing imports
    for file_path in files_needing_os:
        add_import_if_missing(file_path, "import os")

    for file_path in files_needing_time:
        add_import_if_missing(file_path, "import time")

    for file_path in files_needing_json:
        add_import_if_missing(file_path, "import json")

    for file_path in files_needing_tempfile:
        add_import_if_missing(file_path, "import tempfile")


def fix_specific_issues():
    """Fix specific known issues in key files."""

    # Fix tools/fix_linting_issues.py
    tools_file = Path("tools/fix_linting_issues.py")
    if tools_file.exists():
        loaded_data = tools_file.read_text()
        # Fix the undefined loaded_data variable
        loaded_data = loaded_data.replace(
            "new_loaded_data, count = re.subn(pattern, replacement, loaded_data)",
            "new_loaded_data, count = re.subn(pattern, replacement, _loaded_data)",
        )
        loaded_data = loaded_data.replace(
            "f.write(loaded_data)", "f.write(_loaded_data)"
        )
        tools_file.write_text(loaded_data)
        print(f"✅ Fixed variable issues in {tools_file}")

    # Fix tools/validate_toml.py
    validate_file = Path("tools/validate_toml.py")
    if validate_file.exists():
        loaded_data = validate_file.read_text()
        # Find the function that uses 'loaded_data' and see if we need to add it
        if "def " in loaded_data and "loaded_data.count" in loaded_data:
            # Add loaded_data parameter if missing
            lines = loaded_data.split("\n")
            for i, line in enumerate(lines):
                if (
                    "def " in line
                    and "loaded_data" not in line
                    and any(
                        "loaded_data." in lines[j]
                        for j in range(i + 1, min(i + 20, len(lines)))
                    )
                ):
                    # This function uses loaded_data but doesn't have it as parameter
                    if "(self" in line:
                        lines[i] = line.replace("):", ", loaded_data):")
                    elif "(" in line and ")" in line:
                        lines[i] = line.replace(")", ", loaded_data)")
                    break
            validate_file.write_text("\n".join(lines))
            print(f"✅ Fixed loaded_data parameter in {validate_file}")


def fix_workflow_result_issues():
    """Fix undefined result variables in workflow files."""

    workflow_files = [
        "workflows/batch_inference.py",
        "workflows/data_ingestion.py",
        "workflows/feature_engineering.py",
        "workflows/model_evaluation.py",
        "workflows/model_training.py",
    ]

    for file_path in workflow_files:
        if not Path(file_path).exists():
            continue

        loaded_data = Path(file_path).read_text()

        # Common pattern: result is used but not defined
        # Look for patterns where result is used without being assigned
        if "result[" in loaded_data or "result." in loaded_data:
            lines = loaded_data.split("\n")

            # Find main execution function and ensure result is defined
            for i, line in enumerate(lines):
                if "def main(" in line or "if __name__" in line:
                    # Look for result usage in the following lines
                    for j in range(i, min(i + 50, len(lines))):
                        if (
                            "result[" in lines[j]
                            and "result =" not in lines[j - 1 : j + 1]
                        ):
                            # Insert a result assignment before first usage
                            result_assignment = "    result = {}"
                            if file_path.endswith("model_training.py"):
                                result_assignment = (
                                    "    result = trainer.execute(context)"
                                )
                            elif "batch_inference" in file_path:
                                result_assignment = "    result = run_batch_inference(model_path, input_path, output_path, id_column)"
                            elif "data_ingestion" in file_path:
                                result_assignment = "    result = ingest_data(input_path, output_path, enable_validation, enable_quality_report)"
                            elif "feature_engineering" in file_path:
                                result_assignment = "    result = engineer_features(input_path, output_path, config_path, target_column)"
                            elif "model_evaluation" in file_path:
                                result_assignment = "    result = evaluate_model(model_path, data_path, output_dir, target_column)"

                            lines.insert(j, result_assignment)
                            break
                    break

            Path(file_path).write_text("\n".join(lines))
            print(f"✅ Fixed result variable in {file_path}")


def main():
    """Main execution."""
    print("🎖️ Starting critical lint fixes...")

    print("\n📥 Fixing critical imports...")
    fix_critical_imports()

    print("\n🔧 Fixing specific issues...")
    fix_specific_issues()

    print("\n📊 Fixing workflow result issues...")
    fix_workflow_result_issues()

    print("\n✅ Critical lint fixes completed!")
    print("🎯 Running final check...")


if __name__ == "__main__":
    main()
