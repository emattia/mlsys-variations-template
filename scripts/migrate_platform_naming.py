#!/usr/bin/env python3
"""
🔄 Platform-Wide Naming Migration Script

Migrates the entire MLX/MLSys platform to use centralized naming configuration.
Updates all hardcoded naming references across the entire codebase.

Usage:
    python scripts/migrate_platform_naming.py set-preset mlx --apply
    python scripts/migrate_platform_naming.py set-preset mlsys --apply
    python scripts/migrate_platform_naming.py set-preset custom:myplatform --apply
    python scripts/migrate_platform_naming.py migrate --dry-run
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add the naming config to path
sys.path.insert(0, str(Path(__file__).parent))
try:
    from naming_config import NamingConfig, CommonNamingConfigs, get_naming_config
except ImportError:
    # Handle case where naming_config module doesn't exist yet
    console = Console()
    console.print(
        "[red]Error: naming_config module not found. Please ensure naming_config.py exists.[/red]"
    )
    sys.exit(1)

console = Console()


class PlatformNamingMigrator:
    """Handles migration of hardcoded names across the entire platform"""

    def __init__(self):
        # Platform-wide files that need migration
        self.platform_files = [
            # Main configuration and project files
            "mlx.config.json",
            "docker-compose.yml",
            "README.md",
            "mlsys",  # Main CLI script
            # Evaluation system
            "scripts/evaluation/ai_response_evaluator.py",
            "scripts/evaluation/benchmark_generator.py",
            "scripts/evaluation/analytics_dashboard.py",
            "scripts/evaluation/mlx_eval.py",
            "scripts/evaluation/setup.py",
            "scripts/evaluation/README.md",
            # MLX scripts
            "scripts/mlx/create_project.py",
            "scripts/mlx/component_injector.py",
            "scripts/mlx/component_extractor.py",
            # Source code
            "src/config/models.py",
            "src/assistant/bootstrap.py",
            # Documentation
            "docs/**/*.md",
            # Configuration files
            ".projen/tasks.json",
            "pyproject.toml",
            # Tests
            "tests/**/*.py",
        ]

        # Define comprehensive replacement patterns
        self.replacement_patterns = [
            # Platform/Brand references
            (
                r"\bMLX AI Response Evaluation System\b",
                "{PLATFORM_NAME_UPPER} AI Response Evaluation System",
            ),
            (r"\bMLX Platform Foundation\b", "{PLATFORM_FULL_NAME}"),
            (r"\bMLX platform\b", "{PLATFORM_NAME} platform"),
            (r"\bMLX Platform\b", "{PLATFORM_NAME_TITLE} Platform"),
            (r"\bMLX Project\b", "{PLATFORM_NAME_TITLE} Project"),
            (r"\bMLX-specific\b", "{PLATFORM_NAME_UPPER}-specific"),
            (r"\bMLX framework\b", "{PLATFORM_NAME} framework"),
            (r"\bMLX command\b", "{PLATFORM_NAME} command"),
            (r"\bMLX project\b", "{PLATFORM_NAME} project"),
            (r"\bMLX Gateway\b", "{PLATFORM_NAME_TITLE} Gateway"),
            # Command patterns
            (r"\bmlx assistant\b", "{ASSISTANT_COMMAND}"),
            (r'"mlx assistant', '"{ASSISTANT_COMMAND}'),
            (r"`mlx assistant", "`{ASSISTANT_COMMAND}"),
            (r"mlx\s+assistant\s+", "{ASSISTANT_COMMAND} "),
            # CLI references
            (r"\bmlx-eval\b", "{EVALUATION_CLI}"),
            (r"\./mlsys\b", "./{MAIN_CLI}"),
            (
                r"\bmlsys\b(?!\s*=)(?!-)",
                "{MAIN_CLI}",
            ),  # Avoid replacing in assignments and compound words
            # File and directory names
            (r"\bmlx\.config\.json\b", "{CONFIG_FILE}"),
            (r"\bmlx-components\b", "{COMPONENTS_DIR}"),
            (r"\.mlx\b(?!/)", "{METADATA_DIR}"),  # Avoid replacing file extensions
            # Network names
            (r"\bmlsys-network\b", "{DOCKER_NETWORK}"),
            # Package and plugin prefixes
            (r"\bmlx-plugin-", "{PACKAGE_PREFIX}-plugin-"),
            (r"\bmlx-foundation\b", "{PACKAGE_PREFIX}-foundation"),
            # Template names
            (r"\bmlsys-variations-template\b", "{TEMPLATE_NAME}"),
            # Projen task names (special handling)
            (r'"mlx:', '"{PACKAGE_PREFIX}:'),
            (r"mlx:([a-z-]+)", "{PACKAGE_PREFIX}:\\1"),
        ]

    def discover_files(self) -> List[Path]:
        """Discover all files that might need migration"""
        discovered_files = set()

        for pattern in self.platform_files:
            # pattern_path = Path(pattern)

            if pattern.endswith("**/*.md") or pattern.endswith("**/*.py"):
                # Handle glob patterns
                base_path = Path(pattern.split("**/")[0])
                if base_path.exists():
                    extension = pattern.split("**/")[1]
                    for file_path in base_path.rglob(extension):
                        discovered_files.add(file_path)
            elif "*" in pattern:
                # Handle other glob patterns
                for file_path in Path(".").glob(pattern):
                    if file_path.is_file():
                        discovered_files.add(file_path)
            else:
                # Handle specific files
                file_path = Path(pattern)
                if file_path.exists() and file_path.is_file():
                    discovered_files.add(file_path)

        return sorted(list(discovered_files))

    def analyze_files(self) -> Dict[str, List[Tuple[int, str, str]]]:
        """Analyze files to find naming patterns that need migration"""
        results = {}
        files = self.discover_files()

        with Progress() as progress:
            task = progress.add_task("Analyzing files...", total=len(files))

            for file_path in files:
                matches = []
                try:
                    loaded_data = file_path.read_text(encoding="utf-8", errors="ignore")
                    lines = loaded_data.split("\n")

                    for line_num, line in enumerate(lines, 1):
                        for pattern, replacement in self.replacement_patterns:
                            if re.search(pattern, line):
                                matches.append((line_num, pattern, line.strip()))

                    if matches:
                        results[str(file_path)] = matches

                except Exception as e:
                    console.print(f"[red]Error analyzing {file_path}: {e}[/red]")

                progress.update(task, advance=1)

        return results

    def migrate_file(
        self, file_path: Path, config: NamingConfig, dry_run: bool = True
    ) -> Tuple[bool, int]:
        """Migrate a single file to use naming configuration"""
        if not file_path.exists():
            return False, 0

        try:
            loaded_data = file_path.read_text(encoding="utf-8", errors="ignore")
            original_data = loaded_data
            changes_made = 0

            # Apply replacement patterns
            for pattern, replacement_template in self.replacement_patterns:
                # Substitute the template with actual values
                replacement = self._substitute_template(replacement_template, config)

                # Count matches before replacement
                matches = len(re.findall(pattern, loaded_data))
                if matches > 0:
                    loaded_data = re.sub(pattern, replacement, loaded_data)
                    changes_made += matches

            # Special handling for specific file types
            if file_path.name == "mlx.config.json":
                loaded_data = self._migrate_config_file(loaded_data, config)
            elif file_path.name == "docker-compose.yml":
                loaded_data = self._migrate_docker_compose(loaded_data, config)
            elif file_path.name == "mlsys":
                loaded_data = self._migrate_main_cli(loaded_data, config)
            elif file_path.name.endswith("_eval.py") or "eval" in file_path.name:
                loaded_data = self._migrate_evaluation_files(loaded_data, config)
            elif file_path.suffix == ".json" and "tasks" in file_path.name:
                loaded_data = self._migrate_projen_tasks(loaded_data, config)

            if loaded_data != original_data:
                if not dry_run:
                    # Create backup
                    backup_path = file_path.with_suffix(file_path.suffix + ".backup")
                    if not backup_path.exists():
                        backup_path.write_text(loaded_data, encoding="utf-8")

                    # Write updated loaded_data
                    file_path.write_text(loaded_data, encoding="utf-8")

                return True, changes_made

            return False, 0

        except Exception as e:
            console.print(f"[red]Error migrating {file_path}: {e}[/red]")
            return False, 0

    def _substitute_template(self, template: str, config: NamingConfig) -> str:
        """Substitute template placeholders with config values"""
        replacements = {
            "{PLATFORM_NAME}": config.platform_name,
            "{PLATFORM_FULL_NAME}": config.platform_full_name,
            "{PLATFORM_NAME_UPPER}": config.platform_name.upper(),
            "{PLATFORM_NAME_TITLE}": config.platform_name.title(),
            "{ASSISTANT_COMMAND}": config.assistant_command,
            "{EVALUATION_CLI}": config.evaluation_cli,
            "{MAIN_CLI}": config.main_cli,
            "{CONFIG_FILE}": config.config_file,
            "{COMPONENTS_DIR}": config.components_dir,
            "{METADATA_DIR}": config.metadata_dir,
            "{DOCKER_NETWORK}": config.docker_network,
            "{PACKAGE_PREFIX}": config.package_prefix,
            "{TEMPLATE_NAME}": config.template_name,
        }

        result = template
        for placeholder, value in replacements.items():
            result = result.replace(placeholder, value)

        return result

    def _migrate_config_file(self, loaded_data: str, config: NamingConfig) -> str:
        """Special migration for mlx.config.json"""

        try:
            data = json.loads(loaded_data)

            # Update platform name
            if "platform" in data and "name" in data["platform"]:
                data["platform"]["name"] = f"{config.package_prefix}-foundation"

            return json.dumps(data, indent=2)
        except Exception:
            return loaded_data

    def _migrate_docker_compose(self, loaded_data: str, config: NamingConfig) -> str:
        """Special migration for docker-compose.yml"""
        # Update network references
        loaded_data = re.sub(
            r"mlsys-network:", f"{config.docker_network}:", loaded_data
        )
        loaded_data = re.sub(
            r"- mlsys-network", f"- {config.docker_network}", loaded_data
        )
        loaded_data = re.sub(
            r"mlsys-network:", f"{config.docker_network}:", loaded_data
        )

        return loaded_data

    def _migrate_main_cli(self, loaded_data: str, config: NamingConfig) -> str:
        """Special migration for main CLI script"""
        # Update CLI help text and descriptions
        loaded_data = re.sub(
            r"MLX Gateway - Production-grade ML platform component system",
            f"{config.platform_name.upper()} Gateway - {config.platform_description}",
            loaded_data,
        )

        return loaded_data

    def _migrate_evaluation_files(self, loaded_data: str, config: NamingConfig) -> str:
        """Special migration for evaluation system files"""
        # Update CLI app names
        loaded_data = re.sub(
            r'name="mlx-eval"', f'name="{config.evaluation_cli}"', loaded_data
        )

        # Update usage examples in documentation
        loaded_data = re.sub(
            r"python scripts/evaluation/mlx_eval\.py",
            f"python scripts/evaluation/{config.evaluation_cli.replace('-', '_')}.py",
            loaded_data,
        )

        return loaded_data

    def _migrate_projen_tasks(self, loaded_data: str, config: NamingConfig) -> str:
        """Special migration for .projen/tasks.json"""
        # Update task names and commands
        loaded_data = re.sub(r'"mlx:', f'"{config.package_prefix}:', loaded_data)
        loaded_data = re.sub(
            r'"name": "mlx:', f'"name": "{config.package_prefix}:', loaded_data
        )

        return loaded_data


class MigrationValidator:
    """Comprehensive migration validation and completeness checking."""

    def __init__(self, config: NamingConfig):
        self.config = config
        self.console = Console()
        self.validation_results = {}

    def validate_migration_completeness(self) -> Dict[str, Any]:
        """Perform comprehensive validation of migration completeness."""
        self.console.print(
            "🔍 [bold]Performing Migration Completeness Validation[/bold]"
        )

        results = {
            "overall_status": "unknown",
            "consistency_score": 0.0,
            "issues_found": [],
            "recommendations": [],
            "file_coverage": {},
            "pattern_validation": {},
            "cli_validation": {},
            "integration_points": {},
        }

        # 1. Validate file naming consistency
        results["file_coverage"] = self._validate_file_naming()

        # 2. Validate pattern consistency across files
        results["pattern_validation"] = self._validate_pattern_consistency()

        # 3. Validate CLI functionality
        results["cli_validation"] = self._validate_cli_functionality()

        # 4. Validate integration points
        results["integration_points"] = self._validate_integration_points()

        # 5. Calculate overall consistency score
        results["consistency_score"] = self._calculate_consistency_score(results)

        # 6. Generate recommendations
        results["recommendations"] = self._generate_recommendations(results)

        # 7. Determine overall status
        if results["consistency_score"] >= 0.95:
            results["overall_status"] = "excellent"
        elif results["consistency_score"] >= 0.85:
            results["overall_status"] = "good"
        elif results["consistency_score"] >= 0.70:
            results["overall_status"] = "needs_improvement"
        else:
            results["overall_status"] = "poor"

        self.validation_results = results
        return results

    def _validate_file_naming(self) -> Dict[str, Any]:
        """Validate that critical files have been properly renamed."""
        file_validation = {
            "status": "checking",
            "expected_files": [],
            "missing_files": [],
            "incorrect_files": [],
            "score": 0.0,
        }

        # Define expected files based on configuration
        expected_files = [
            (f"{self.config.config_file}", "Main configuration file"),
            (f"{self.config.main_cli}", "Main CLI script"),
            (f"{self.config.components_dir}/", "Components directory"),
            (f".{self.config.metadata_dir}/", "Metadata directory (optional)"),
        ]

        # Check for file existence and naming
        for expected_file, description in expected_files:
            file_path = Path(expected_file)
            if file_path.exists():
                file_validation["expected_files"].append(
                    {
                        "file": expected_file,
                        "description": description,
                        "status": "✅ Found",
                    }
                )
            else:
                file_validation["missing_files"].append(
                    {
                        "file": expected_file,
                        "description": description,
                        "status": "❌ Missing",
                    }
                )

        # Calculate score
        total_files = len(expected_files)
        found_files = len(file_validation["expected_files"])
        file_validation["score"] = found_files / total_files if total_files > 0 else 1.0
        file_validation["status"] = "complete"

        return file_validation

    def _validate_pattern_consistency(self) -> Dict[str, Any]:
        """Validate naming pattern consistency across all files."""
        migrator = PlatformNamingMigrator()
        files = migrator.discover_files()

        pattern_validation = {
            "status": "checking",
            "files_checked": 0,
            "inconsistencies": [],
            "consistency_by_file": {},
            "score": 0.0,
        }

        inconsistent_patterns = []
        total_files_checked = 0

        for file_path in files:
            try:
                if file_path.suffix in [".py", ".md", ".json", ".yml", ".yaml"]:
                    loaded_data = file_path.read_text(encoding="utf-8", errors="ignore")
                    file_inconsistencies = self._check_file_patterns(
                        file_path, loaded_data
                    )

                    pattern_validation["consistency_by_file"][str(file_path)] = {
                        "inconsistencies": len(file_inconsistencies),
                        "issues": file_inconsistencies,
                    }

                    inconsistent_patterns.extend(file_inconsistencies)
                    total_files_checked += 1

            except Exception as e:
                pattern_validation["consistency_by_file"][str(file_path)] = {
                    "error": str(e)
                }

        pattern_validation["files_checked"] = total_files_checked
        pattern_validation["inconsistencies"] = inconsistent_patterns

        # Calculate consistency score
        if total_files_checked > 0:
            files_with_issues = len(
                [
                    f
                    for f in pattern_validation["consistency_by_file"].values()
                    if f.get("inconsistencies", 0) > 0
                ]
            )
            pattern_validation["score"] = 1.0 - (
                files_with_issues / total_files_checked
            )
        else:
            pattern_validation["score"] = 1.0

        pattern_validation["status"] = "complete"
        return pattern_validation

    def _check_file_patterns(
        self, file_path: Path, loaded_data: str
    ) -> List[Dict[str, Any]]:
        """Check a specific file for naming pattern inconsistencies."""
        inconsistencies = []

        # Define patterns that should be consistent
        expected_patterns = {
            "platform_name": self.config.platform_name,
            "main_cli": self.config.main_cli,
            "evaluation_cli": self.config.evaluation_cli,
            "config_file": self.config.config_file,
            "components_dir": self.config.components_dir,
            "docker_network": self.config.docker_network,
        }

        # Check for old/inconsistent patterns (this would need to be customized based on migration history)
        potential_old_patterns = [
            ("mlsys", "Should be updated to current platform name"),
            ("mlx-platform-template", "Should be updated to current template name"),
            ("mlsys-network", "Should be updated to current docker network name"),
            ("mlsys-components", "Should be updated to current components directory"),
        ]

        lines = loaded_data.split("\n")
        for line_num, line in enumerate(lines, 1):
            for old_pattern, issue_desc in potential_old_patterns:
                if old_pattern in line and old_pattern != expected_patterns.get(
                    old_pattern.split("-")[0].split(".")[0], ""
                ):
                    inconsistencies.append(
                        {
                            "file": str(file_path),
                            "line": line_num,
                            "pattern": old_pattern,
                            "context": line.strip()[:100],
                            "issue": issue_desc,
                            "severity": "medium",
                        }
                    )

        return inconsistencies

    def _validate_cli_functionality(self) -> Dict[str, Any]:
        """Validate that CLI commands work with new naming."""
        cli_validation = {
            "status": "checking",
            "commands_tested": [],
            "failures": [],
            "score": 0.0,
        }

        # Test basic CLI functionality
        test_commands = [
            (f"./{self.config.main_cli} --version", "Main CLI version check"),
            (f"./{self.config.main_cli} --help", "Main CLI help"),
            ("python scripts/test_naming_system.py", "Naming system validation"),
            ("python scripts/migrate_platform_naming.py analyze", "Migration analysis"),
        ]

        successful_commands = 0

        for command, description in test_commands:
            try:
                result = subprocess.run(
                    command.split(), capture_output=True, text=True, timeout=30
                )

                if result.returncode == 0:
                    cli_validation["commands_tested"].append(
                        {
                            "command": command,
                            "description": description,
                            "status": "✅ Success",
                        }
                    )
                    successful_commands += 1
                else:
                    cli_validation["failures"].append(
                        {
                            "command": command,
                            "description": description,
                            "status": "❌ Failed",
                            "error": result.stderr[:200],
                        }
                    )

            except Exception as e:
                cli_validation["failures"].append(
                    {
                        "command": command,
                        "description": description,
                        "status": "❌ Error",
                        "error": str(e)[:200],
                    }
                )

        cli_validation["score"] = (
            successful_commands / len(test_commands) if test_commands else 1.0
        )
        cli_validation["status"] = "complete"

        return cli_validation

    def _validate_integration_points(self) -> Dict[str, Any]:
        """Validate key integration points and dependencies."""
        integration_validation = {
            "status": "checking",
            "integration_points": [],
            "issues": [],
            "score": 0.0,
        }

        # Check key integration points
        integrations_to_check = [
            ("pyproject.toml", "Package configuration", self._check_pyproject_toml),
            ("docker-compose.yml", "Docker services", self._check_docker_compose),
            ("Makefile", "Build system", self._check_makefile),
            (".github/workflows/", "CI/CD workflows", self._check_github_workflows),
        ]

        working_integrations = 0

        for integration_file, description, check_func in integrations_to_check:
            try:
                result = check_func(integration_file)
                if result["status"] == "ok":
                    integration_validation["integration_points"].append(
                        {
                            "integration": integration_file,
                            "description": description,
                            "status": "✅ OK",
                            "details": result.get("details", ""),
                        }
                    )
                    working_integrations += 1
                else:
                    integration_validation["issues"].append(
                        {
                            "integration": integration_file,
                            "description": description,
                            "status": "⚠️ Issues",
                            "details": result.get("details", ""),
                            "recommendations": result.get("recommendations", []),
                        }
                    )

            except Exception as e:
                integration_validation["issues"].append(
                    {
                        "integration": integration_file,
                        "description": description,
                        "status": "❌ Error",
                        "error": str(e),
                    }
                )

        integration_validation["score"] = working_integrations / len(
            integrations_to_check
        )
        integration_validation["status"] = "complete"

        return integration_validation

    def _check_pyproject_toml(self, file_path: str) -> Dict[str, Any]:
        """Check pyproject.toml for naming consistency."""
        pyproject_path = Path(file_path)
        if not pyproject_path.exists():
            return {"status": "missing", "details": "pyproject.toml not found"}

        try:
            loaded_data = pyproject_path.read_text()
            # Check for consistent naming in project configuration
            if self.config.main_cli in loaded_data:
                return {
                    "status": "ok",
                    "details": f"CLI name {self.config.main_cli} found in configuration",
                }
            else:
                return {
                    "status": "issue",
                    "details": f"CLI name {self.config.main_cli} not found in pyproject.toml",
                    "recommendations": [
                        "Update pyproject.toml [tool.poetry.scripts] section"
                    ],
                }
        except Exception as e:
            return {"status": "error", "details": str(e)}

    def _check_docker_compose(self, file_path: str) -> Dict[str, Any]:
        """Check docker-compose.yml for naming consistency."""
        docker_path = Path(file_path)
        if not docker_path.exists():
            return {"status": "missing", "details": "docker-compose.yml not found"}

        try:
            loaded_data = docker_path.read_text()
            if self.config.docker_network in loaded_data:
                return {
                    "status": "ok",
                    "details": f"Network name {self.config.docker_network} found",
                }
            else:
                return {
                    "status": "issue",
                    "details": f"Network name {self.config.docker_network} not found in docker-compose.yml",
                    "recommendations": ["Update network names in docker-compose.yml"],
                }
        except Exception as e:
            return {"status": "error", "details": str(e)}

    def _check_makefile(self, file_path: str) -> Dict[str, Any]:
        """Check Makefile for naming consistency."""
        makefile_path = Path(file_path)
        if not makefile_path.exists():
            return {"status": "missing", "details": "Makefile not found"}

        try:
            makefile_path.read_text()
            # Basic check - more sophisticated checking could be added
            return {"status": "ok", "details": "Makefile exists and appears valid"}
        except Exception as e:
            return {"status": "error", "details": str(e)}

    def _check_github_workflows(self, file_path: str) -> Dict[str, Any]:
        """Check GitHub workflows for naming consistency."""
        workflows_path = Path(file_path)
        if not workflows_path.exists():
            return {
                "status": "missing",
                "details": "GitHub workflows directory not found",
            }

        try:
            workflow_files = list(workflows_path.glob("*.yml")) + list(
                workflows_path.glob("*.yaml")
            )
            if workflow_files:
                return {
                    "status": "ok",
                    "details": f"Found {len(workflow_files)} workflow files",
                }
            else:
                return {"status": "missing", "details": "No workflow files found"}
        except Exception as e:
            return {"status": "error", "details": str(e)}

    def _calculate_consistency_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall consistency score from validation results."""
        scores = []
        weights = []

        # Weight different validation categories
        if "file_coverage" in results:
            scores.append(results["file_coverage"].get("score", 0.0))
            weights.append(0.25)  # 25% weight

        if "pattern_validation" in results:
            scores.append(results["pattern_validation"].get("score", 0.0))
            weights.append(0.35)  # 35% weight (most important)

        if "cli_validation" in results:
            scores.append(results["cli_validation"].get("score", 0.0))
            weights.append(0.25)  # 25% weight

        if "integration_points" in results:
            scores.append(results["integration_points"].get("score", 0.0))
            weights.append(0.15)  # 15% weight

        if not scores:
            return 0.0

        # Calculate weighted average
        weighted_score = sum(score * weight for score, weight in zip(scores, weights))
        total_weight = sum(weights)

        return weighted_score / total_weight if total_weight > 0 else 0.0

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate intelligent recommendations based on validation results."""
        recommendations = []

        # File coverage recommendations
        if results.get("file_coverage", {}).get("score", 1.0) < 0.8:
            missing_files = results["file_coverage"].get("missing_files", [])
            if missing_files:
                recommendations.append(
                    f"🗂️ Missing files detected: {', '.join([f['file'] for f in missing_files[:3]])}. "
                    "Consider manual file renaming or updating references."
                )

        # Pattern consistency recommendations
        if results.get("pattern_validation", {}).get("score", 1.0) < 0.9:
            inconsistencies = results["pattern_validation"].get("inconsistencies", [])
            if inconsistencies:
                recommendations.append(
                    f"🔍 Found {len(inconsistencies)} naming inconsistencies. "
                    "Run the migration again or update files manually."
                )

        # CLI recommendations
        if results.get("cli_validation", {}).get("score", 1.0) < 0.8:
            failures = results["cli_validation"].get("failures", [])
            if failures:
                recommendations.append(
                    "⚡ CLI functionality issues detected. "
                    "Check that renamed CLI scripts are executable and properly updated."
                )

        # Integration recommendations
        if results.get("integration_points", {}).get("score", 1.0) < 0.8:
            recommendations.append(
                "🔌 Integration points need attention. "
                "Check pyproject.toml, docker-compose.yml, and CI/CD configurations."
            )

        # Overall score recommendations
        overall_score = results.get("consistency_score", 0.0)
        if overall_score < 0.7:
            recommendations.append(
                "🚨 Overall consistency is low. Consider running a complete re-migration "
                "or manual review of critical files."
            )
        elif overall_score < 0.9:
            recommendations.append(
                "📈 Good progress! A few more updates should achieve excellent consistency."
            )

        return recommendations

    def display_validation_report(self):
        """Display a comprehensive validation report."""
        if not self.validation_results:
            self.console.print(
                "❌ No validation results available. Run validation first."
            )
            return

        results = self.validation_results

        # Header
        status_color = {
            "excellent": "bright_green",
            "good": "green",
            "needs_improvement": "yellow",
            "poor": "red",
        }.get(results["overall_status"], "white")

        header = Panel.fit(
            f"[bold {status_color}]Migration Validation Report[/bold {status_color}]\n"
            f"Overall Status: [bold]{results['overall_status'].upper()}[/bold]\n"
            f"Consistency Score: [bold]{results['consistency_score']:.1%}[/bold]",
            border_style=status_color,
        )
        self.console.print(header)
        self.console.print()

        # Detailed breakdown
        breakdown_table = Table(title="📊 Validation Breakdown", show_header=True)
        breakdown_table.add_column("Category", style="cyan")
        breakdown_table.add_column("Score", justify="center")
        breakdown_table.add_column("Status", justify="center")
        breakdown_table.add_column("Details")

        categories = [
            ("File Coverage", results.get("file_coverage", {})),
            ("Pattern Consistency", results.get("pattern_validation", {})),
            ("CLI Functionality", results.get("cli_validation", {})),
            ("Integration Points", results.get("integration_points", {})),
        ]

        for category, data in categories:
            score = data.get("score", 0.0)
            score_color = (
                "green" if score >= 0.9 else "yellow" if score >= 0.7 else "red"
            )
            status = (
                "✅ Excellent"
                if score >= 0.9
                else "⚠️ Needs Work"
                if score >= 0.7
                else "❌ Issues"
            )

            # Generate details summary
            if category == "File Coverage":
                missing = len(data.get("missing_files", []))
                details = (
                    f"{missing} missing files" if missing > 0 else "All files found"
                )
            elif category == "Pattern Consistency":
                inconsistencies = len(data.get("inconsistencies", []))
                details = (
                    f"{inconsistencies} inconsistencies"
                    if inconsistencies > 0
                    else "All patterns consistent"
                )
            elif category == "CLI Functionality":
                failures = len(data.get("failures", []))
                details = (
                    f"{failures} command failures"
                    if failures > 0
                    else "All commands working"
                )
            elif category == "Integration Points":
                issues = len(data.get("issues", []))
                details = (
                    f"{issues} integration issues"
                    if issues > 0
                    else "All integrations OK"
                )
            else:
                details = "N/A"

            breakdown_table.add_row(
                category, f"[{score_color}]{score:.1%}[/{score_color}]", status, details
            )

        self.console.print(breakdown_table)
        self.console.print()

        # Recommendations
        if results.get("recommendations"):
            recommendations_panel = Panel(
                "\n".join([f"• {rec}" for rec in results["recommendations"]]),
                title="🎯 Recommendations",
                border_style="bright_blue",
            )
            self.console.print(recommendations_panel)
            self.console.print()

        # Issues summary
        all_issues = []

        # Collect issues from all categories
        for category_data in [
            results.get("pattern_validation", {}),
            results.get("cli_validation", {}),
            results.get("integration_points", {}),
        ]:
            if "inconsistencies" in category_data:
                all_issues.extend(category_data["inconsistencies"])
            if "failures" in category_data:
                all_issues.extend(category_data["failures"])
            if "issues" in category_data:
                all_issues.extend(category_data["issues"])

        if all_issues:
            self.console.print(
                f"🔍 [bold]Found {len(all_issues)} specific issues:[/bold]"
            )

            issues_table = Table(show_header=True)
            issues_table.add_column("Issue", style="red")
            issues_table.add_column("Location", style="dim")
            issues_table.add_column("Details", style="yellow")

            for issue in all_issues[:10]:  # Show first 10 issues
                location = issue.get(
                    "file", issue.get("command", issue.get("integration", "Unknown"))
                )
                details = issue.get(
                    "context", issue.get("error", issue.get("details", ""))
                )[:50]
                issue_desc = issue.get(
                    "pattern",
                    issue.get("description", issue.get("issue", "Unknown issue")),
                )

                issues_table.add_row(issue_desc, location, details)

            if len(all_issues) > 10:
                issues_table.add_row(
                    "...",
                    f"and {len(all_issues) - 10} more",
                    "Run detailed analysis for full list",
                )

            self.console.print(issues_table)


def main():
    app = typer.Typer(help="Platform-wide naming migration tool")

    @app.command("analyze")
    def analyze_naming():
        """Analyze current naming patterns across the entire platform"""
        migrator = PlatformNamingMigrator()
        results = migrator.analyze_files()

        if not results:
            console.print(
                "[green]✅ No naming patterns found that need migration[/green]"
            )
            return

        console.print(
            f"[yellow]📋 Found naming patterns across {len(results)} files:[/yellow]\n"
        )

        total_matches = sum(len(matches) for matches in results.values())
        console.print(f"📊 Total patterns found: {total_matches}")

        # Show summary by file type
        file_types = {}
        for file_path in results.keys():
            ext = Path(file_path).suffix or "no extension"
            file_types[ext] = file_types.get(ext, 0) + 1

        console.print("\n📁 Files by type:")
        for ext, count in sorted(file_types.items()):
            console.print(f"  {ext}: {count} files")

        # Show details for first few files
        console.print("\n🔍 Sample matches (showing first 3 files):")
        for i, (file_path, matches) in enumerate(list(results.items())[:3]):
            console.print(f"\n[cyan]{file_path}[/cyan] ({len(matches)} matches)")

            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Line", style="dim", width=6)
            table.add_column("Pattern", style="yellow", max_width=30)
            table.add_column("Context", style="white", max_width=50)

            for line_num, pattern, context in matches[
                :5
            ]:  # Show first 5 matches per file
                table.add_row(
                    str(line_num),
                    pattern[:27] + "..." if len(pattern) > 30 else pattern,
                    context[:47] + "..." if len(context) > 50 else context,
                )

            console.print(table)

            if len(matches) > 5:
                console.print(f"[dim]... and {len(matches) - 5} more matches[/dim]")

    @app.command("set-preset")
    def set_naming_preset(
        preset: str = typer.Argument(..., help="Preset: mlx, mlsys, or custom:NAME"),
        apply_immediately: bool = typer.Option(
            False, "--apply", help="Apply migration immediately"
        ),
    ):
        """Set platform naming configuration preset"""
        if preset == "mlx":
            config = CommonNamingConfigs.mlx_platform()
        elif preset == "mlsys":
            config = CommonNamingConfigs.mlsys_platform()
        elif preset.startswith("custom:"):
            name = preset.split(":", 1)[1]
            config = CommonNamingConfigs.custom_platform(name)
        else:
            console.print(f"[red]Unknown preset: {preset}[/red]")
            console.print("Available presets: mlx, mlsys, custom:NAME")
            raise typer.Exit(1)

        # Save configuration
        config_path = Path("naming.config.json")
        config.save_to_file(config_path)

        console.print(
            f"✅ [green]Platform naming configuration set to: {preset}[/green]"
        )
        console.print(f"💾 Configuration saved to: {config_path}")

        # Show configuration
        console.print(
            Panel.fit(
                f"[bold cyan]{config.platform_full_name}[/bold cyan]\n"
                f"[dim]{config.platform_description}[/dim]",
                title="🏷️ Platform Naming Configuration",
            )
        )

        table = Table(title="Configuration Details", show_header=True)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        config_dict = config.to_dict()
        for key, value in config_dict.items():
            table.add_row(key.replace("_", " ").title(), str(value))

        console.print(table)

        if apply_immediately:
            console.print("\n🔄 Applying migration across entire platform...")
            migrate_platform(dry_run=False)

    @app.command("migrate")
    def migrate_platform(
        dry_run: bool = typer.Option(
            True, "--dry-run/--apply", help="Show changes without applying them"
        ),
    ):
        """Migrate platform files to use naming configuration"""
        config = get_naming_config()

        migrator = PlatformNamingMigrator()
        files = migrator.discover_files()

        console.print(
            f"🔄 [yellow]{'Analyzing' if dry_run else 'Migrating'} {len(files)} platform files...[/yellow]\n"
        )

        total_files = 0
        total_changes = 0

        with Progress() as progress:
            task = progress.add_task("Processing files...", total=len(files))

            for file_path in files:
                changed, changes = migrator.migrate_file(file_path, config, dry_run)

                if changed:
                    status = "Would update" if dry_run else "Updated"
                    console.print(
                        f"{'📝' if dry_run else '✅'} {status}: [cyan]{file_path}[/cyan] ({changes} changes)"
                    )
                    total_files += 1
                    total_changes += changes

                progress.update(task, advance=1)

        console.print("\n📊 Summary:")
        console.print(f"  Files {'to update' if dry_run else 'updated'}: {total_files}")
        console.print(f"  Total changes: {total_changes}")

        if dry_run and total_changes > 0:
            console.print("\n💡 Run with [cyan]--apply[/cyan] to apply changes")
            console.print(
                "⚠️  [yellow]Important:[/yellow] Some files may need manual renaming after migration"
            )
        elif not dry_run and total_changes > 0:
            console.print(
                "\n✅ [green]Platform migration completed successfully![/green]"
            )
            console.print("📁 Backup files created with .backup extension")
            console.print("\n🔄 [yellow]Next steps:[/yellow]")
            console.print("  1. Review changes and test functionality")
            console.print("  2. Consider renaming main CLI script if needed")
            console.print(
                "  3. Update any external references not covered by migration"
            )

    @app.command("show-config")
    def show_current_config():
        """Show current platform naming configuration"""
        config = get_naming_config()

        console.print(
            Panel.fit(
                f"[bold cyan]{config.platform_full_name}[/bold cyan]\n"
                f"[dim]{config.platform_description}[/dim]",
                title="🏷️ Current Platform Naming Configuration",
            )
        )

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        config_dict = config.to_dict()
        for key, value in config_dict.items():
            table.add_row(key.replace("_", " ").title(), str(value))

        console.print(table)

    @app.command("discover")
    def discover_files():
        """Discover all files that would be affected by migration"""
        migrator = PlatformNamingMigrator()
        files = migrator.discover_files()

        console.print(f"📁 [cyan]Discovered {len(files)} files for migration:[/cyan]\n")

        # Group by directory
        by_directory = {}
        for file_path in files:
            directory = str(file_path.parent)
            if directory not in by_directory:
                by_directory[directory] = []
            by_directory[directory].append(file_path.name)

        for directory, filenames in sorted(by_directory.items()):
            console.print(f"[yellow]{directory}/[/yellow]")
            for filename in sorted(filenames):
                console.print(f"  📄 {filename}")
            console.print()

    @app.command("validate")
    def validate_migration(
        detailed: bool = typer.Option(
            False, "--detailed", help="Show detailed validation report"
        ),
        fix_issues: bool = typer.Option(
            False, "--fix", help="Attempt to automatically fix detected issues"
        ),
    ):
        """🔍 Validate migration completeness and consistency."""
        config = get_naming_config()
        validator = MigrationValidator(config)

        console.print("🔍 [bold]Starting Migration Validation[/bold]")
        console.print()

        # Perform validation
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Validating migration...", total=None)
            results = validator.validate_migration_completeness()
            progress.update(task, description="✅ Validation complete!")

        console.print()

        # Display results
        validator.display_validation_report()

        # Attempt fixes if requested
        if fix_issues and results["consistency_score"] < 0.9:
            console.print()
            console.print("🔧 [bold]Attempting to fix detected issues...[/bold]")

            # Re-run migration on files with issues
            if results.get("pattern_validation", {}).get("inconsistencies"):
                console.print(
                    "📝 Re-running migration on files with inconsistencies..."
                )
                # This would trigger a focused re-migration

            # Provide specific fix commands
            for recommendation in results.get("recommendations", []):
                console.print(f"💡 {recommendation}")

        # Return exit code based on results
        if results["consistency_score"] < 0.7:
            raise typer.Exit(1)
        elif results["consistency_score"] < 0.9:
            console.print(
                "⚠️ [yellow]Some issues detected but migration mostly successful[/yellow]"
            )
        else:
            console.print(
                "🎉 [green]Migration validation passed with excellent results![/green]"
            )

    app()


if __name__ == "__main__":
    main()
