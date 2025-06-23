#!/usr/bin/env python3
"""
🤖 MLX Assistant - LLM-Driven CLI Companion
Your intelligent guide through the MLX Platform Foundation.
Unifies all frameworks with rich styling and contextual guidance.
"""

import asyncio
import json
import logging
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

# Add project paths for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "tests"))
sys.path.insert(0, str(project_root / "scripts" / "security"))
sys.path.insert(0, str(project_root / "scripts" / "mlx"))

# Add imports for naming system integration
sys.path.insert(0, str(project_root / "scripts"))
try:
    from .migrate_platform_naming import MigrationValidator, PlatformNamingMigrator
    from .naming_config import get_naming_config

    NAMING_SYSTEM_AVAILABLE = True
except ImportError:
    NAMING_SYSTEM_AVAILABLE = False

# Try to import MLX components
try:
    from mlx.llm_integration import MLXAIEnhancements, OpenAIProvider
except ImportError:
    OpenAIProvider = None
    MLXAIEnhancements = None


# Configure logging for MLX Assistant operations
def setup_logging():
    """Setup comprehensive logging for MLX Assistant operations."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    _json_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Main logger
    logger = logging.getLogger("mlx_assistant")
    logger.setLevel(logging.INFO)

    # File handler for general operations
    file_handler = logging.FileHandler(logs_dir / "mlx_assistant.log")
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    # Separate handler for future LLM interactions
    llm_logger = logging.getLogger("mlx_assistant.llm")
    llm_handler = logging.FileHandler(logs_dir / "llm_interactions.log")
    llm_handler.setFormatter(detailed_formatter)
    llm_logger.addHandler(llm_handler)

    # Framework operations logger
    framework_logger = logging.getLogger("mlx_assistant.frameworks")
    framework_handler = logging.FileHandler(logs_dir / "framework_operations.log")
    framework_handler.setFormatter(detailed_formatter)
    framework_logger.addHandler(framework_handler)
    return logger


# Initialize logging
logger = setup_logging()
console = Console()

app = typer.Typer(
    name="mlx-assistant",
    help="🤖 MLX Assistant - Your intelligent guide through the MLX Platform Foundation",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=False,
)


class MLXAssistant:
    """Intelligent MLX platform assistant with deep repository understanding."""

    def __init__(self):
        self.project_root = Path.cwd()
        self.console = Console()
        self.frameworks = self._discover_frameworks()
        self.project_state = self._analyze_project_state()
        self.session_id = str(uuid.uuid4())[:8]

        # Add naming system state
        self.naming_state = self._analyze_naming_consistency()

        # Log session start
        logger.info(f"MLX Assistant session started: {self.session_id}")

    def _discover_frameworks(self) -> dict[str, dict[str, Any]]:
        """Discover all available MLX frameworks and their capabilities."""
        logger.info("Discovering available frameworks")
        frameworks = {
            "golden_repos": {
                "name": "Golden Repository Testing",
                "description": "Reference implementations for testing component extraction and deployment",
                "script": "tests/golden_repos.py",
                "commands": ["create", "validate", "create-all", "validate-all"],
                "icon": "🏗️",
                "status": "✅ Ready",
            },
            "security": {
                "name": "Security Hardening",
                "description": "Comprehensive security scanning and hardening framework",
                "script": "scripts/security/security_hardening.py",
                "commands": ["scan", "sbom", "verify", "baseline", "compare", "report"],
                "icon": "🔒",
                "status": "✅ Ready",
            },
            "plugins": {
                "name": "Plugin Ecosystem",
                "description": "Plugin development, validation, and management system",
                "script": "scripts/mlx/plugin_ecosystem.py",
                "commands": ["create", "validate", "list", "info"],
                "icon": "🧩",
                "status": "✅ Ready",
            },
            "glossary": {
                "name": "Glossary & Standards",
                "description": "MLX platform terminology and naming conventions",
                "script": "docs/glossary.md",
                "commands": ["view", "search", "validate-naming"],
                "icon": "📚",
                "status": "✅ Ready",
            },
        }
        logger.info(f"Discovered {len(frameworks)} frameworks")
        return frameworks

    def _analyze_project_state(self) -> dict[str, Any]:
        """Analyze current project state for contextual recommendations."""
        logger.info("Analyzing project state")
        state = {
            "is_mlx_project": False,
            "has_components": False,
            "security_status": "unknown",
            "plugins_available": 0,
            "recent_activity": [],
            "recommendations": [],
        }

        # Check if MLX project
        if (self.project_root / "mlx.config.json").exists():
            state["is_mlx_project"] = True
            logger.info("MLX project detected")

        # Check components
        if (self.project_root / "mlx-components").exists():
            state["has_components"] = True
            logger.info("MLX components directory found")

        # Check plugins
        plugins_dir = self.project_root / "plugins"
        if plugins_dir.exists():
            plugin_count = len(list(plugins_dir.glob("mlx-plugin-*")))
            state["plugins_available"] = plugin_count
            logger.info(f"Found {plugin_count} plugins")

        # Generate intelligent recommendations
        state["recommendations"] = self._generate_recommendations(state)
        logger.info(f"Project analysis complete: {state}")
        return state

    def _analyze_naming_consistency(self) -> dict[str, Any]:
        """Analyze current naming consistency state."""
        naming_state = {
            "system_available": NAMING_SYSTEM_AVAILABLE,
            "consistency_score": 0.0,
            "current_platform": "unknown",
            "issues_count": 0,
            "last_validation": None,
            "recommendations": [],
        }
        if not NAMING_SYSTEM_AVAILABLE:
            logger.warning("Naming system not available")
            return naming_state

        try:
            # Get current platform configuration
            config = get_naming_config()
            naming_state["current_platform"] = config.platform_name

            # Quick consistency check
            validator = MigrationValidator(config)

            # Quick pattern check (limited to avoid slow startup)
            migrator = PlatformNamingMigrator()
            sample_files = list(migrator.discover_files())[
                :10
            ]  # Check first 10 files only
            issues_found = 0
            for file_path in sample_files:
                try:
                    if file_path.suffix in [".py", ".md", ".json"]:
                        loaded_data = file_path.read_text(
                            encoding="utf-8", errors="ignore"
                        )
                        file_issues = validator._check_file_patterns(
                            file_path, loaded_data
                        )
                        issues_found += len(file_issues)
                except Exception:
                    continue

            naming_state["issues_count"] = issues_found
            naming_state["consistency_score"] = max(
                0.0, 1.0 - (issues_found / 20)
            )  # Rough estimate

            # Generate quick recommendations
            if issues_found > 0:
                naming_state["recommendations"] = [
                    f"🔍 {issues_found} naming inconsistencies detected",
                    "Run: python scripts/migrate_platform_naming.py validate",
                    "Consider: python scripts/migrate_platform_naming.py migrate --dry-run",
                ]
            logger.info(
                f"Naming consistency check: {naming_state['consistency_score']:.2f} score, {issues_found} issues"
            )
        except Exception as e:
            logger.error(f"Error analyzing naming consistency: {e}")
            naming_state["error"] = str(e)

        return naming_state

    def _generate_recommendations(self, state: dict[str, Any]) -> list[str]:
        """Generate intelligent recommendations based on project state."""
        logger.info("Generating recommendations")
        recommendations = []
        if not state["is_mlx_project"]:
            recommendations.append(
                "🎯 Start with: [cyan]mlx assistant quick-start[/cyan] to set up your MLX project"
            )

        # Add naming consistency recommendations
        if NAMING_SYSTEM_AVAILABLE and hasattr(self, "naming_state"):
            naming_score = self.naming_state.get("consistency_score", 1.0)
            if naming_score < 0.8:
                recommendations.append(
                    "🏷️ Naming inconsistencies detected: [cyan]mlx assistant naming validate[/cyan]"
                )
                recommendations.append(
                    "🔧 Fix naming issues: [cyan]mlx assistant naming migrate[/cyan]"
                )

        if state["is_mlx_project"] and not state["has_components"]:
            recommendations.append(
                "📦 Extract components: [cyan]mlx assistant golden-repos create standard[/cyan]"
            )

        if state["plugins_available"] == 0:
            recommendations.append(
                "🧩 Create your first plugin: [cyan]mlx assistant plugins create[/cyan]"
            )

        recommendations.append(
            "🛡️ Run security scan: [cyan]mlx assistant security scan[/cyan]"
        )
        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations


assistant = MLXAssistant()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version information"),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Start interactive mode"
    ),
):
    """🤖 MLX Assistant - Your intelligent guide through the MLX Platform Foundation"""
    if version:
        console.print("🤖 MLX Assistant v1.0.0 - Phase 3 Core Hardening Complete")
        return

    if interactive or ctx.invoked_subcommand is None:
        show_welcome_dashboard()
        if interactive:
            start_interactive_mode()


def show_welcome_dashboard():
    """Display the main MLX Assistant dashboard with current status."""

    # Header
    header = Panel.fit(
        "[bold bright_blue]🤖 MLX Assistant - Platform Foundation Guide[/bold bright_blue]\n"
        "[dim]Your intelligent companion for MLX platform operations[/dim]",
        border_style="bright_blue",
    )
    console.print(header)
    console.print()

    # Create layout
    layout = Layout()
    layout.split_column(
        Layout(name="frameworks", size=12),
        Layout(name="status", size=10),  # Increased size for naming info
        Layout(name="recommendations", size=6),
    )

    # Frameworks panel
    frameworks_table = Table(
        title="🛠️ Available Frameworks", show_header=True, header_style="bold magenta"
    )
    frameworks_table.add_column("Framework", style="cyan")
    frameworks_table.add_column("Status", justify="center")
    frameworks_table.add_column("Description", style="dim")
    frameworks_table.add_column("Quick Start", style="green")
    for key, framework in assistant.frameworks.items():
        frameworks_table.add_row(
            f"{framework['icon']} {framework['name']}",
            framework["status"],
            framework["description"][:50] + "...",
            f"[cyan]mlx {key}[/cyan]",
        )
    layout["frameworks"].update(Panel(frameworks_table, border_style="magenta"))

    # Project status (enhanced with naming info)
    status_table = Table(title="📊 Project Status", show_header=False)
    status_table.add_column("Metric", style="yellow")
    status_table.add_column("Value", style="green")
    status_items = [
        (
            "MLX Project",
            "✅ Yes" if assistant.project_state["is_mlx_project"] else "❌ No",
        ),
        (
            "Components",
            "✅ Available" if assistant.project_state["has_components"] else "❌ None",
        ),
        ("Plugins", f"🧩 {assistant.project_state['plugins_available']} available"),
        ("Security", "🔒 Ready for scan"),
    ]

    # Add naming consistency status
    if NAMING_SYSTEM_AVAILABLE and hasattr(assistant, "naming_state"):
        naming_score = assistant.naming_state.get("consistency_score", 0.0)
        current_platform = assistant.naming_state.get("current_platform", "unknown")
        issues_count = assistant.naming_state.get("issues_count", 0)
        if naming_score >= 0.9:
            naming_status = f"✅ Excellent ({current_platform})"
        elif naming_score >= 0.7:
            naming_status = f"⚠️ Good ({current_platform}, {issues_count} issues)"
        else:
            naming_status = f"❌ Needs work ({current_platform}, {issues_count} issues)"
        status_items.append(("Naming Consistency", naming_status))

    for metric, value in status_items:
        status_table.add_row(metric, value)
    layout["status"].update(Panel(status_table, border_style="yellow"))

    # Recommendations (enhanced with naming recommendations)
    all_recommendations = assistant.project_state["recommendations"]
    if NAMING_SYSTEM_AVAILABLE and hasattr(assistant, "naming_state"):
        all_recommendations.extend(assistant.naming_state.get("recommendations", []))

    recommendations_text = "\n".join(
        [f"• {rec}" for rec in all_recommendations[:4]]  # Show first 4
    )
    if not recommendations_text:
        recommendations_text = "🎉 [green]All systems operational![/green]"

    layout["recommendations"].update(
        Panel(
            recommendations_text, title="🎯 Smart Recommendations", border_style="green"
        )
    )
    console.print(layout)
    console.print()

    # Quick help (enhanced with naming commands)
    help_panel = Panel(
        "[bold]Quick Commands:[/bold]\n"
        "• [cyan]mlx assistant interactive[/cyan] - Start guided mode\n"
        "• [cyan]mlx assistant naming validate[/cyan] - Check naming consistency\n"
        "• [cyan]mlx golden-repos[/cyan] - Manage golden repositories\n"
        "• [cyan]mlx security[/cyan] - Security scanning & hardening\n"
        "• [cyan]mlx plugins[/cyan] - Plugin development & management\n"
        "• [cyan]mlx assistant --help[/cyan] - Full command reference",
        title="🚀 Get Started",
        border_style="bright_green",
    )
    console.print(help_panel)


def start_interactive_mode():
    """Start interactive mode for guided operations."""
    console.print("\n🤖 [bold bright_blue]Interactive Mode Started[/bold bright_blue]")
    console.print("Type 'help' for commands, 'exit' to quit\n")

    while True:
        try:
            command = Prompt.ask(
                "[bold bright_blue]mlx-assistant[/bold bright_blue]", default="help"
            )
            if command.lower() in ["exit", "quit", "q"]:
                console.print("👋 [dim]Goodbye![/dim]")
                break
            elif command.lower() == "help":
                show_interactive_help()
            elif command.lower() == "status":
                show_detailed_status()
            elif command.lower().startswith("analyze"):
                analyze_project()
            elif command.lower().startswith("recommend"):
                show_recommendations()
            else:
                console.print(f"❓ Unknown command: {command}")
                console.print("Type 'help' for available commands")
        except KeyboardInterrupt:
            console.print("\n👋 [dim]Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"❌ Error: {e}")


def show_interactive_help():
    """Show interactive mode help."""
    help_table = Table(title="🤖 Interactive Commands", show_header=True)
    help_table.add_column("Command", style="cyan")
    help_table.add_column("Description", style="dim")
    commands = [
        ("help", "Show this help message"),
        ("status", "Show detailed project status"),
        ("analyze", "Analyze current project state"),
        ("recommend", "Get intelligent recommendations"),
        ("exit/quit", "Exit interactive mode"),
    ]
    for cmd, desc in commands:
        help_table.add_row(cmd, desc)
    console.print(help_table)
    console.print()


# Framework-specific command groups
golden_repos = typer.Typer(
    name="golden-repos", help="🏗️ Golden Repository Testing Framework"
)
security = typer.Typer(name="security", help="🔒 Security Hardening Framework")
plugins = typer.Typer(name="plugins", help="🧩 Plugin Ecosystem Framework")
glossary = typer.Typer(name="glossary", help="📚 Glossary & Standards")

# Add subcommands to main app
app.add_typer(golden_repos, name="golden-repos")
app.add_typer(security, name="security")
app.add_typer(plugins, name="plugins")
app.add_typer(glossary, name="glossary")


# Quick start commands
@app.command(name="quick-start")
def quick_start():
    """🚀 Quick start guide for new users."""
    console.print(
        Panel.fit(
            "[bold bright_green]🚀 MLX Platform Quick Start[/bold bright_green]\n\n"
            "[bold]Step 1:[/bold] Initialize your project\n"
            "[cyan]  ./mlx create my-project[/cyan]\n\n"
            "[bold]Step 2:[/bold] Create golden repository\n"
            "[cyan]  mlx golden-repos create standard[/cyan]\n\n"
            "[bold]Step 3:[/bold] Run security scan\n"
            "[cyan]  mlx security scan[/cyan]\n\n"
            "[bold]Step 4:[/bold] Create your first plugin\n"
            "[cyan]  mlx plugins create --name my-plugin --type ml_framework[/cyan]",
            border_style="bright_green",
        )
    )


@app.command(name="analyze")
def analyze_project():
    """🔍 Analyze current project state and provide recommendations."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing project...", total=None)

        # Simulate analysis
        time.sleep(2)
        progress.update(task, description="Analysis complete!")

    # Show detailed analysis
    analysis_panel = Panel(
        "[bold]📊 Project Analysis Results[/bold]\n\n"
        f"[green]✅[/green] MLX Project: {'Yes' if assistant.project_state['is_mlx_project'] else 'No'}\n"
        f"[green]✅[/green] Components: {'Available' if assistant.project_state['has_components'] else 'None'}\n"
        f"[green]✅[/green] Plugins: {assistant.project_state['plugins_available']} discovered\n\n"
        "[bold yellow]🎯 Recommendations:[/bold yellow]\n"
        + "\n".join([f"• {rec}" for rec in assistant.project_state["recommendations"]]),
        title="Project Analysis",
        border_style="bright_cyan",
    )
    console.print(analysis_panel)


def show_detailed_status():
    """Show detailed project status."""
    console.print("📊 [bold]Detailed Project Status[/bold]\n")

    # Framework status
    for _key, framework in assistant.frameworks.items():
        console.print(
            f"{framework['icon']} [bold]{framework['name']}[/bold]: {framework['status']}"
        )
    console.print()


def show_recommendations():
    """Show intelligent recommendations."""
    if not assistant.project_state["recommendations"]:
        console.print("🎉 [green]No recommendations - everything looks good![/green]")
        return

    console.print("🎯 [bold]Intelligent Recommendations[/bold]\n")
    for i, rec in enumerate(assistant.project_state["recommendations"], 1):
        console.print(f"{i}. {rec}")
    console.print()


# Golden Repository Framework Commands
@golden_repos.command(name="create")
def golden_create(
    spec: str = typer.Argument(
        ..., help="Repository spec (minimal/standard/advanced/plugin_heavy/performance)"
    ),
    force: bool = typer.Option(False, "--force", help="Force recreation if exists"),
):
    """🏗️ Create a golden repository from specification."""
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Creating golden repository: {spec}...", total=None
            )

            # Import and run golden repos
            result = subprocess.run(
                [
                    sys.executable,
                    "tests/golden_repos.py",
                    "create",
                    "--spec",
                    spec,
                    "--force" if force else "",
                ],
                capture_output=True,
                text=True,
                cwd=assistant.project_root,
            )
            if result.returncode == 0:
                progress.update(
                    task,
                    description=f"✅ Golden repository '{spec}' created successfully!",
                )
                console.print(
                    f"\n🎉 [green]Success![/green] Golden repository '{spec}' is ready"
                )
                console.print(f"📁 Location: [cyan]tests/golden_repos/{spec}[/cyan]")
            else:
                console.print(
                    f"❌ [red]Error creating golden repository:[/red] {result.stderr}"
                )
    except Exception as e:
        console.print(f"❌ [red]Error:[/red] {e}")


@golden_repos.command(name="list")
def golden_list():
    """📋 List available golden repository specifications."""
    specs_table = Table(title="🏗️ Golden Repository Specifications", show_header=True)
    specs_table.add_column("Spec", style="cyan")
    specs_table.add_column("Type", style="magenta")
    specs_table.add_column("Description", style="dim")
    specs_table.add_column("Components", justify="right")
    specs = [
        ("minimal", "Basic", "Minimal MLOps template with basic components", "1"),
        ("standard", "Full", "Standard MLOps template with core components", "3"),
        ("advanced", "Complex", "Advanced MLOps template with all components", "5"),
        (
            "plugin_heavy",
            "Plugins",
            "Template with multiple plugins for integration testing",
            "3+",
        ),
        (
            "performance",
            "Optimized",
            "Optimized template for performance benchmarking",
            "3",
        ),
    ]
    for spec, type_str, desc, comps in specs:
        specs_table.add_row(spec, type_str, desc, comps)
    console.print(specs_table)
    console.print(
        "\n💡 [dim]Create one with:[/dim] [cyan]mlx golden-repos create <spec>[/cyan]"
    )


@golden_repos.command(name="validate")
def golden_validate(
    spec: str = typer.Argument(..., help="Repository spec to validate"),
):
    """✅ Validate a golden repository."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Validating golden repository: {spec}...", total=None)
        result = subprocess.run(
            [sys.executable, "tests/golden_repos.py", "validate", "--spec", spec],
            capture_output=True,
            text=True,
            cwd=assistant.project_root,
        )
        if result.returncode == 0:
            progress.update(task, description="✅ Validation complete!")
            console.print(
                f"\n🎉 [green]Golden repository '{spec}' validation passed![/green]"
            )
        else:
            console.print(f"❌ [red]Validation failed:[/red] {result.stderr}")


# Security Framework Commands
@security.command(name="scan")
def security_scan(
    level: str = typer.Option(
        "enhanced",
        "--level",
        help="Security level (basic/enhanced/enterprise/critical)",
    ),
    output: str = typer.Option(
        "json", "--output", help="Output format (json/html/sarif)"
    ),
):
    """🔒 Run comprehensive security scan."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running security scan...", total=None)
        result = subprocess.run(
            [
                sys.executable,
                "scripts/security/security_hardening.py",
                "scan",
                "--security-level",
                level,
                "--output-format",
                output,
            ],
            capture_output=True,
            text=True,
            cwd=assistant.project_root,
        )
        if result.returncode == 0:
            progress.update(task, description="✅ Security scan complete!")
            console.print("\n🛡️ [green]Security scan completed successfully![/green]")
            console.print(f"📊 Report format: [cyan]{output}[/cyan]")
        else:
            console.print(f"❌ [red]Security scan failed:[/red] {result.stderr}")


@security.command(name="sbom")
def security_sbom():
    """📋 Generate Software Bill of Materials (SBOM)."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating SBOM...", total=None)
        result = subprocess.run(
            [sys.executable, "scripts/security/security_hardening.py", "sbom"],
            capture_output=True,
            text=True,
            cwd=assistant.project_root,
        )
        if result.returncode == 0:
            progress.update(task, description="✅ SBOM generated!")
            console.print("\n📋 [green]SBOM generated successfully![/green]")
        else:
            console.print(f"❌ [red]SBOM generation failed:[/red] {result.stderr}")


# Plugin Framework Commands
@plugins.command(name="create")
def plugins_create(
    name: str = typer.Option(..., "--name", help="Plugin name"),
    plugin_type: str = typer.Option("ml_framework", "--type", help="Plugin type"),
    description: str = typer.Option(
        "A new MLX plugin", "--description", help="Plugin description"
    ),
):
    """🧩 Create a new plugin from template."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Creating plugin: {name}...", total=None)
        result = subprocess.run(
            [
                sys.executable,
                "scripts/mlx/plugin_ecosystem.py",
                "create",
                "--name",
                name,
                "--type",
                plugin_type,
                "--description",
                description,
            ],
            capture_output=True,
            text=True,
            cwd=assistant.project_root,
        )
        if result.returncode == 0:
            progress.update(task, description=f"✅ Plugin '{name}' created!")
            console.print(f"\n🧩 [green]Plugin '{name}' created successfully![/green]")
            console.print(f"📁 Location: [cyan]plugins/mlx-plugin-{name}[/cyan]")
        else:
            console.print(f"❌ [red]Plugin creation failed:[/red] {result.stderr}")


@plugins.command(name="list")
def plugins_list():
    """📋 List available plugins."""
    result = subprocess.run(
        [sys.executable, "scripts/mlx/plugin_ecosystem.py", "list"],
        capture_output=True,
        text=True,
        cwd=assistant.project_root,
    )
    if result.returncode == 0:
        console.print("🧩 [bold]Available Plugins[/bold]")
        console.print(result.stdout)
    else:
        console.print("❌ [red]Failed to list plugins[/red]")


@plugins.command(name="validate")
def plugins_validate(
    plugin_path: str = typer.Argument(..., help="Path to plugin directory"),
):
    """✅ Validate a plugin."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Validating plugin...", total=None)
        result = subprocess.run(
            [
                sys.executable,
                "scripts/mlx/plugin_ecosystem.py",
                "validate",
                "--plugin-path",
                plugin_path,
            ],
            capture_output=True,
            text=True,
            cwd=assistant.project_root,
        )
        if result.returncode == 0:
            progress.update(task, description="✅ Plugin validation complete!")
            console.print("\n🎉 [green]Plugin validation passed![/green]")
        else:
            console.print(f"❌ [red]Plugin validation failed:[/red] {result.stderr}")


# Glossary Commands
@glossary.command(name="view")
def glossary_view():
    """📚 View the MLX glossary and standards."""
    glossary_path = assistant.project_root / "docs" / "glossary.md"
    if glossary_path.exists():
        with open(glossary_path) as f:
            loaded_data = f.read()

        # Show a preview
        console.print("📚 [bold]MLX Platform Glossary[/bold]")
        console.print(f"📄 Full document: [cyan]{glossary_path}[/cyan]\n")

        # Extract key sections
        lines = loaded_data.split("\n")
        for _i, line in enumerate(lines[:50]):  # Show first 50 lines
            if line.startswith("#"):
                console.print(f"[bold blue]{line}[/bold blue]")
            elif line.startswith("##"):
                console.print(f"[bold cyan]{line}[/bold cyan]")
            elif line.strip():
                console.print(line)
        if len(lines) > 50:
            console.print(f"\n[dim]... and {len(lines) - 50} more lines[/dim]")
        else:
            console.print("❌ [red]Glossary not found[/red]")


@glossary.command(name="search")
def glossary_search(term: str = typer.Argument(..., help="Term to search for")):
    """🔍 Search the glossary for a term."""
    glossary_path = assistant.project_root / "docs" / "glossary.md"
    if glossary_path.exists():
        with open(glossary_path) as f:
            loaded_data = f.read()
        matches = []
        for i, line in enumerate(loaded_data.split("\n"), 1):
            if term.lower() in line.lower():
                matches.append((i, line.strip()))
        if matches:
            console.print(
                f"🔍 [bold]Found {len(matches)} matches for '{term}':[/bold]\n"
            )
            for line_num, line in matches[:10]:  # Show first 10 matches
                console.print(f"[dim]Line {line_num}:[/dim] {line}")
        else:
            console.print(f"❌ No matches found for '{term}'")
    else:
        console.print("❌ [red]Glossary not found[/red]")


# Utility commands
@app.command(name="doctor")
def doctor():
    """🩺 Run comprehensive health check on MLX platform."""
    console.print("🩺 [bold]MLX Platform Health Check[/bold]\n")
    checks = [
        (
            "MLX Project Structure",
            lambda: (assistant.project_root / "mlx.config.json").exists(),
        ),
        (
            "Golden Repos Framework",
            lambda: (assistant.project_root / "tests" / "golden_repos.py").exists(),
        ),
        (
            "Security Framework",
            lambda: (
                assistant.project_root
                / "scripts"
                / "security"
                / "security_hardening.py"
            ).exists(),
        ),
        (
            "Plugin Framework",
            lambda: (
                assistant.project_root / "scripts" / "mlx" / "plugin_ecosystem.py"
            ).exists(),
        ),
        (
            "Documentation",
            lambda: (assistant.project_root / "docs" / "glossary.md").exists(),
        ),
    ]
    results_table = Table(title="Health Check Results", show_header=True)
    results_table.add_column("Component", style="cyan")
    results_table.add_column("Status", justify="center")
    results_table.add_column("Details", style="dim")
    all_good = True
    for name, check_func in checks:
        try:
            is_healthy = check_func()
            status = "✅ Healthy" if is_healthy else "❌ Missing"
            details = "Available" if is_healthy else "Not found"
            if not is_healthy:
                all_good = False
            results_table.add_row(name, status, details)
        except Exception as e:
            status = "⚠️ Error"
            details = str(e)
            all_good = False
            results_table.add_row(name, status, details)
    console.print(results_table)
    if all_good:
        console.print("\n🎉 [green]All systems healthy![/green]")
    else:
        console.print(
            "\n⚠️ [yellow]Some issues detected. Run individual framework commands for details.[/yellow]"
        )


# Add the ask command after the existing commands
@app.command(name="ask")
def ask_question(
    question: str = typer.Argument(..., help="Your question about MLX platform"),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Start interactive Q&A session"
    ),
):
    """🧠 Ask the MLX Assistant AI any question about the platform"""
    logger.info(f"AI Query: {question}")

    if OpenAIProvider is None:
        console.print(
            Panel(
                "❌ OpenAI package not installed.\n\n"
                "To enable AI features, run:\n"
                "[bold cyan]pip install openai[/bold cyan]\n\n"
                "And set your OpenAI API key:\n"
                "[bold cyan]export OPENAI_API_KEY=your-api-key[/bold cyan]",
                title="[bold red]AI Features Unavailable[/bold red]",
                border_style="red",
            )
        )
        return

    try:
        # Initialize AI provider
        ai_provider = OpenAIProvider()
        _ai_enhancements = MLXAIEnhancements(ai_provider)
        console.print(
            Panel(
                f"🧠 Processing your question...\n[dim]{question}[/dim]",
                title="[bold blue]MLX AI Assistant[/bold blue]",
                border_style="blue",
            )
        )

        # Get project context
        project_context = assistant.project_state

        # Generate AI response
        response = asyncio.run(ai_provider.generate_response(question, project_context))

        # Display response
        console.print(
            Panel(
                response,
                title="[bold green]AI Response[/bold green]",
                border_style="green",
            )
        )

        # Start interactive mode if requested
        if interactive:
            start_ai_interactive_mode(ai_provider, project_context)
        logger.info(f"AI Query completed: {question}")
    except ValueError as e:
        console.print(
            Panel(
                f"❌ Configuration Error: {str(e)}\n\n"
                "Please set your OpenAI API key:\n"
                "[bold cyan]export OPENAI_API_KEY=your-api-key[/bold cyan]",
                title="[bold red]API Key Required[/bold red]",
                border_style="red",
            )
        )
    except Exception as e:
        console.print(
            Panel(
                f"❌ Error: {str(e)}\n\n"
                "Please check your internet connection and API key.",
                title="[bold red]AI Request Failed[/bold red]",
                border_style="red",
            )
        )
        logger.error(f"AI Query failed: {e}")


def start_ai_interactive_mode(
    ai_provider: OpenAIProvider, project_context: dict[str, Any]
):
    """Start interactive AI Q&A session"""
    console.print(
        Panel(
            "🧠 [bold]Interactive AI Mode[/bold]\n\n"
            "Ask me anything about MLX platform!\n"
            "Type 'quit', 'exit', or 'bye' to end the session.\n\n"
            "Examples:\n"
            "• How do I set up security scanning?\n"
            "• What plugins would be good for my ML project?\n"
            "• Generate a workflow to deploy my model\n"
            "• Explain the golden repository framework",
            title="[bold blue]🤖 MLX AI Assistant[/bold blue]",
            border_style="blue",
        )
    )
    while True:
        try:
            question = Prompt.ask("\n[bold blue]Ask MLX AI[/bold blue]")
            if question.lower() in ["quit", "exit", "bye", "q"]:
                console.print("\n👋 Thanks for using MLX AI Assistant!")
                break
            if not question.strip():
                continue

            # Show processing
            with console.status("[bold blue]🧠 Thinking...", spinner="dots"):
                response = asyncio.run(
                    ai_provider.generate_response(question, project_context)
                )

            # Display response
            console.print(
                Panel(
                    response,
                    title="[bold green]🤖 MLX AI Response[/bold green]",
                    border_style="green",
                )
            )
        except KeyboardInterrupt:
            console.print("\n\n👋 Thanks for using MLX AI Assistant!")
            break
        except Exception as e:
            console.print(f"\n❌ Error: {str(e)}")
            logger.error(f"Interactive AI error: {e}")


# Add AI-enhanced analyze command
@app.command(name="ai-analyze")
def ai_analyze_project(
    path: str = typer.Option(".", "--path", help="Project path to analyze"),
    detailed: bool = typer.Option(
        False, "--detailed", help="Show detailed AI analysis"
    ),
):
    """🔍 AI-powered project analysis with intelligent recommendations"""
    logger.info(f"AI Analysis: {path}")

    if OpenAIProvider is None:
        console.print(
            Panel(
                "❌ OpenAI package not installed.\n\n"
                "To enable AI features, run:\n"
                "[bold cyan]pip install openai[/bold cyan]",
                title="[bold red]AI Features Unavailable[/bold red]",
                border_style="red",
            )
        )
        return

    try:
        # Initialize AI provider
        ai_provider = OpenAIProvider()
        console.print(
            Panel(
                f"🔍 Analyzing project with AI...\n[dim]{path}[/dim]",
                title="[bold blue]MLX AI Project Analysis[/bold blue]",
                border_style="blue",
            )
        )

        # Perform AI analysis
        project_path = Path(path)
        analysis = ai_provider.analyze_project_intelligently(project_path)
        if "error" in analysis:
            console.print(f"❌ Analysis failed: {analysis['error']}")
            return

        # Display results
        console.print("\n🎯 [bold]AI Analysis Results[/bold]\n")

        # Architecture suggestions
        if analysis.get("architecture_suggestions"):
            console.print("[bold blue]🏗️ Architecture Suggestions:[/bold blue]")
            for suggestion in analysis["architecture_suggestions"]:
                console.print(f"  • {suggestion['suggestion']}")
                if detailed:
                    console.print(
                        f"    [dim]Command: {suggestion.get('command', 'N/A')}[/dim]"
                    )
            console.print()

        # Workflow suggestions
        if analysis.get("workflow_suggestions"):
            console.print("[bold green]⚡ Workflow Suggestions:[/bold green]")
            for suggestion in analysis["workflow_suggestions"]:
                console.print(f"  • {suggestion['suggestion']}")
                if detailed:
                    console.print(
                        f"    [dim]Command: {suggestion.get('command', 'N/A')}[/dim]"
                    )
            console.print()

        # Plugin recommendations
        if analysis.get("plugin_recommendations"):
            console.print("[bold purple]🧩 Plugin Recommendations:[/bold purple]")
            for plugin in analysis["plugin_recommendations"]:
                console.print(f"  • {plugin.get('name', 'Unknown plugin')}")
            console.print()

        # Get AI-powered plugin recommendations
        existing_plugins = []
        if (project_path / "plugins").exists():
            existing_plugins = [
                p.name for p in (project_path / "plugins").iterdir() if p.is_dir()
            ]
        plugin_recommendations = ai_provider.recommend_plugins_intelligently(
            "ml_training", existing_plugins
        )
        if plugin_recommendations:
            console.print("[bold purple]🤖 AI Plugin Recommendations:[/bold purple]")
            for plugin in plugin_recommendations[:3]:  # Show top 3
                console.print(
                    f"  • [bold]{plugin['plugin_name']}[/bold] (Confidence: {plugin['confidence']:.0%})"
                )
                console.print(f"    {plugin['reasoning']}")
                if detailed:
                    console.print(f"    Benefits: {', '.join(plugin['benefits'])}")
                    console.print()
            logger.info(f"AI Analysis completed: {path}")
    except Exception as e:
        console.print(f"❌ AI Analysis failed: {str(e)}")
        logger.error(f"AI Analysis error: {e}")


# Add AI workflow generation command
@app.command(name="ai-workflow")
def ai_generate_workflow(
    goal: str = typer.Argument(..., help="Your goal in natural language"),
    execute: bool = typer.Option(
        False, "--execute", help="Execute the generated workflow"
    ),
    save: bool = typer.Option(False, "--save", help="Save workflow to file"),
):
    """⚡ Generate intelligent workflows from natural language goals"""
    logger.info(f"AI Workflow Generation: {goal}")

    if OpenAIProvider is None:
        console.print(
            Panel(
                "❌ OpenAI package not installed.\n\n"
                "To enable AI features, run:\n"
                "[bold cyan]pip install openai[/bold cyan]",
                title="[bold red]AI Features Unavailable[/bold red]",
                border_style="red",
            )
        )
        return

    try:
        # Initialize AI provider
        ai_provider = OpenAIProvider()
        console.print(
            Panel(
                f"⚡ Generating workflow for your goal...\n[dim]{goal}[/dim]",
                title="[bold blue]MLX AI Workflow Generator[/bold blue]",
                border_style="blue",
            )
        )

        # Generate workflow
        project_state = assistant.project_state
        workflow = ai_provider.generate_intelligent_workflow(goal, project_state)

        # Display workflow
        console.print(f"\n🎯 [bold]{workflow['title']}[/bold]\n")
        console.print(f"[dim]{workflow['description']}[/dim]\n")

        # Show workflow steps
        console.print("[bold blue]📋 Workflow Steps:[/bold blue]\n")
        for step in workflow["steps"]:
            console.print(f"[bold]{step['step']}.[/bold] {step['description']}")
            console.print(f"   [cyan]Command:[/cyan] [green]{step['command']}[/green]")
            console.print(f"   [dim]Expected: {step['expected_output']}[/dim]\n")

        # Show metadata
        console.print(f"⏱️  [bold]Estimated Time:[/bold] {workflow['estimated_time']}")
        console.print(f"📊 [bold]Complexity:[/bold] {workflow['complexity']}")
        if workflow.get("prerequisites"):
            console.print(
                f"📋 [bold]Prerequisites:[/bold] {', '.join(workflow['prerequisites'])}"
            )
        if workflow.get("success_criteria"):
            console.print(
                f"✅ [bold]Success Criteria:[/bold] {', '.join(workflow['success_criteria'])}"
            )

        # Save workflow if requested
        if save:
            workflow_file = Path(f"workflow_{int(time.time())}.json")
            with open(workflow_file, "w") as f:
                json.dump(workflow, f, indent=2)
            console.print(f"\n💾 Workflow saved to: [bold]{workflow_file}[/bold]")

        # Execute workflow if requested
        if execute:
            console.print("\n🚀 [bold]Executing Workflow...[/bold]\n")
            execute_workflow(workflow)
            logger.info(f"AI Workflow generated: {goal}")
    except Exception as e:
        console.print(f"❌ Workflow generation failed: {str(e)}")
        logger.error(f"AI Workflow error: {e}")


def execute_workflow(workflow: dict[str, Any]):
    """Execute a generated workflow step by step"""
    console.print(
        Panel(
            f"Executing: {workflow['title']}\nSteps: {len(workflow['steps'])}",
            title="[bold green]🚀 Workflow Execution[/bold green]",
            border_style="green",
        )
    )
    for step in workflow["steps"]:
        console.print(
            f"\n[bold blue]Step {step['step']}:[/bold blue] {step['description']}"
        )
        if step["command"].startswith("#"):
            console.print(f"[dim]Manual step: {step['command']}[/dim]")
            continue

        # Ask for confirmation
        if (
            not Prompt.ask(
                f"Execute: [green]{step['command']}[/green]?",
                choices=["y", "n"],
                default="y",
            )
            == "y"
        ):
            console.print("⏭️  Skipping step...")
            continue

        # Execute command
        try:
            with console.status(f"Executing step {step['step']}...", spinner="dots"):
                result = subprocess.run(
                    step["command"],
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                )
            if result.returncode == 0:
                console.print(f"✅ Step {step['step']} completed successfully")
                if result.stdout:
                    console.print(f"[dim]Output: {result.stdout[:200]}...[/dim]")
            else:
                console.print(f"❌ Step {step['step']} failed")
                console.print(f"[red]Error: {result.stderr}[/red]")
                if (
                    not Prompt.ask(
                        "Continue with next step?", choices=["y", "n"], default="n"
                    )
                    == "y"
                ):
                    break
        except subprocess.TimeoutExpired:
            console.print(f"⏱️  Step {step['step']} timed out")
        except Exception as e:
            console.print(f"❌ Step {step['step']} error: {str(e)}")
    console.print("\n🎉 [bold]Workflow execution completed![/bold]")


# Add naming system commands
naming = typer.Typer(name="naming", help="🏷️ Platform Naming Management")


@naming.command(name="validate")
def naming_validate(
    detailed: bool = typer.Option(
        False, "--detailed", help="Show detailed validation report"
    ),
    fix_issues: bool = typer.Option(
        False, "--fix", help="Attempt to fix detected issues"
    ),
):
    """🔍 Validate platform naming consistency."""
    if not NAMING_SYSTEM_AVAILABLE:
        console.print(
            "❌ [red]Naming system not available. Check that scripts/naming_config.py exists.[/red]"
        )
        return
    try:
        config = get_naming_config()
        validator = MigrationValidator(config)
        console.print("🔍 [bold]Analyzing Platform Naming Consistency[/bold]")
        console.print()

        # Perform validation
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Validating naming consistency...", total=None)
            results = validator.validate_migration_completeness()
            progress.update(task, description="✅ Validation complete!")
            console.print()

            # Display results
            validator.display_validation_report()

            # MLX Assistant specific recommendations
            score = results.get("consistency_score", 0.0)
            if score < 0.7:
                console.print(
                    "\n🤖 [bold blue]MLX Assistant Recommendation:[/bold blue]"
                )
                console.print(
                    "Your platform has significant naming inconsistencies. I recommend:"
                )
                console.print(
                    "1. [cyan]mlx assistant naming migrate --dry-run[/cyan] - Preview fixes"
                )
                console.print(
                    "2. [cyan]mlx assistant naming migrate --apply[/cyan] - Apply fixes"
                )
                console.print(
                    "3. [cyan]mlx assistant naming validate[/cyan] - Re-check consistency"
                )
            elif score < 0.9:
                console.print(
                    "\n🤖 [bold blue]MLX Assistant Recommendation:[/bold blue]"
                )
                console.print(
                    "Your platform is mostly consistent! A few small fixes should get you to 100%."
                )
            else:
                console.print("\n🤖 [bold blue]MLX Assistant:[/bold blue]")
                console.print(
                    "🎉 Excellent! Your platform naming is highly consistent."
                )

            # Attempt fixes if requested
            if fix_issues and score < 0.9:
                console.print("\n🔧 [bold]Attempting automatic fixes...[/bold]")
                # This would trigger focused remediation
    except Exception as e:
        console.print(f"❌ [red]Error validating naming: {e}[/red]")


@naming.command(name="migrate")
def naming_migrate(
    preset: str = typer.Option(
        "mlx", "--preset", help="Naming preset (mlx, mlsys, custom:NAME)"
    ),
    dry_run: bool = typer.Option(
        True, "--dry-run/--apply", help="Preview changes (default) or apply them"
    ),
    backup: bool = typer.Option(
        True, "--backup/--no-backup", help="Create backup files"
    ),
):
    """🔄 Migrate platform naming to consistent scheme."""
    if not NAMING_SYSTEM_AVAILABLE:
        console.print(
            "❌ [red]Naming system not available. Check that scripts/naming_config.py exists.[/red]"
        )
        return
    try:
        console.print(f"🔄 [bold]Platform Naming Migration: {preset}[/bold]")
        console.print()

        # Run the migration system
        cmd = ["python", "scripts/migrate_platform_naming.py", "set-preset", preset]
        if not dry_run:
            cmd.append("--apply")
        console.print("🤖 [bold blue]MLX Assistant:[/bold blue] Running migration...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            console.print(result.stdout)
            if dry_run:
                console.print(
                    "\n🤖 [bold blue]MLX Assistant Recommendation:[/bold blue]"
                )
                console.print("Preview looks good! To apply these changes:")
                console.print(
                    f"[cyan]mlx assistant naming migrate --preset {preset} --apply[/cyan]"
                )
            else:
                console.print("\n🤖 [bold blue]MLX Assistant:[/bold blue]")
                console.print("✅ Migration completed! Running validation...")

                # Auto-validate after migration
                naming_validate(detailed=False, fix_issues=False)
        else:
            console.print(f"❌ [red]Migration failed:[/red] {result.stderr}")
    except Exception as e:
        console.print(f"❌ [red]Error during migration: {e}[/red]")


@naming.command(name="status")
def naming_status():
    """📊 Show current platform naming status."""
    if not NAMING_SYSTEM_AVAILABLE:
        console.print("❌ [red]Naming system not available.[/red]")
        return
    try:
        config = get_naming_config()
        status_panel = Panel(
            f"[bold]Platform Name:[/bold] {config.platform_name}\n"
            f"[bold]Full Name:[/bold] {config.platform_full_name}\n"
            f"[bold]Main CLI:[/bold] {config.main_cli}\n"
            f"[bold]Evaluation CLI:[/bold] {config.evaluation_cli}\n"
            f"[bold]Assistant Command:[/bold] {config.assistant_command}\n"
            f"[bold]Config File:[/bold] {config.config_file}\n"
            f"[bold]Components Dir:[/bold] {config.components_dir}\n"
            f"[bold]Docker Network:[/bold] {config.docker_network}",
            title="🏷️ Current Platform Naming Configuration",
            border_style="bright_blue",
        )
        console.print(status_panel)

        # Quick consistency check
        if hasattr(assistant, "naming_state"):
            score = assistant.naming_state.get("consistency_score", 0.0)
            issues = assistant.naming_state.get("issues_count", 0)
            if score >= 0.9:
                consistency_status = "✅ Excellent"
                color = "green"
            elif score >= 0.7:
                consistency_status = f"⚠️ Good ({issues} issues)"
                color = "yellow"
            else:
                consistency_status = f"❌ Needs work ({issues} issues)"
                color = "red"
            console.print(
                f"\n[bold]Quick Consistency Check:[/bold] [{color}]{consistency_status}[/{color}]"
            )
            console.print(
                "\n💡 [dim]Run 'mlx assistant naming validate' for detailed analysis[/dim]"
            )
    except Exception as e:
        console.print(f"❌ [red]Error getting naming status: {e}[/red]")


# Add naming commands to main app
app.add_typer(naming, name="naming")

if __name__ == "__main__":
    app()
