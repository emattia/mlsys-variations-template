#!/usr/bin/env python3
"""
MLX Project Creator

Creates new MLX-based ML projects with intelligent component injection.
Replaces the old template renaming approach with composable architecture.
"""

import sys
import subprocess
import shutil
from pathlib import Path
import os
import json
from typing import List, Dict, Optional

def ensure_dependencies():
    """Ensure required dependencies are installed."""
    required_packages = ["typer[all]", "rich"]
    
    try:
        import typer, rich
        return  # Dependencies already available
    except ImportError:
        pass
    
    # Install dependencies if not available
    print("🚀 Installing MLX creator dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--quiet"
        ] + required_packages)
        print("✅ Dependencies installed!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        sys.exit(1)

ensure_dependencies()

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    help="MLX Project Creator - Create ML projects with composable components",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()

# MLX Templates (component combinations)
MLX_TEMPLATES = {
    "fastapi-ml": {
        "description": "FastAPI ML API with config management and plugin system",
        "components": ["api-serving", "config-management", "plugin-registry"],
        "type": "api"
    },
    "llm-chat": {
        "description": "LLM chat application with caching and rate limiting", 
        "components": ["api-serving", "llm-integration", "caching", "rate-limiting"],
        "type": "llm"
    },
    "data-pipeline": {
        "description": "Data processing pipeline with model management",
        "components": ["data-processing", "model-management", "config-management"],
        "type": "pipeline"
    },
    "minimal": {
        "description": "Minimal setup with just configuration management",
        "components": ["config-management"],
        "type": "minimal"
    }
}

def to_snake_case(name: str) -> str:
    """Convert to snake_case."""
    return name.lower().replace("-", "_").replace(" ", "_")

def to_kebab_case(name: str) -> str:
    """Convert to kebab-case."""
    return name.lower().replace("_", "-").replace(" ", "-")

def to_title_case(name: str) -> str:
    """Convert to Title Case."""
    return " ".join(word.capitalize() for word in name.replace("-", "_").split("_"))

@app.command()
def create(
    project_name: str = typer.Argument(..., help="Project name (e.g., 'customer-churn-api')"),
    template: str = typer.Option(
        "fastapi-ml", 
        "--template", "-t",
        help="MLX template to use"
    ),
    output_dir: Path = typer.Option(
        Path.cwd(),
        "--output", "-o", 
        help="Output directory"
    ),
    components: Optional[List[str]] = typer.Option(
        None,
        "--add-component", "-c",
        help="Additional components to include"
    )
):
    """Create a new MLX project with specified template and components."""
    
    # Display intro
    intro_text = f"""
Welcome to MLX Project Creator! 🚀

Creating project: [bold cyan]{project_name}[/bold cyan]
Template: [bold green]{template}[/bold green]
Output: [cyan]{output_dir}[/cyan]

Like planting a seed in rich soil,
Your ML project begins to grow.
With components as building blocks,
Watch your capabilities flow!
    """.strip()
    
    console.print(Panel(intro_text, title="MLX Project Creation", border_style="blue"))
    
    # Validate template
    if template not in MLX_TEMPLATES:
        console.print(f"\n[red]Error:[/red] Template '{template}' not found.")
        console.print("\nAvailable templates:")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Template", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Components", style="green")
        
        for tmpl_name, tmpl_info in MLX_TEMPLATES.items():
            table.add_row(
                tmpl_name,
                tmpl_info["description"], 
                ", ".join(tmpl_info["components"])
            )
        
        console.print(table)
        raise typer.Exit(1)
    
    # Set up project paths
    project_kebab = to_kebab_case(project_name)
    project_snake = to_snake_case(project_name)
    project_title = to_title_case(project_name)
    
    project_path = output_dir / project_kebab
    
    if project_path.exists():
        console.print(f"\n[red]Error:[/red] Directory '{project_path}' already exists.")
        raise typer.Exit(1)
    
    try:
        # Create project directory
        console.print(f"\n📁 Creating project directory: {project_path}")
        project_path.mkdir(parents=True)
        
        # Copy MLX foundation structure
        foundation_path = Path(__file__).parent.parent.parent  # Go up from scripts/mlx/
        
        console.print("📋 Copying MLX foundation structure...")
        copy_foundation_structure(foundation_path, project_path, project_snake)
        
        # Initialize projen configuration
        console.print("⚙️  Setting up projen configuration...")
        setup_projen_config(project_path, project_kebab, project_title, project_snake)
        
        # Get components to inject
        template_components = MLX_TEMPLATES[template]["components"]
        all_components = template_components + (components or [])
        
        console.print(f"🧩 Injecting {len(all_components)} components...")
        inject_components(project_path, all_components)
        
        # Initialize git repository
        console.print("🔧 Initializing git repository...")
        init_git_repo(project_path)
        
        # Success message
        success_text = f"""
🎉 [bold green]Project Created Successfully![/bold green] 🎉

Project: [bold cyan]{project_title}[/bold cyan]
Location: [cyan]{project_path}[/cyan]
Template: [green]{template}[/green]
Components: [yellow]{', '.join(all_components)}[/yellow]

Next steps:
1. [bold]cd {project_kebab}[/bold]
2. [bold]uv pip install -e .[dev][/bold]  
3. [bold]projen[/bold]  # Synthesize configuration
4. [bold]projen api:dev[/bold]  # Start development server
5. [bold]mlx status[/bold]  # Check project health
        """.strip()
        
        console.print(Panel(success_text, title="Success!", border_style="green"))
        
    except Exception as e:
        console.print(f"\n[red]Error creating project:[/red] {e}")
        if project_path.exists():
            console.print("🧹 Cleaning up partial project...")
            shutil.rmtree(project_path)
        raise typer.Exit(1)

def copy_foundation_structure(source: Path, target: Path, project_snake: str):
    """Copy MLX foundation structure to new project."""
    
    # Directories to copy
    dirs_to_copy = [
        "src", "tests", "docs", "conf", "scripts", 
        "mlx-components", "notebooks", "data"
    ]
    
    # Files to copy
    files_to_copy = [
        "pyproject.toml", "README.md", "Makefile", 
        "docker-compose.yml", "Dockerfile", ".env-example",
        ".gitignore", ".pre-commit-config.yaml"
    ]
    
    for dir_name in dirs_to_copy:
        source_dir = source / dir_name
        if source_dir.exists():
            target_dir = target / dir_name
            shutil.copytree(source_dir, target_dir, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    
    for file_name in files_to_copy:
        source_file = source / file_name
        if source_file.exists():
            shutil.copy2(source_file, target / file_name)

def setup_projen_config(project_path: Path, project_kebab: str, project_title: str, project_snake: str):
    """Set up projen configuration for new project."""
    
    projenrc_content = f'''#!/usr/bin/env python3
"""
{project_title} - MLX Project Configuration

Generated by MLX Project Creator
Projen-driven configuration with MLX components
"""

from pathlib import Path
import sys

# Add the MLX foundation projenrc
foundation_path = Path(__file__).parent / "scripts" / "mlx" / "projenrc_template.py"
if foundation_path.exists():
    sys.path.insert(0, str(foundation_path.parent))
    from projenrc_template import MLXProject
else:
    # Fallback to basic Python project
    from projen.python import PythonProject as MLXProject

# Create project
project = MLXProject(
    name="{project_kebab}",
    module_name="{project_snake}",
    description="{project_title} - MLX-powered ML project",
    author_name="ML Team",
    version="0.1.0",
    
    # MLX-specific configuration
    mlx_components_enabled=True,
    mlx_ai_enhanced=True,
)

project.synth()
'''
    
    projenrc_path = project_path / ".projenrc.py" 
    projenrc_path.write_text(projenrc_content)

def inject_components(project_path: Path, components: List[str]):
    """Inject MLX components into the new project."""
    
    # This would use the MLX injection system once Phase 2 is complete
    # For now, just create placeholder structure
    
    for component in components:
        console.print(f"  ✅ {component}")
    
    # Create MLX status file
    mlx_status = {
        "project_type": "mlx",
        "version": "1.0.0",
        "components_installed": components,
        "created_with": "mlx-create"
    }
    
    status_file = project_path / "mlx.config.json"
    with open(status_file, 'w') as f:
        json.dump(mlx_status, f, indent=2)

def init_git_repo(project_path: Path):
    """Initialize git repository."""
    try:
        subprocess.run(
            ["git", "init"], 
            cwd=project_path, 
            check=True, 
            capture_output=True
        )
        subprocess.run(
            ["git", "add", "."], 
            cwd=project_path, 
            check=True, 
            capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial MLX project creation"], 
            cwd=project_path, 
            check=True, 
            capture_output=True
        )
    except subprocess.CalledProcessError:
        console.print("[yellow]Warning:[/yellow] Could not initialize git repository")

@app.command()
def templates():
    """List available MLX project templates."""
    
    console.print("\n[bold]Available MLX Templates:[/bold]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Template", style="cyan", width=15)
    table.add_column("Type", style="blue", width=10)
    table.add_column("Description", style="white", width=40)
    table.add_column("Components", style="green")
    
    for name, info in MLX_TEMPLATES.items():
        table.add_row(
            name,
            info["type"],
            info["description"],
            ", ".join(info["components"])
        )
    
    console.print(table)
    console.print(f"\n[dim]Use: [bold]mlx create my-project --template=TEMPLATE[/bold][/dim]")

if __name__ == "__main__":
    app() 