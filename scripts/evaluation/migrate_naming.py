#!/usr/bin/env python3
"""
🔄 Naming Migration Script

Migrates the evaluation system to use centralized naming configuration.
Updates all hardcoded naming references to use the configurable system.

Usage:
    python scripts/evaluation/migrate_naming.py --preset mlx
    python scripts/evaluation/migrate_naming.py --preset mlsys  
    python scripts/evaluation/migrate_naming.py --preset custom:myplatform
    python scripts/evaluation/migrate_naming.py --apply --dry-run
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add the naming config to path
sys.path.insert(0, str(Path(__file__).parent.parent))  # Go up to scripts/ directory
from naming_config import NamingConfig, CommonNamingConfigs, get_naming_config

console = Console()

class NamingMigrator:
    """Handles migration of hardcoded names to configurable system"""
    
    def __init__(self):
        self.evaluation_files = [
            "scripts/evaluation/ai_response_evaluator.py",
            "scripts/evaluation/benchmark_generator.py", 
            "scripts/evaluation/analytics_dashboard.py",
            "scripts/evaluation/mlx_eval.py",
            "scripts/evaluation/setup.py",
            "scripts/evaluation/README.md"
        ]
        
        # Define replacement patterns
        self.replacement_patterns = [
            # MLX/Platform references
            (r'\bMLX AI Response Evaluation System\b', '{PLATFORM_NAME_UPPER} AI Response Evaluation System'),
            (r'\bMLX Platform Foundation\b', '{PLATFORM_FULL_NAME}'),
            (r'\bMLX platform\b', '{PLATFORM_NAME} platform'),
            (r'\bMLX Platform\b', '{PLATFORM_NAME_TITLE} Platform'),
            (r'\bMLX-specific\b', '{PLATFORM_NAME_UPPER}-specific'),
            (r'\bMLX framework\b', '{PLATFORM_NAME} framework'),
            (r'\bMLX command\b', '{PLATFORM_NAME} command'),
            (r'\bMLX project\b', '{PLATFORM_NAME} project'),
            
            # Command patterns
            (r'\bmlx assistant\b', '{ASSISTANT_COMMAND}'),
            (r'"mlx assistant', '"{ASSISTANT_COMMAND}'),
            (r'`mlx assistant', '`{ASSISTANT_COMMAND}'),
            
            # CLI references
            (r'\bmlx-eval\b', '{EVALUATION_CLI}'),
            (r'\bmlsys\b(?!\s*=)', '{MAIN_CLI}'),  # Avoid replacing in assignments
            
            # File and directory names
            (r'\bmlx\.config\.json\b', '{CONFIG_FILE}'),
            (r'\bmlx-components\b', '{COMPONENTS_DIR}'),
            (r'\.mlx\b', '{METADATA_DIR}'),
            
            # Network names
            (r'\bmlsys-network\b', '{DOCKER_NETWORK}'),
            
            # Package prefixes
            (r'\bmlx-plugin-', '{PACKAGE_PREFIX}-plugin-'),
        ]
    
    def analyze_files(self) -> Dict[str, List[Tuple[int, str, str]]]:
        """Analyze files to find naming patterns that need migration"""
        results = {}
        
        for file_path_str in self.evaluation_files:
            file_path = Path(file_path_str)
            if not file_path.exists():
                continue
                
            matches = []
            try:
                content = file_path.read_text()
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    for pattern, replacement in self.replacement_patterns:
                        if re.search(pattern, line):
                            matches.append((line_num, pattern, line.strip()))
                
                if matches:
                    results[str(file_path)] = matches
                    
            except Exception as e:
                console.print(f"[red]Error analyzing {file_path}: {e}[/red]")
        
        return results
    
    def migrate_file(self, file_path: Path, config: NamingConfig, dry_run: bool = True) -> Tuple[bool, int]:
        """Migrate a single file to use naming configuration"""
        if not file_path.exists():
            return False, 0
        
        try:
            content = file_path.read_text()
            original_content = content
            changes_made = 0
            
            # Apply replacement patterns
            for pattern, replacement_template in self.replacement_patterns:
                # Substitute the template with actual values
                replacement = self._substitute_template(replacement_template, config)
                
                # Count matches before replacement
                matches = len(re.findall(pattern, content))
                if matches > 0:
                    content = re.sub(pattern, replacement, content)
                    changes_made += matches
            
            # Special handling for specific file types
            if file_path.name == "mlx_eval.py":
                content = self._migrate_cli_file(content, config)
            elif file_path.name == "README.md":
                content = self._migrate_readme_file(content, config)
            
            if content != original_content:
                if not dry_run:
                    # Create backup
                    backup_path = file_path.with_suffix(file_path.suffix + '.backup')
                    if not backup_path.exists():
                        backup_path.write_text(original_content)
                    
                    # Write updated content
                    file_path.write_text(content)
                
                return True, changes_made
            
            return False, 0
            
        except Exception as e:
            console.print(f"[red]Error migrating {file_path}: {e}[/red]")
            return False, 0
    
    def _substitute_template(self, template: str, config: NamingConfig) -> str:
        """Substitute template placeholders with config values"""
        replacements = {
            '{PLATFORM_NAME}': config.platform_name,
            '{PLATFORM_FULL_NAME}': config.platform_full_name,
            '{PLATFORM_NAME_UPPER}': config.platform_name.upper(),
            '{PLATFORM_NAME_TITLE}': config.platform_name.title(),
            '{ASSISTANT_COMMAND}': config.assistant_command,
            '{EVALUATION_CLI}': config.evaluation_cli,
            '{MAIN_CLI}': config.main_cli,
            '{CONFIG_FILE}': config.config_file,
            '{COMPONENTS_DIR}': config.components_dir,
            '{METADATA_DIR}': config.metadata_dir,
            '{DOCKER_NETWORK}': config.docker_network,
            '{PACKAGE_PREFIX}': config.package_prefix,
        }
        
        result = template
        for placeholder, value in replacements.items():
            result = result.replace(placeholder, value)
        
        return result
    
    def _migrate_cli_file(self, content: str, config: NamingConfig) -> str:
        """Special migration for CLI file"""
        # Update CLI app name and help text
        content = re.sub(
            r'name="mlx-eval"',
            f'name="{config.evaluation_cli}"',
            content
        )
        
        content = re.sub(
            r'MLX AI Evaluation System',
            f'{config.platform_name.upper()} AI Evaluation System',
            content
        )
        
        return content
    
    def _migrate_readme_file(self, content: str, config: NamingConfig) -> str:
        """Special migration for README file"""
        # Update title and main headings
        content = re.sub(
            r'# 🎯 MLX AI Response Evaluation System',
            f'# 🎯 {config.platform_name.upper()} AI Response Evaluation System',
            content
        )
        
        # Update usage examples
        content = re.sub(
            r'python scripts/evaluation/mlx_eval\.py',
            f'python scripts/evaluation/{config.evaluation_cli.replace("-", "_")}.py',
            content
        )
        
        return content
    
    def generate_updated_files(self, config: NamingConfig):
        """Generate updated versions of key files with new naming"""
        
        # Update the main CLI file name
        old_cli_path = Path("scripts/evaluation/mlx_eval.py")
        new_cli_name = config.evaluation_cli.replace("-", "_") + ".py"
        new_cli_path = Path(f"scripts/evaluation/{new_cli_name}")
        
        if old_cli_path.exists() and old_cli_path != new_cli_path:
            console.print(f"📝 CLI file should be renamed: {old_cli_path} → {new_cli_path}")
        
        # Update setup.py to reference correct CLI name
        setup_path = Path("scripts/evaluation/setup.py")
        if setup_path.exists():
            content = setup_path.read_text()
            updated_content = content.replace("mlx-eval", config.evaluation_cli)
            updated_content = updated_content.replace("mlx_eval", config.evaluation_cli.replace("-", "_"))
            
            if content != updated_content:
                console.print(f"📝 setup.py needs updates for CLI naming")

def main():
    app = typer.Typer(help="Naming migration tool for evaluation system")
    
    @app.command("analyze")
    def analyze_naming():
        """Analyze current naming patterns in evaluation files"""
        migrator = NamingMigrator()
        results = migrator.analyze_files()
        
        if not results:
            console.print("[green]✅ No naming patterns found that need migration[/green]")
            return
        
        console.print("[yellow]📋 Found naming patterns that could be migrated:[/yellow]\n")
        
        for file_path, matches in results.items():
            console.print(f"[cyan]{file_path}[/cyan]")
            
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Line", style="dim", width=6)
            table.add_column("Pattern", style="yellow")
            table.add_column("Context", style="white")
            
            for line_num, pattern, context in matches[:10]:  # Show first 10 matches
                table.add_row(str(line_num), pattern, context[:80] + "..." if len(context) > 80 else context)
            
            console.print(table)
            
            if len(matches) > 10:
                console.print(f"[dim]... and {len(matches) - 10} more matches[/dim]")
            
            console.print()
    
    @app.command("set-preset")
    def set_naming_preset(
        preset: str = typer.Argument(..., help="Preset: mlx, mlsys, or custom:NAME"),
        apply_immediately: bool = typer.Option(False, "--apply", help="Apply migration immediately")
    ):
        """Set naming configuration preset"""
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
        
        console.print(f"✅ [green]Naming configuration set to: {preset}[/green]")
        console.print(f"💾 Configuration saved to: {config_path}")
        
        # Show configuration
        table = Table(title="Naming Configuration", show_header=True)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        config_dict = config.to_dict()
        for key, value in config_dict.items():
            table.add_row(key.replace('_', ' ').title(), str(value))
        
        console.print(table)
        
        if apply_immediately:
            console.print("\n🔄 Applying migration...")
            migrate_files(dry_run=False)
    
    @app.command("migrate")
    def migrate_files(
        dry_run: bool = typer.Option(True, "--dry-run/--apply", help="Show changes without applying them")
    ):
        """Migrate evaluation files to use naming configuration"""
        config = get_naming_config()
        
        migrator = NamingMigrator()
        
        console.print(f"🔄 [yellow]{'Analyzing' if dry_run else 'Migrating'} evaluation files...[/yellow]\n")
        
        total_files = 0
        total_changes = 0
        
        for file_path_str in migrator.evaluation_files:
            file_path = Path(file_path_str)
            if not file_path.exists():
                continue
            
            changed, changes = migrator.migrate_file(file_path, config, dry_run)
            
            if changed:
                status = "Would update" if dry_run else "Updated"
                console.print(f"{'📝' if dry_run else '✅'} {status}: [cyan]{file_path}[/cyan] ({changes} changes)")
                total_files += 1
                total_changes += changes
        
        console.print(f"\n📊 Summary:")
        console.print(f"  Files {'to update' if dry_run else 'updated'}: {total_files}")
        console.print(f"  Total changes: {total_changes}")
        
        if dry_run and total_changes > 0:
            console.print(f"\n💡 Run with [cyan]--apply[/cyan] to apply changes")
        elif not dry_run and total_changes > 0:
            console.print(f"\n✅ [green]Migration completed successfully![/green]")
            console.print(f"📁 Backup files created with .backup extension")
    
    @app.command("show-config")
    def show_current_config():
        """Show current naming configuration"""
        config = get_naming_config()
        
        console.print(Panel.fit(
            f"[bold cyan]{config.platform_full_name}[/bold cyan]\n"
            f"[dim]{config.platform_description}[/dim]",
            title="🏷️ Current Naming Configuration"
        ))
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        config_dict = config.to_dict()
        for key, value in config_dict.items():
            table.add_row(key.replace('_', ' ').title(), str(value))
        
        console.print(table)
    
    app()

if __name__ == "__main__":
    main() 