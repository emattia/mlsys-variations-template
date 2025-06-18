#!/usr/bin/env python3
"""
🏷️ Centralized Naming Configuration

Defines all naming patterns used throughout the MLX/MLSys platform.
Change these values to uniformly update naming across all components.

This allows easy rebranding/renaming of the entire platform by modifying
a single configuration file.
"""

from dataclasses import dataclass
from typing import Dict, Any
import json
from pathlib import Path

@dataclass
class NamingConfig:
    """Centralized naming configuration for the entire platform"""
    
    # Core Platform Names
    platform_name: str = "mlx"              # Core platform identifier (mlx, mlsys, etc.)
    platform_full_name: str = "MLX Platform Foundation"  # Full descriptive name
    platform_description: str = "Production-grade ML platform component system"
    
    # CLI and Script Names
    main_cli: str = "mlx"                    # Main executable script name
    evaluation_cli: str = "mlx-eval"         # Evaluation system CLI name
    
    # Package and Module Names
    package_prefix: str = "mlx"              # Package prefix (mlx-plugin-*, mlx-components, etc.)
    module_namespace: str = "mlx"            # Python module namespace
    
    # Directory and File Names
    config_file: str = "mlx.config.json"    # Main configuration file
    components_dir: str = "mlx-components"   # Components directory
    metadata_dir: str = ".mlx"              # Metadata directory
    
    # Command Patterns
    assistant_command: str = "mlx assistant" # Base assistant command pattern
    
    # Network and Service Names
    docker_network: str = "mlx-network"     # Docker network name
    
    # Repository and Project Names
    template_name: str = "mlx-variations-template"  # Template repository name
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "platform_name": self.platform_name,
            "platform_full_name": self.platform_full_name,
            "platform_description": self.platform_description,
            "main_cli": self.main_cli,
            "evaluation_cli": self.evaluation_cli,
            "package_prefix": self.package_prefix,
            "module_namespace": self.module_namespace,
            "config_file": self.config_file,
            "components_dir": self.components_dir,
            "metadata_dir": self.metadata_dir,
            "assistant_command": self.assistant_command,
            "docker_network": self.docker_network,
            "template_name": self.template_name
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NamingConfig':
        """Create from dictionary"""
        return cls(**data)
    
    def save_to_file(self, path: Path):
        """Save naming configuration to file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, path: Path) -> 'NamingConfig':
        """Load naming configuration from file"""
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                return cls.from_dict(data)
        return cls()  # Return default if file doesn't exist

# Global naming configuration instance
_naming_config = None

def get_naming_config() -> NamingConfig:
    """Get the global naming configuration"""
    global _naming_config
    if _naming_config is None:
        config_path = Path("naming.config.json")
        _naming_config = NamingConfig.load_from_file(config_path)
    return _naming_config

def set_naming_config(config: NamingConfig):
    """Set the global naming configuration"""
    global _naming_config
    _naming_config = config

# Convenience functions for common naming patterns
def get_platform_name() -> str:
    """Get the platform name (e.g., 'mlx', 'mlsys')"""
    return get_naming_config().platform_name

def get_platform_full_name() -> str:
    """Get the full platform name (e.g., 'MLX Platform Foundation')"""
    return get_naming_config().platform_full_name

def get_main_cli_name() -> str:
    """Get the main CLI name (e.g., 'mlsys')"""
    return get_naming_config().main_cli

def get_evaluation_cli_name() -> str:
    """Get the evaluation CLI name (e.g., 'mlx-eval')"""
    return get_naming_config().evaluation_cli

def get_assistant_command() -> str:
    """Get the assistant command pattern (e.g., 'mlx assistant')"""
    return get_naming_config().assistant_command

def get_package_prefix() -> str:
    """Get the package prefix (e.g., 'mlx')"""
    return get_naming_config().package_prefix

def get_config_file_name() -> str:
    """Get the main config file name (e.g., 'mlx.config.json')"""
    return get_naming_config().config_file

def get_components_dir_name() -> str:
    """Get the components directory name (e.g., 'mlx-components')"""
    return get_naming_config().components_dir

# Template substitution functions
def substitute_naming_in_text(text: str, config: NamingConfig = None) -> str:
    """Replace naming placeholders in text with actual values"""
    if config is None:
        config = get_naming_config()
    
    replacements = {
        "{PLATFORM_NAME}": config.platform_name,
        "{PLATFORM_FULL_NAME}": config.platform_full_name,
        "{PLATFORM_DESCRIPTION}": config.platform_description,
        "{MAIN_CLI}": config.main_cli,
        "{EVALUATION_CLI}": config.evaluation_cli,
        "{PACKAGE_PREFIX}": config.package_prefix,
        "{MODULE_NAMESPACE}": config.module_namespace,
        "{CONFIG_FILE}": config.config_file,
        "{COMPONENTS_DIR}": config.components_dir,
        "{METADATA_DIR}": config.metadata_dir,
        "{ASSISTANT_COMMAND}": config.assistant_command,
        "{DOCKER_NETWORK}": config.docker_network,
        "{TEMPLATE_NAME}": config.template_name,
        
        # Common variations
        "{PLATFORM_NAME_UPPER}": config.platform_name.upper(),
        "{PLATFORM_NAME_TITLE}": config.platform_name.title(),
        "{MAIN_CLI_UPPER}": config.main_cli.upper(),
        "{EVALUATION_CLI_UPPER}": config.evaluation_cli.upper(),
    }
    
    result = text
    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value)
    
    return result

def substitute_naming_in_file(file_path: Path, config: NamingConfig = None, backup: bool = True) -> bool:
    """Replace naming placeholders in a file with actual values"""
    try:
        if backup:
            backup_path = file_path.with_suffix(file_path.suffix + '.backup')
            if not backup_path.exists():
                backup_path.write_text(file_path.read_text())
        
        content = file_path.read_text()
        updated_content = substitute_naming_in_text(content, config)
        
        if content != updated_content:
            file_path.write_text(updated_content)
            return True
        return False
        
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

# Predefined naming configurations for common use cases
class CommonNamingConfigs:
    """Common naming configuration presets"""
    
    @staticmethod
    def mlx_platform() -> NamingConfig:
        """MLX Platform Foundation naming"""
        return NamingConfig(
            platform_name="mlx",
            platform_full_name="MLX Platform Foundation",
            platform_description="Production-grade ML platform component system",
            main_cli="mlx",
            evaluation_cli="mlx-eval",
            package_prefix="mlx",
            module_namespace="mlx",
            config_file="mlx.config.json",
            components_dir="mlx-components",
            metadata_dir=".mlx",
            assistant_command="mlx assistant",
            docker_network="mlx-network",
            template_name="mlx-variations-template"
        )
    
    @staticmethod
    def mlsys_platform() -> NamingConfig:
        """MLSys Platform naming"""
        return NamingConfig(
            platform_name="mlsys",
            platform_full_name="MLSys Platform Foundation",
            platform_description="Production-grade ML systems platform",
            main_cli="mlsys",
            evaluation_cli="mlsys-eval",
            package_prefix="mlsys",
            module_namespace="mlsys",
            config_file="mlsys.config.json",
            components_dir="mlsys-components",
            metadata_dir=".mlsys",
            assistant_command="mlsys assistant",
            docker_network="mlsys-network",
            template_name="mlsys-platform-template"
        )
    
    @staticmethod
    def custom_platform(name: str) -> NamingConfig:
        """Create custom platform naming"""
        return NamingConfig(
            platform_name=name.lower(),
            platform_full_name=f"{name.title()} Platform Foundation",
            platform_description=f"Production-grade {name.lower()} platform component system",
            main_cli=name.lower(),
            evaluation_cli=f"{name.lower()}-eval",
            package_prefix=name.lower(),
            module_namespace=name.lower(),
            config_file=f"{name.lower()}.config.json",
            components_dir=f"{name.lower()}-components",
            metadata_dir=f".{name.lower()}",
            assistant_command=f"{name.lower()} assistant",
            docker_network=f"{name.lower()}-network",
            template_name=f"{name.lower()}-platform-template"
        )

if __name__ == "__main__":
    # Example usage and testing
    import typer
    
    app = typer.Typer(help="Platform naming configuration management")
    
    @app.command("show")
    def show_config():
        """Show current naming configuration"""
        config = get_naming_config()
        print("Current Platform Naming Configuration:")
        print(json.dumps(config.to_dict(), indent=2))
    
    @app.command("set-preset")
    def set_preset(preset: str = typer.Argument(..., help="Preset name: mlx, mlsys, or custom:NAME")):
        """Set a predefined naming configuration"""
        if preset == "mlx":
            config = CommonNamingConfigs.mlx_platform()
        elif preset == "mlsys":
            config = CommonNamingConfigs.mlsys_platform()
        elif preset.startswith("custom:"):
            name = preset.split(":", 1)[1]
            config = CommonNamingConfigs.custom_platform(name)
        else:
            print(f"Unknown preset: {preset}")
            raise typer.Exit(1)
        
        config.save_to_file(Path("naming.config.json"))
        print(f"✅ Platform naming configuration set to: {preset}")
    
    @app.command("apply")
    def apply_to_files(
        pattern: str = typer.Option("**/*.py", help="File pattern to update"),
        dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be changed without applying")
    ):
        """Apply naming configuration to files"""
        from pathlib import Path
        import glob
        
        config = get_naming_config()
        files_updated = 0
        
        for file_path in Path(".").glob(pattern):
            if file_path.is_file():
                if dry_run:
                    content = file_path.read_text()
                    updated = substitute_naming_in_text(content, config)
                    if content != updated:
                        print(f"Would update: {file_path}")
                        files_updated += 1
                else:
                    if substitute_naming_in_file(file_path, config):
                        print(f"Updated: {file_path}")
                        files_updated += 1
        
        if dry_run:
            print(f"Would update {files_updated} files")
        else:
            print(f"Updated {files_updated} files")
    
    app() 