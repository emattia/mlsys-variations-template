"""
MLX Component Injection System

Handles intelligent injection of extracted components into existing MLX projects
with proper conflict resolution, template variable substitution, and dependency management.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import sys


class ComponentInjector:
    """Handles intelligent component injection with conflict resolution."""

    def __init__(self, project_root: Path, components_dir: Path):
        self.project_root = project_root
        self.components_dir = components_dir
        self.registry = self._load_registry()
        self.project_config = self._load_project_config()

    def add_component(self, component_name: str, force: bool = False) -> bool:
        """Add a component to the current project."""
        if component_name not in self.registry["components"]:
            print(f"❌ Component '{component_name}' not found in registry")
            return False

        component_meta = self.registry["components"][component_name]
        component_dir = self.components_dir / component_name

        if not component_dir.exists():
            print(f"❌ Component directory not found: {component_dir}")
            return False

        print(f"🧩 Adding component: {component_name}")

        # Check dependencies
        if not self._check_dependencies(component_meta, force):
            return False

        # Check for conflicts
        if not force and not self._check_conflicts(component_meta):
            return False

        # Install system dependencies
        if not self._install_system_dependencies(component_meta):
            print("❌ Failed to install system dependencies")
            return False

        # Install Python dependencies
        if not self._install_python_dependencies(component_meta):
            print("❌ Failed to install Python dependencies")
            return False

        # Inject component files
        if not self._inject_component_files(component_name, component_meta):
            print("❌ Failed to inject component files")
            return False

        # Update project configuration
        self._update_project_config(component_name, component_meta)

        print(f"✅ Component '{component_name}' added successfully!")
        return True

    def list_available_components(self) -> List[Dict]:
        """List all available components with their metadata."""
        components = []
        for name, meta in self.registry["components"].items():
            components.append(
                {
                    "name": name,
                    "description": meta.get("description", ""),
                    "type": meta.get("component_type", ""),
                    "version": meta.get("version", ""),
                    "dependencies": len(meta.get("python_dependencies", [])),
                    "files": len(meta.get("source_files", [])),
                }
            )
        return components

    def get_component_info(self, component_name: str) -> Optional[Dict]:
        """Get detailed information about a specific component."""
        if component_name not in self.registry["components"]:
            return None

        return self.registry["components"][component_name]

    def _load_registry(self) -> Dict:
        """Load the component registry."""
        registry_path = self.components_dir / "registry.json"
        if not registry_path.exists():
            return {"components": {}}

        try:
            with open(registry_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Warning: Could not load registry: {e}")
            return {"components": {}}

    def _load_project_config(self) -> Dict:
        """Load the project configuration."""
        config_path = self.project_root / "mlx.config.json"
        if not config_path.exists():
            return {"platform": {"components": []}}

        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Warning: Could not load project config: {e}")
            return {"platform": {"components": []}}

    def _check_dependencies(self, component_meta: Dict, force: bool) -> bool:
        """Check if component dependencies are satisfied."""
        compatibility = component_meta.get("compatibility_matrix", {})
        requires = compatibility.get("requires", [])

        if not requires:
            return True

        installed_components = self.project_config.get("platform", {}).get(
            "components", []
        )

        missing_deps = [dep for dep in requires if dep not in installed_components]

        if missing_deps:
            print(f"❌ Missing required dependencies: {', '.join(missing_deps)}")
            if not force:
                print("   Use --force to override dependency check")
                return False
            else:
                print("   ⚠️  Proceeding with --force (may cause issues)")

        return True

    def _check_conflicts(self, component_meta: Dict) -> bool:
        """Check for component conflicts."""
        conflicts = component_meta.get("conflicts_with", [])
        if not conflicts:
            return True

        installed_components = self.project_config.get("platform", {}).get(
            "components", []
        )

        conflicting = [comp for comp in conflicts if comp in installed_components]

        if conflicting:
            print(f"❌ Conflicts with installed components: {', '.join(conflicting)}")
            print("   Remove conflicting components first or use --force")
            return False

        return True

    def _install_system_dependencies(self, component_meta: Dict) -> bool:
        """Install system dependencies (Docker services, etc.)."""
        system_deps = component_meta.get("system_dependencies", [])

        if not system_deps:
            return True

        print("📦 Installing system dependencies...")

        # Check if docker-compose is available
        try:
            subprocess.run(
                ["docker-compose", "--version"], check=True, capture_output=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  Docker Compose not available, skipping system dependencies")
            return True

        # Update docker-compose.yml if needed
        compose_path = self.project_root / "docker-compose.yml"
        if compose_path.exists():
            self._update_docker_compose(system_deps)

        return True

    def _install_python_dependencies(self, component_meta: Dict) -> bool:
        """Install Python dependencies using uv."""
        python_deps = component_meta.get("python_dependencies", [])

        if not python_deps:
            return True

        print(f"📦 Installing {len(python_deps)} Python dependencies...")

        # Check if uv is available
        try:
            subprocess.run(["uv", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  uv not available, falling back to pip")
            # Fall back to pip
            try:
                for dep in python_deps:
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", dep], check=True
                    )
                return True
            except subprocess.CalledProcessError:
                return False

        # Use uv to install dependencies
        try:
            for dep in python_deps:
                subprocess.run(["uv", "add", dep], check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def _inject_component_files(
        self, component_name: str, component_meta: Dict
    ) -> bool:
        """Inject component files into the project."""
        component_dir = self.components_dir / component_name
        template_dir = component_dir / "templates"

        if not template_dir.exists():
            print(f"⚠️  No templates found for {component_name}")
            return True

        # Get template variables
        template_vars = self._get_template_variables(component_meta)

        # Process each template file
        for template_file in template_dir.glob("*.template"):
            target_path = self._determine_target_path(template_file, component_meta)
            if not target_path:
                continue

            # Read template content
            try:
                with open(template_file, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                print(f"⚠️  Could not read template {template_file}: {e}")
                continue

            # Substitute template variables
            final_content = self._substitute_template_variables(content, template_vars)

            # Determine merge strategy
            merge_strategy = self._get_merge_strategy(str(target_path), component_meta)

            # Apply merge strategy
            if not self._apply_merge_strategy(
                target_path, final_content, merge_strategy
            ):
                print(f"❌ Failed to apply {merge_strategy} strategy to {target_path}")
                return False

        return True

    def _get_template_variables(self, component_meta: Dict) -> Dict[str, str]:
        """Get template variables for substitution."""
        # Get project name from directory or config
        project_name = self.project_root.name
        if (
            "platform" in self.project_config
            and "name" in self.project_config["platform"]
        ):
            project_name = self.project_config["platform"]["name"]

        return {
            "PROJECT_NAME": project_name,
            "API_HOST": "localhost",
            "API_PORT": "8000",
            "ENVIRONMENT": "development",
        }

    def _determine_target_path(
        self, template_file: Path, component_meta: Dict
    ) -> Optional[Path]:
        """Determine where to place the template file in the project."""
        # Remove .template extension
        filename = template_file.name.replace(".template", "")

        # Map to target location based on component structure
        if "api" in str(template_file) or filename in ["app.py", "middleware.py"]:
            return self.project_root / "src" / "api" / filename
        elif "config" in str(template_file) or filename in ["manager.py", "models.py"]:
            return self.project_root / "src" / "config" / filename
        elif "plugin" in str(template_file):
            return self.project_root / "src" / "plugins" / filename
        elif filename.endswith(".yaml"):
            return self.project_root / "conf" / filename
        else:
            # Default to src directory
            return self.project_root / "src" / filename

    def _substitute_template_variables(
        self, content: str, variables: Dict[str, str]
    ) -> str:
        """Substitute template variables in content."""
        for var_name, var_value in variables.items():
            content = content.replace(f"{{{{{var_name}}}}}", var_value)
        return content

    def _get_merge_strategy(self, target_path: str, component_meta: Dict) -> str:
        """Get the merge strategy for a specific file."""
        merge_strategies = component_meta.get("merge_strategies", {})

        # Check for exact match
        for pattern, strategy in merge_strategies.items():
            if pattern in target_path:
                return strategy

        # Default strategies based on file type
        if target_path.endswith(".yaml"):
            return "merge"
        elif "development" in target_path:
            return "merge"
        elif "production" in target_path:
            return "enhance"
        else:
            return "replace"

    def _apply_merge_strategy(
        self, target_path: Path, content: str, strategy: str
    ) -> bool:
        """Apply the specified merge strategy."""
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if strategy == "replace" or not target_path.exists():
            # Replace or create new file
            try:
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(content)
                return True
            except Exception as e:
                print(f"⚠️  Could not write to {target_path}: {e}")
                return False

        elif strategy == "merge":
            # Merge YAML files
            if target_path.suffix in [".yaml", ".yml"]:
                return self._merge_yaml_files(target_path, content)
            else:
                # For non-YAML files, append content
                return self._append_to_file(target_path, content)

        elif strategy == "enhance":
            # Enhance existing file with new content
            return self._enhance_file(target_path, content)

        else:
            print(f"⚠️  Unknown merge strategy: {strategy}")
            return False

    def _merge_yaml_files(self, target_path: Path, new_content: str) -> bool:
        """Merge YAML content intelligently."""
        try:
            # Load existing YAML
            with open(target_path, "r", encoding="utf-8") as f:
                existing_data = yaml.safe_load(f) or {}

            # Parse new YAML
            new_data = yaml.safe_load(new_content) or {}

            # Deep merge
            merged_data = self._deep_merge_dict(existing_data, new_data)

            # Write back
            with open(target_path, "w", encoding="utf-8") as f:
                yaml.dump(merged_data, f, default_flow_style=False, indent=2)

            return True
        except Exception as e:
            print(f"⚠️  Could not merge YAML content: {e}")
            return False

    def _deep_merge_dict(self, base: Dict, update: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in update.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge_dict(result[key], value)
            else:
                result[key] = value

        return result

    def _append_to_file(self, target_path: Path, content: str) -> bool:
        """Append content to file."""
        try:
            with open(target_path, "a", encoding="utf-8") as f:
                f.write(f"\n# Added by MLX component injection\n{content}\n")
            return True
        except Exception as e:
            print(f"⚠️  Could not append to {target_path}: {e}")
            return False

    def _enhance_file(self, target_path: Path, new_content: str) -> bool:
        """Enhance existing file with new content."""
        try:
            # Read existing content
            with open(target_path, "r", encoding="utf-8") as f:
                existing_content = f.read()

            # Look for enhancement points (comments like # MLX_INJECT_POINT)
            if "# MLX_INJECT_POINT" in existing_content:
                enhanced_content = existing_content.replace(
                    "# MLX_INJECT_POINT", new_content
                )
            else:
                # Append to end
                enhanced_content = (
                    existing_content + f"\n\n# Enhanced by MLX\n{new_content}\n"
                )

            # Write back
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(enhanced_content)

            return True
        except Exception as e:
            print(f"⚠️  Could not enhance {target_path}: {e}")
            return False

    def _update_docker_compose(self, system_deps: List[Dict]):
        """Update docker-compose.yml with system dependencies."""
        compose_path = self.project_root / "docker-compose.yml"

        try:
            with open(compose_path, "r") as f:
                compose_data = yaml.safe_load(f)
        except Exception:
            compose_data = {"version": "3.8", "services": {}}

        # Add services for system dependencies
        for dep in system_deps:
            if dep.get("docker_image"):
                service_name = dep["name"]
                compose_data["services"][service_name] = {
                    "image": dep["docker_image"],
                    "ports": self._get_service_ports(dep),
                    "environment": self._get_service_environment(dep),
                }

        # Write back
        try:
            with open(compose_path, "w") as f:
                yaml.dump(compose_data, f, default_flow_style=False, indent=2)
        except Exception as e:
            print(f"⚠️  Could not update docker-compose.yml: {e}")

    def _get_service_ports(self, dep: Dict) -> List[str]:
        """Get default ports for a service."""
        port_mapping = {
            "redis": ["6379:6379"],
            "postgresql": ["5432:5432"],
            "mongodb": ["27017:27017"],
        }
        return port_mapping.get(dep["name"], [])

    def _get_service_environment(self, dep: Dict) -> Dict[str, str]:
        """Get default environment variables for a service."""
        env_mapping = {
            "postgresql": {
                "POSTGRES_DB": "mlx_db",
                "POSTGRES_USER": "mlx_user",
                "POSTGRES_PASSWORD": "mlx_password",
            },
            "mongodb": {
                "MONGO_INITDB_ROOT_USERNAME": "mlx_user",
                "MONGO_INITDB_ROOT_PASSWORD": "mlx_password",
            },
        }
        return env_mapping.get(dep["name"], {})

    def _update_project_config(self, component_name: str, component_meta: Dict):
        """Update project configuration with new component."""
        if "platform" not in self.project_config:
            self.project_config["platform"] = {}

        if "components" not in self.project_config["platform"]:
            self.project_config["platform"]["components"] = []

        # Add component if not already present
        if component_name not in self.project_config["platform"]["components"]:
            self.project_config["platform"]["components"].append(component_name)

        # Save config
        config_path = self.project_root / "mlx.config.json"
        try:
            with open(config_path, "w") as f:
                json.dump(self.project_config, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not update project config: {e}")


def main():
    """Main function for testing component injection."""
    project_root = Path(".")
    components_dir = Path("mlx-components")

    injector = ComponentInjector(project_root, components_dir)

    print("📋 Available Components:")
    components = injector.list_available_components()
    for comp in components:
        print(f"  📦 {comp['name']}: {comp['description']}")

    # Example usage
    if len(sys.argv) > 1:
        component_name = sys.argv[1]
        force = "--force" in sys.argv

        if injector.add_component(component_name, force):
            print(f"✅ Successfully added {component_name}")
        else:
            print(f"❌ Failed to add {component_name}")


if __name__ == "__main__":
    main()
