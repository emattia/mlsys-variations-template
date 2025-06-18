#!/usr/bin/env python3
"""
Setup script for MLX AI Evaluation System

Installs dependencies and configures the evaluation system.
"""

import subprocess
import sys
from pathlib import Path

def install_dependencies():
    """Install required Python packages"""
    dependencies = [
        "rich>=13.0.0",
        "typer>=0.9.0", 
        "pandas>=1.5.0",
        "numpy>=1.24.0"
    ]
    
    print("🔧 Installing dependencies...")
    for dep in dependencies:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"  ✅ {dep}")
        except subprocess.CalledProcessError:
            print(f"  ❌ Failed to install {dep}")
            return False
    
    return True

def setup_directories():
    """Create necessary directories"""
    directories = [
        "data/evaluations",
        "data/benchmarks", 
        "data/benchmark_results",
        "reports"
    ]
    
    print("📁 Setting up directories...")
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {dir_path}")

def create_alias_script():
    """Create a simple alias script for easy access"""
    script_content = """#!/usr/bin/env python3
import sys
from pathlib import Path

# Add the evaluation system to Python path
eval_dir = Path(__file__).parent
sys.path.insert(0, str(eval_dir))

from mlx_eval import app

if __name__ == "__main__":
    app()
"""
    
    alias_path = Path("scripts/evaluation/mlx-eval")
    with open(alias_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    alias_path.chmod(0o755)
    print(f"✅ Created executable: {alias_path}")

def main():
    """Main setup function"""
    print("🎯 MLX AI Evaluation System Setup\n")
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        return 1
    
    # Setup directories
    setup_directories()
    
    # Create alias script
    create_alias_script()
    
    print("\n✅ Setup complete!")
    print("\n📋 Usage examples:")
    print("  python scripts/evaluation/mlx_eval.py run --query 'How do I set up security scanning?'")
    print("  python scripts/evaluation/mlx_eval.py dashboard")
    print("  python scripts/evaluation/mlx_eval.py benchmark --category security")
    print("  python scripts/evaluation/mlx_eval.py status")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 