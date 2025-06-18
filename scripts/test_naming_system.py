#!/usr/bin/env python3
"""
🧪 Comprehensive Naming System Test Runner

Tests the entire naming migration system including:
- Core naming configuration functionality
- Migration scripts
- Template substitution
- Error handling
- Integration tests
"""

import sys
import subprocess
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress
import tempfile
import json

console = Console()

def run_test_suite():
    """Run the complete test suite for the naming system"""
    
    console.print(Panel.fit(
        "[bold cyan]🧪 Naming Migration System Test Suite[/bold cyan]\n"
        "[dim]Testing platform-wide naming configuration and migration[/dim]",
        title="Test Runner"
    ))
    
    test_results = []
    
    # Test 1: Basic naming configuration
    console.print("\n[yellow]1. Testing basic naming configuration...[/yellow]")
    try:
        from naming_config import NamingConfig, CommonNamingConfigs
        
        # Test default config
        config = NamingConfig()
        assert config.platform_name == "mlx"
        assert config.platform_full_name == "MLX Platform Foundation"
        
        # Test presets
        mlx_config = CommonNamingConfigs.mlx_platform()
        mlsys_config = CommonNamingConfigs.mlsys_platform()
        custom_config = CommonNamingConfigs.custom_platform("test")
        
        assert mlx_config.platform_name == "mlx"
        assert mlsys_config.platform_name == "mlsys"
        assert custom_config.platform_name == "test"
        
        console.print("✅ [green]Basic naming configuration: PASSED[/green]")
        test_results.append(("Basic Configuration", "PASSED", None))
        
    except Exception as e:
        console.print(f"❌ [red]Basic naming configuration: FAILED - {e}[/red]")
        test_results.append(("Basic Configuration", "FAILED", str(e)))
    
    # Test 2: Template substitution
    console.print("\n[yellow]2. Testing template substitution...[/yellow]")
    try:
        from naming_config import substitute_naming_in_text, NamingConfig
        
        config = NamingConfig(
            platform_name="testplatform",
            platform_full_name="Test Platform Foundation"
        )
        
        template = "Welcome to {PLATFORM_FULL_NAME} ({PLATFORM_NAME})"
        result = substitute_naming_in_text(template, config)
        
        assert "Test Platform Foundation" in result
        assert "testplatform" in result
        assert "{" not in result  # All templates should be substituted
        
        console.print("✅ [green]Template substitution: PASSED[/green]")
        test_results.append(("Template Substitution", "PASSED", None))
        
    except Exception as e:
        console.print(f"❌ [red]Template substitution: FAILED - {e}[/red]")
        test_results.append(("Template Substitution", "FAILED", str(e)))
    
    # Test 3: File operations
    console.print("\n[yellow]3. Testing file operations...[/yellow]")
    try:
        from naming_config import NamingConfig
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"
            
            # Create and save config
            test_config = NamingConfig(
                platform_name="filetest",
                platform_full_name="File Test Platform"
            )
            test_config.save_to_file(config_path)
            
            # Load config
            loaded_config = NamingConfig.load_from_file(config_path)
            assert loaded_config.platform_name == "filetest"
            assert loaded_config.platform_full_name == "File Test Platform"
        
        console.print("✅ [green]File operations: PASSED[/green]")
        test_results.append(("File Operations", "PASSED", None))
        
    except Exception as e:
        console.print(f"❌ [red]File operations: FAILED - {e}[/red]")
        test_results.append(("File Operations", "FAILED", str(e)))
    
    # Test 4: Platform migration script
    console.print("\n[yellow]4. Testing platform migration script...[/yellow]")
    try:
        from migrate_platform_naming import PlatformNamingMigrator
        
        migrator = PlatformNamingMigrator()
        
        # Test file discovery
        files = migrator.discover_files()
        assert isinstance(files, list)
        
        # Test replacement patterns
        assert hasattr(migrator, 'replacement_patterns')
        assert len(migrator.replacement_patterns) > 0
        
        # Test template substitution
        config = NamingConfig(platform_name="test")
        template = "{PLATFORM_NAME} test"
        result = migrator._substitute_template(template, config)
        assert "test test" in result
        
        console.print("✅ [green]Platform migration script: PASSED[/green]")
        test_results.append(("Platform Migration Script", "PASSED", None))
        
    except Exception as e:
        console.print(f"❌ [red]Platform migration script: FAILED - {e}[/red]")
        test_results.append(("Platform Migration Script", "FAILED", str(e)))
    
    # Test 5: Evaluation migration script
    console.print("\n[yellow]5. Testing evaluation migration script...[/yellow]")
    try:
        sys.path.insert(0, str(Path(__file__).parent / "evaluation"))
        from migrate_naming import NamingMigrator
        
        migrator = NamingMigrator()
        
        # Test evaluation files list
        assert hasattr(migrator, 'evaluation_files')
        assert len(migrator.evaluation_files) > 0
        
        # Test replacement patterns
        assert hasattr(migrator, 'replacement_patterns')
        assert len(migrator.replacement_patterns) > 0
        
        console.print("✅ [green]Evaluation migration script: PASSED[/green]")
        test_results.append(("Evaluation Migration Script", "PASSED", None))
        
    except Exception as e:
        console.print(f"❌ [red]Evaluation migration script: FAILED - {e}[/red]")
        test_results.append(("Evaluation Migration Script", "FAILED", str(e)))
    
    # Test 6: CLI functionality
    console.print("\n[yellow]6. Testing CLI functionality...[/yellow]")
    try:
        # Test platform migration CLI
        result = subprocess.run([
            sys.executable, "scripts/migrate_platform_naming.py", "show-config"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            console.print("✅ [green]Platform CLI functionality: PASSED[/green]")
            test_results.append(("Platform CLI", "PASSED", None))
        else:
            console.print(f"❌ [red]Platform CLI functionality: FAILED - {result.stderr}[/red]")
            test_results.append(("Platform CLI", "FAILED", result.stderr))
        
        # Test evaluation migration CLI
        result = subprocess.run([
            sys.executable, "scripts/evaluation/migrate_naming.py", "show-config"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            console.print("✅ [green]Evaluation CLI functionality: PASSED[/green]")
            test_results.append(("Evaluation CLI", "PASSED", None))
        else:
            console.print(f"❌ [red]Evaluation CLI functionality: FAILED - {result.stderr}[/red]")
            test_results.append(("Evaluation CLI", "FAILED", result.stderr))
            
    except Exception as e:
        console.print(f"❌ [red]CLI functionality: FAILED - {e}[/red]")
        test_results.append(("CLI Functionality", "FAILED", str(e)))
    
    # Test 7: Integration test - dry run migration
    console.print("\n[yellow]7. Testing dry-run migration...[/yellow]")
    try:
        result = subprocess.run([
            sys.executable, "scripts/migrate_platform_naming.py", "migrate", "--dry-run"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            console.print("✅ [green]Dry-run migration: PASSED[/green]")
            test_results.append(("Dry-run Migration", "PASSED", None))
        else:
            console.print(f"❌ [red]Dry-run migration: FAILED - {result.stderr}[/red]")
            test_results.append(("Dry-run Migration", "FAILED", result.stderr))
            
    except Exception as e:
        console.print(f"❌ [red]Dry-run migration: FAILED - {e}[/red]")
        test_results.append(("Dry-run Migration", "FAILED", str(e)))
    
    # Test 8: Analyze current naming patterns
    console.print("\n[yellow]8. Testing naming pattern analysis...[/yellow]")
    try:
        result = subprocess.run([
            sys.executable, "scripts/migrate_platform_naming.py", "analyze"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            console.print("✅ [green]Naming pattern analysis: PASSED[/green]")
            test_results.append(("Pattern Analysis", "PASSED", None))
        else:
            console.print(f"❌ [red]Naming pattern analysis: FAILED - {result.stderr}[/red]")
            test_results.append(("Pattern Analysis", "FAILED", result.stderr))
            
    except Exception as e:
        console.print(f"❌ [red]Naming pattern analysis: FAILED - {e}[/red]")
        test_results.append(("Pattern Analysis", "FAILED", str(e)))
    
    # Display test results summary
    console.print("\n" + "="*60)
    console.print(Panel.fit(
        "[bold cyan]🎯 Test Results Summary[/bold cyan]",
        title="Results"
    ))
    
    table = Table(title="Test Results", show_header=True, header_style="bold magenta")
    table.add_column("Test", style="cyan", width=25)
    table.add_column("Status", style="white", width=10)
    table.add_column("Error", style="red", max_width=40)
    
    passed_count = 0
    failed_count = 0
    
    for test_name, status, error in test_results:
        if status == "PASSED":
            status_display = "[green]✅ PASSED[/green]"
            passed_count += 1
        else:
            status_display = "[red]❌ FAILED[/red]"
            failed_count += 1
        
        error_display = error[:37] + "..." if error and len(error) > 40 else (error or "")
        table.add_row(test_name, status_display, error_display)
    
    console.print(table)
    
    # Overall result
    total_tests = len(test_results)
    console.print(f"\n📊 [bold]Overall Results:[/bold]")
    console.print(f"   Total Tests: {total_tests}")
    console.print(f"   Passed: [green]{passed_count}[/green]")
    console.print(f"   Failed: [red]{failed_count}[/red]")
    console.print(f"   Success Rate: {(passed_count/total_tests)*100:.1f}%")
    
    if failed_count == 0:
        console.print("\n🎉 [bold green]All tests passed! The naming system is working correctly.[/bold green]")
        return True
    else:
        console.print(f"\n⚠️ [bold yellow]{failed_count} test(s) failed. Please review the errors above.[/bold yellow]")
        return False

def test_naming_consistency():
    """Test current naming consistency across the platform"""
    
    console.print(Panel.fit(
        "[bold cyan]🔍 Naming Consistency Analysis[/bold cyan]\n"
        "[dim]Analyzing current naming patterns across the platform[/dim]",
        title="Consistency Check"
    ))
    
    try:
        result = subprocess.run([
            sys.executable, "scripts/migrate_platform_naming.py", "analyze"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            console.print(result.stdout)
            
            # Check if there are inconsistencies
            if "Found naming patterns" in result.stdout:
                console.print("\n⚠️ [yellow]Naming inconsistencies detected![/yellow]")
                console.print("💡 [dim]Consider running a migration to fix these issues:[/dim]")
                console.print("   [cyan]python scripts/migrate_platform_naming.py set-preset mlx --apply[/cyan]")
                console.print("   [cyan]python scripts/migrate_platform_naming.py set-preset mlsys --apply[/cyan]")
                return False
            else:
                console.print("✅ [green]No naming inconsistencies found![/green]")
                return True
        else:
            console.print(f"❌ [red]Analysis failed: {result.stderr}[/red]")
            return False
            
    except Exception as e:
        console.print(f"❌ [red]Analysis failed: {e}[/red]")
        return False

def main():
    """Main test runner"""
    console.print("[bold blue]🚀 Starting Naming Migration System Tests[/bold blue]\n")
    
    # Change to the project root directory
    project_root = Path(__file__).parent.parent
    import os
    os.chdir(project_root)
    
    # Add scripts to Python path
    sys.path.insert(0, str(project_root / "scripts"))
    
    # Run the test suite
    tests_passed = run_test_suite()
    
    # Run consistency analysis
    console.print("\n" + "="*60)
    consistency_ok = test_naming_consistency()
    
    # Final summary
    console.print("\n" + "="*60)
    if tests_passed and consistency_ok:
        console.print(Panel.fit(
            "[bold green]🎉 All tests passed and naming is consistent![/bold green]\n"
            "[dim]The naming migration system is working correctly.[/dim]",
            title="✅ SUCCESS"
        ))
        return 0
    else:
        console.print(Panel.fit(
            "[bold yellow]⚠️ Some issues detected[/bold yellow]\n"
            "[dim]Please review the test results and fix any issues.[/dim]",
            title="⚠️ ISSUES FOUND"
        ))
        return 1

if __name__ == "__main__":
    sys.exit(main()) 