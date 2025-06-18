"""
Tests for MLX Assistant functionality.

This module provides comprehensive testing for the MLX Assistant CLI tool,
including framework integration, project analysis, and interactive features.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typer.testing import CliRunner
import subprocess

# Import the MLX Assistant module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from mlx_assistant import app, MLXAssistant, show_welcome_dashboard

runner = CliRunner()

class TestMLXAssistantCore:
    """Test core MLX Assistant functionality."""
    
    def test_mlx_assistant_initialization(self):
        """Test MLX Assistant initializes correctly."""
        assistant = MLXAssistant()
        
        assert assistant.project_root is not None
        assert assistant.frameworks is not None
        assert assistant.project_state is not None
        assert len(assistant.frameworks) == 4  # golden_repos, security, plugins, glossary
    
    def test_framework_discovery(self):
        """Test framework discovery functionality."""
        assistant = MLXAssistant()
        frameworks = assistant._discover_frameworks()
        
        expected_frameworks = ["golden_repos", "security", "plugins", "glossary"]
        assert all(fw in frameworks for fw in expected_frameworks)
        
        # Test framework structure
        for fw_name, fw_data in frameworks.items():
            assert "name" in fw_data
            assert "description" in fw_data
            assert "script" in fw_data
            assert "commands" in fw_data
            assert "icon" in fw_data
            assert "status" in fw_data
    
    def test_project_state_analysis(self):
        """Test project state analysis functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mock mlx project structure
            (temp_path / "mlx.config.json").write_text('{"platform": {"name": "test"}}')
            (temp_path / "mlx-components").mkdir()
            (temp_path / "plugins").mkdir()
            
            # Create MLX Assistant with temp directory
            assistant = MLXAssistant()
            assistant.project_root = temp_path
            
            state = assistant._analyze_project_state()
            
            assert state["is_mlx_project"] == True
            assert state["has_components"] == True
            assert "recommendations" in state
    
    def test_recommendation_generation(self):
        """Test intelligent recommendation generation."""
        assistant = MLXAssistant()
        
        # Test recommendations for non-mlx project
        state = {"is_mlx_project": False, "has_components": False, "plugins_available": 0}
        recommendations = assistant._generate_recommendations(state)
        
        assert len(recommendations) > 0
        assert any("quick-start" in rec for rec in recommendations)
        
        # Test recommendations for mlx project without components
        state = {"is_mlx_project": True, "has_components": False, "plugins_available": 0}
        recommendations = assistant._generate_recommendations(state)
        
        assert any("golden-repos" in rec for rec in recommendations)


class TestMLXAssistantCLI:
    """Test MLX Assistant CLI interface."""
    
    def test_help_command(self):
        """Test MLX Assistant help command."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "MLX Assistant" in result.stdout
        assert "intelligent guide" in result.stdout
    
    def test_version_command(self):
        """Test version command."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "MLX Assistant v1.0.0" in result.stdout
    
    def test_doctor_command(self):
        """Test health check command."""
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        assert "Health Check" in result.stdout
    
    def test_quick_start_command(self):
        """Test quick start guide."""
        result = runner.invoke(app, ["quick-start"])
        assert result.exit_code == 0
        assert "Quick Start" in result.stdout
        assert "Step 1" in result.stdout
    
    def test_analyze_command(self):
        """Test project analysis command."""
        result = runner.invoke(app, ["analyze"])
        assert result.exit_code == 0
        assert "Project Analysis" in result.stdout


class TestFrameworkIntegration:
    """Test integration with underlying frameworks."""
    
    @patch('subprocess.run')
    def test_golden_repos_list(self, mock_subprocess):
        """Test golden repositories list command."""
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")
        
        result = runner.invoke(app, ["golden-repos", "list"])
        assert result.exit_code == 0
        assert "Golden Repository Specifications" in result.stdout
    
    @patch('subprocess.run')  
    def test_golden_repos_create(self, mock_subprocess):
        """Test golden repositories create command."""
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")
        
        result = runner.invoke(app, ["golden-repos", "create", "minimal"])
        assert result.exit_code == 0
        # Should show progress and success message
    
    @patch('subprocess.run')
    def test_security_scan(self, mock_subprocess):
        """Test security scan command.""" 
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")
        
        result = runner.invoke(app, ["security", "scan"])
        assert result.exit_code == 0
        # Should show progress for security scan
    
    @patch('subprocess.run')
    def test_plugins_list(self, mock_subprocess):
        """Test plugins list command."""
        mock_subprocess.return_value = Mock(returncode=0, stdout="Available plugins")
        
        result = runner.invoke(app, ["plugins", "list"])
        assert result.exit_code == 0
    
    def test_glossary_view(self):
        """Test glossary view command."""
        result = runner.invoke(app, ["glossary", "view"])
        # Should work regardless of whether glossary exists
        assert result.exit_code == 0


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @patch('subprocess.run')
    def test_framework_command_failure(self, mock_subprocess):
        """Test handling of framework command failures."""
        mock_subprocess.return_value = Mock(returncode=1, stdout="", stderr="Error occurred")
        
        result = runner.invoke(app, ["golden-repos", "create", "invalid-spec"])
        # Should handle error gracefully and show error message
        assert "Error" in result.stdout or result.exit_code != 0
    
    def test_invalid_commands(self):
        """Test handling of invalid commands."""
        result = runner.invoke(app, ["invalid-command"])
        assert result.exit_code != 0
    
    def test_missing_arguments(self):
        """Test handling of missing required arguments."""
        result = runner.invoke(app, ["golden-repos", "validate"])
        assert result.exit_code != 0  # Should require spec argument


class TestInteractiveMode:
    """Test interactive mode functionality (mocked)."""
    
    @patch('rich.prompt.Prompt.ask')
    @patch('scripts.mlx_assistant.start_interactive_mode')
    def test_interactive_mode_start(self, mock_interactive, mock_prompt):
        """Test starting interactive mode."""
        mock_prompt.return_value = "exit"
        mock_interactive.return_value = None
        
        result = runner.invoke(app, ["--interactive"])
        # Should start interactive mode
        assert result.exit_code == 0


class TestProjectStateScenarios:
    """Test different project state scenarios."""
    
    def test_empty_directory_analysis(self):
        """Test analysis of empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            assistant = MLXAssistant()
            assistant.project_root = Path(temp_dir)
            
            state = assistant._analyze_project_state()
            
            assert state["is_mlx_project"] == False
            assert state["has_components"] == False
            assert state["plugins_available"] == 0
    
    def test_partial_mlx_project_analysis(self):
        """Test analysis of partial mlx project."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create only config, no components
            (temp_path / "mlx.config.json").write_text('{"platform": {"name": "test"}}')
            
            assistant = MLXAssistant()
            assistant.project_root = temp_path
            
            state = assistant._analyze_project_state()
            
            assert state["is_mlx_project"] == True
            assert state["has_components"] == False
    
    def test_full_mlx_project_analysis(self):
        """Test analysis of complete mlx project."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create full mlx project structure
            (temp_path / "mlx.config.json").write_text('{"platform": {"name": "test"}}')
            (temp_path / "mlx-components").mkdir()
            plugins_dir = temp_path / "plugins"
            plugins_dir.mkdir()
            (plugins_dir / "mlx-plugin-test").mkdir()
            
            assistant = MLXAssistant()
            assistant.project_root = temp_path
            
            state = assistant._analyze_project_state()
            
            assert state["is_mlx_project"] == True
            assert state["has_components"] == True
            assert state["plugins_available"] >= 1


@pytest.fixture
def mock_mlx_project():
    """Fixture to create a mock mlx project structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mlx project structure
        config = {
            "platform": {"name": "test-project", "version": "0.1.0"},
            "components": ["api-serving", "config-management"]
        }
        (temp_path / "mlx.config.json").write_text(json.dumps(config))
        (temp_path / "mlx-components").mkdir()
        (temp_path / "plugins").mkdir()
        
        yield temp_path


class TestIntegrationWithMlsys:
    """Test integration with main mlx script."""
    
    def test_mlsys_frameworks_command(self):
        """Test frameworks command in main mlx script."""
        # This would require running the actual mlx script
        # For now, we test that the integration points exist
        from pathlib import Path
        mlsys_path = Path(__file__).parent.parent / "mlx"
        assert mlsys_path.exists()
    
    def test_mlsys_doctor_command(self):
        """Test doctor command integration."""
        # Test that the doctor command exists in mlx
        from pathlib import Path
        mlsys_path = Path(__file__).parent.parent / "mlx"
        content = mlsys_path.read_text()
        assert "doctor" in content
        assert "assistant" in content


if __name__ == "__main__":
    pytest.main([__file__]) 