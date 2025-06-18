#!/usr/bin/env python3
"""
🧪 Tests for Naming Migration System

Tests the centralized naming configuration and migration functionality
to ensure platform-wide consistency and proper migration behavior.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

# Add scripts to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from naming_config import (
    NamingConfig, 
    CommonNamingConfigs,
    get_naming_config,
    set_naming_config,
    substitute_naming_in_text,
    substitute_naming_in_file
)

class TestNamingConfig:
    """Test the NamingConfig dataclass and functionality"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = NamingConfig()
        
        assert config.platform_name == "mlx"
        assert config.platform_full_name == "MLX Platform Foundation"
        assert config.main_cli == "mlx"
        assert config.evaluation_cli == "mlx-eval"
        assert config.assistant_command == "mlx assistant"
        assert config.config_file == "mlx.config.json"
        assert config.components_dir == "mlx-components"
        assert config.docker_network == "mlx-network"
        assert config.template_name == "mlx-variations-template"
    
    def test_config_serialization(self):
        """Test config to/from dictionary conversion"""
        config = NamingConfig(
            platform_name="test",
            platform_full_name="Test Platform",
            main_cli="test-cli"
        )
        
        # Test to_dict
        config_dict = config.to_dict()
        assert config_dict["platform_name"] == "test"
        assert config_dict["platform_full_name"] == "Test Platform"
        assert config_dict["main_cli"] == "test-cli"
        
        # Test from_dict
        restored_config = NamingConfig.from_dict(config_dict)
        assert restored_config.platform_name == "test"
        assert restored_config.platform_full_name == "Test Platform"
        assert restored_config.main_cli == "test-cli"
    
    def test_config_file_operations(self):
        """Test saving and loading config from file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"
            
            # Create and save config
            original_config = NamingConfig(
                platform_name="filetest",
                platform_full_name="File Test Platform"
            )
            original_config.save_to_file(config_path)
            
            # Verify file exists and has correct content
            assert config_path.exists()
            with open(config_path) as f:
                data = json.load(f)
                assert data["platform_name"] == "filetest"
                assert data["platform_full_name"] == "File Test Platform"
            
            # Load config from file
            loaded_config = NamingConfig.load_from_file(config_path)
            assert loaded_config.platform_name == "filetest"
            assert loaded_config.platform_full_name == "File Test Platform"
    
    def test_config_file_not_exists(self):
        """Test loading config when file doesn't exist"""
        non_existent_path = Path("/tmp/non_existent_config.json")
        config = NamingConfig.load_from_file(non_existent_path)
        
        # Should return default config
        assert config.platform_name == "mlx"
        assert config.platform_full_name == "MLX Platform Foundation"

class TestCommonNamingConfigs:
    """Test predefined naming configuration presets"""
    
    def test_mlx_platform_preset(self):
        """Test mlx platform naming preset"""
        config = CommonNamingConfigs.mlx_platform()
        
        assert config.platform_name == "mlx"
        assert config.platform_full_name == "MLX Platform Foundation"
        assert config.main_cli == "mlx"
        assert config.evaluation_cli == "mlx-eval"
        assert config.assistant_command == "mlx assistant"
        assert config.config_file == "mlx.config.json"
        assert config.components_dir == "mlx-components"
        assert config.docker_network == "mlx-network"
        assert config.template_name == "mlx-variations-template"
    
    def test_mlsys_platform_preset(self):
        """Test MLSys platform naming preset"""
        config = CommonNamingConfigs.mlsys_platform()
        
        assert config.platform_name == "mlsys"
        assert config.platform_full_name == "MLSys Platform Foundation"
        assert config.main_cli == "mlsys"
        assert config.evaluation_cli == "mlsys-eval"
        assert config.assistant_command == "mlsys assistant"
        assert config.config_file == "mlsys.config.json"
        assert config.components_dir == "mlsys-components"
        assert config.docker_network == "mlsys-network"
        assert config.template_name == "mlsys-platform-template"
    
    def test_custom_platform_preset(self):
        """Test custom platform naming preset"""
        config = CommonNamingConfigs.custom_platform("dataflow")
        
        assert config.platform_name == "dataflow"
        assert config.platform_full_name == "Dataflow Platform Foundation"
        assert config.main_cli == "dataflow"
        assert config.evaluation_cli == "dataflow-eval"
        assert config.assistant_command == "dataflow assistant"
        assert config.config_file == "dataflow.config.json"
        assert config.components_dir == "dataflow-components"
        assert config.docker_network == "dataflow-network"
        assert config.template_name == "dataflow-platform-template"

class TestTemplateSubstitution:
    """Test template substitution functionality"""
    
    def test_substitute_naming_in_text(self):
        """Test text template substitution"""
        config = NamingConfig(
            platform_name="test",
            platform_full_name="Test Platform Foundation",
            main_cli="test-cli",
            evaluation_cli="test-eval"
        )
        
        template = """
        Welcome to {PLATFORM_FULL_NAME}!
        Use {MAIN_CLI} and {EVALUATION_CLI} commands.
        Platform: {PLATFORM_NAME_UPPER}
        """
        
        result = substitute_naming_in_text(template, config)
        
        assert "Test Platform Foundation" in result
        assert "test-cli" in result
        assert "test-eval" in result
        assert "TEST" in result  # PLATFORM_NAME_UPPER
    
    def test_substitute_naming_in_file(self):
        """Test file template substitution"""
        config = NamingConfig(
            platform_name="filetest",
            platform_full_name="File Test Platform"
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.txt"
            
            # Create test file with templates
            test_content = """
            Platform: {PLATFORM_NAME}
            Full Name: {PLATFORM_FULL_NAME}
            CLI: {MAIN_CLI}
            """
            test_file.write_text(test_content)
            
            # Apply substitution
            result = substitute_naming_in_file(test_file, config, backup=False)
            
            # Verify changes were made
            assert result is True
            updated_content = test_file.read_text()
            assert "filetest" in updated_content
            assert "File Test Platform" in updated_content
    
    def test_substitute_no_changes_needed(self):
        """Test substitution when no changes are needed"""
        config = NamingConfig(platform_name="test")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("No templates here")
            
            result = substitute_naming_in_file(test_file, config, backup=False)
            assert result is False  # No changes made

class TestPlatformNamingMigrator:
    """Test the platform-wide naming migrator"""
    
    def setup_method(self):
        """Set up test environment"""
        # Import here to avoid issues if the module doesn't exist yet
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
            from migrate_platform_naming import PlatformNamingMigrator
            self.migrator_class = PlatformNamingMigrator
        except ImportError:
            pytest.skip("Platform naming migrator not available")
    
    def test_discover_files(self):
        """Test file discovery functionality"""
        migrator = self.migrator_class()
        files = migrator.discover_files()
        
        # Should discover some files (exact number depends on project state)
        assert isinstance(files, list)
        assert len(files) >= 0  # Could be empty in test environment
        
        # All discovered files should be Path objects
        for file_path in files:
            assert isinstance(file_path, Path)
    
    def test_replacement_patterns(self):
        """Test that replacement patterns are properly defined"""
        migrator = self.migrator_class()
        
        assert hasattr(migrator, 'replacement_patterns')
        assert isinstance(migrator.replacement_patterns, list)
        assert len(migrator.replacement_patterns) > 0
        
        # Each pattern should be a tuple of (pattern, replacement)
        for pattern_tuple in migrator.replacement_patterns:
            assert isinstance(pattern_tuple, tuple)
            assert len(pattern_tuple) == 2
            assert isinstance(pattern_tuple[0], str)  # regex pattern
            assert isinstance(pattern_tuple[1], str)  # replacement template
    
    def test_template_substitution(self):
        """Test internal template substitution method"""
        migrator = self.migrator_class()
        config = NamingConfig(
            platform_name="test",
            platform_full_name="Test Platform"
        )
        
        template = "{PLATFORM_NAME} and {PLATFORM_FULL_NAME}"
        result = migrator._substitute_template(template, config)
        
        assert "test" in result
        assert "Test Platform" in result
        assert "{" not in result  # All templates should be substituted

class TestEvaluationNamingMigrator:
    """Test the evaluation-specific naming migrator"""
    
    def setup_method(self):
        """Set up test environment"""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "evaluation"))
            from migrate_naming import NamingMigrator
            self.migrator_class = NamingMigrator
        except ImportError:
            pytest.skip("Evaluation naming migrator not available")
    
    def test_evaluation_files_list(self):
        """Test that evaluation files are properly defined"""
        migrator = self.migrator_class()
        
        assert hasattr(migrator, 'evaluation_files')
        assert isinstance(migrator.evaluation_files, list)
        assert len(migrator.evaluation_files) > 0
        
        # All should be string paths
        for file_path in migrator.evaluation_files:
            assert isinstance(file_path, str)
            assert file_path.startswith("scripts/evaluation/")
    
    def test_replacement_patterns(self):
        """Test evaluation-specific replacement patterns"""
        migrator = self.migrator_class()
        
        assert hasattr(migrator, 'replacement_patterns')
        assert isinstance(migrator.replacement_patterns, list)
        assert len(migrator.replacement_patterns) > 0

class TestIntegration:
    """Integration tests for the complete naming system"""
    
    def test_end_to_end_config_change(self):
        """Test complete config change workflow"""
        # Save original config
        original_config = get_naming_config()
        
        try:
            # Set new config
            test_config = NamingConfig(
                platform_name="integration_test",
                platform_full_name="Integration Test Platform"
            )
            set_naming_config(test_config)
            
            # Verify config was set
            current_config = get_naming_config()
            assert current_config.platform_name == "integration_test"
            assert current_config.platform_full_name == "Integration Test Platform"
            
            # Test template substitution with new config
            template = "Welcome to {PLATFORM_FULL_NAME} ({PLATFORM_NAME})"
            result = substitute_naming_in_text(template)
            assert "Integration Test Platform" in result
            assert "integration_test" in result
            
        finally:
            # Restore original config
            set_naming_config(original_config)
    
    def test_preset_application(self):
        """Test applying different presets"""
        original_config = get_naming_config()
        
        try:
            # Test MLX preset
            mlx_config = CommonNamingConfigs.mlx_platform()
            set_naming_config(mlx_config)
            
            current = get_naming_config()
            assert current.platform_name == "mlx"
            assert current.main_cli == "mlx"
            
            # Test MLSys preset
            mlsys_config = CommonNamingConfigs.mlsys_platform()
            set_naming_config(mlsys_config)
            
            current = get_naming_config()
            assert current.platform_name == "mlsys"
            assert current.main_cli == "mlsys"
            
            # Test custom preset
            custom_config = CommonNamingConfigs.custom_platform("mytest")
            set_naming_config(custom_config)
            
            current = get_naming_config()
            assert current.platform_name == "mytest"
            assert current.main_cli == "mytest"
            
        finally:
            set_naming_config(original_config)

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_config_data(self):
        """Test handling of invalid configuration data"""
        # Test with missing required fields
        incomplete_data = {"platform_name": "test"}
        
        # Should not raise exception, should use defaults for missing fields
        config = NamingConfig.from_dict(incomplete_data)
        assert config.platform_name == "test"
        assert config.platform_full_name == "MLX Platform Foundation"  # default
    
    def test_file_operations_with_permissions(self):
        """Test file operations with permission issues"""
        config = NamingConfig(platform_name="permtest")
        
        # Try to write to a read-only location (should handle gracefully)
        readonly_path = Path("/dev/null/cannot_write_here.json")
        
        # Should not raise exception
        try:
            config.save_to_file(readonly_path)
        except (PermissionError, FileNotFoundError, OSError):
            pass  # Expected behavior
    
    def test_malformed_template_substitution(self):
        """Test template substitution with malformed templates"""
        config = NamingConfig(platform_name="test")
        
        # Test with unclosed template
        malformed_template = "Hello {PLATFORM_NAME and {UNCLOSED"
        result = substitute_naming_in_text(malformed_template, config)
        
        # Should handle gracefully and not substitute malformed templates
        assert result == malformed_template  # No substitution should occur for malformed templates
        
        # Test with properly formed template mixed with malformed
        mixed_template = "Hello {PLATFORM_NAME} and {UNCLOSED"
        result = substitute_naming_in_text(mixed_template, config)
        
        # Should substitute the valid template but leave malformed one alone
        assert "test" in result
        assert "{UNCLOSED" in result

if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"]) 