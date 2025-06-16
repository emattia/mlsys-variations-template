#!/usr/bin/env python3
"""
Debug Configuration and Tools for MLSys Template
Provides debugging utilities, test runners, and diagnostics.
"""

import os
import sys
import logging
import pytest
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from dataclasses import dataclass
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

@dataclass
class DebugConfig:
    """Debug configuration settings."""
    log_level: str = "DEBUG"
    test_mode: bool = True
    verbose: bool = True
    save_logs: bool = True
    log_file: Optional[str] = None
    test_filter: Optional[str] = None
    coverage_report: bool = True

class DebugLogger:
    """Enhanced logger for debugging."""
    
    def __init__(self, config: DebugConfig):
        self.config = config
        self.setup_logging()
    
    def setup_logging(self):
        """Setup enhanced logging for debugging."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format=log_format,
            handlers=[]
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.config.log_level))
        console_formatter = logging.Formatter(log_format)
        console_handler.setFormatter(console_formatter)
        
        # File handler if specified
        handlers = [console_handler]
        if self.config.save_logs:
            log_file = self.config.log_file or f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, self.config.log_level))
            file_handler.setFormatter(console_formatter)
            handlers.append(file_handler)
        
        # Update root logger
        root_logger = logging.getLogger()
        root_logger.handlers = handlers
        
        # Set specific loggers to debug level
        for logger_name in [
            'src.utils.rate_limiter',
            'src.utils.cache_manager', 
            'src.utils.templates',
            'src.plugins.registry',
            'tests'
        ]:
            logging.getLogger(logger_name).setLevel(logging.DEBUG)

class TestRunner:
    """Enhanced test runner with debugging capabilities."""
    
    def __init__(self, config: DebugConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def run_single_test(self, test_path: str, **kwargs) -> int:
        """Run a single test with debugging."""
        args = [
            test_path,
            "-v",
            "--tb=long",
            "--capture=no" if self.config.verbose else "--capture=sys",
        ]
        
        if self.config.coverage_report:
            args.extend([
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov"
            ])
        
        # Add any additional args
        args.extend(kwargs.get("extra_args", []))
        
        self.logger.info(f"Running test: {test_path}")
        self.logger.debug(f"Pytest args: {args}")
        
        return pytest.main(args)
    
    def run_failing_tests(self) -> Dict[str, Any]:
        """Run only the failing tests identified earlier."""
        failing_tests = [
            "tests/integration/test_ml_systems_integration.py::TestMLSystemsIntegration::test_llm_pipeline_with_all_components",
            "tests/integration/test_ml_systems_integration.py::TestMLSystemsIntegration::test_concurrent_usage_with_all_systems"
        ]
        
        results = {}
        for test in failing_tests:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"DEBUGGING TEST: {test}")
            self.logger.info(f"{'='*60}")
            
            result = self.run_single_test(
                test,
                extra_args=["--pdb-trace", "--maxfail=1"]
            )
            results[test] = result
            
            if result != 0:
                self.logger.error(f"Test failed: {test}")
            else:
                self.logger.info(f"Test passed: {test}")
        
        return results
    
    def run_unit_tests(self) -> int:
        """Run unit tests only."""
        return self.run_single_test(
            "tests/",
            extra_args=["-m", "not integration", "--maxfail=5"]
        )
    
    def run_integration_tests(self) -> int:
        """Run integration tests only.""" 
        return self.run_single_test(
            "tests/integration/",
            extra_args=["--maxfail=3"]
        )

class SystemDiagnostics:
    """System diagnostics and health checks."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check if all dependencies are properly installed."""
        deps_status = {}
        
        try:
            import pytest
            deps_status["pytest"] = {"installed": True, "version": pytest.__version__}
        except ImportError as e:
            deps_status["pytest"] = {"installed": False, "error": str(e)}
        
        try:
            import coverage
            deps_status["coverage"] = {"installed": True, "version": coverage.__version__}
        except ImportError as e:
            deps_status["coverage"] = {"installed": False, "error": str(e)}
        
        # Check our modules
        try:
            from src.utils.rate_limiter import RateLimiter
            from src.utils.cache_manager import CacheManager
            from src.utils.templates import TemplateManager
            deps_status["core_modules"] = {"installed": True}
        except ImportError as e:
            deps_status["core_modules"] = {"installed": False, "error": str(e)}
        
        return deps_status
    
    def check_test_environment(self) -> Dict[str, Any]:
        """Check test environment setup."""
        env_status = {}
        
        # Check test directories
        test_dirs = ["tests/", "tests/unit/", "tests/integration/"]
        for test_dir in test_dirs:
            path = Path(test_dir)
            env_status[test_dir] = {
                "exists": path.exists(),
                "is_dir": path.is_dir(),
                "files": list(path.glob("*.py")) if path.exists() else []
            }
        
        # Check source directories
        src_dirs = ["src/", "src/utils/", "src/models/", "src/data/"]
        for src_dir in src_dirs:
            path = Path(src_dir)
            env_status[src_dir] = {
                "exists": path.exists(),
                "is_dir": path.is_dir(),
                "files": list(path.glob("*.py")) if path.exists() else []
            }
        
        return env_status
    
    def run_diagnostics(self) -> Dict[str, Any]:
        """Run all diagnostics."""
        self.logger.info("Running system diagnostics...")
        
        diagnostics = {
            "timestamp": datetime.now().isoformat(),
            "dependencies": self.check_dependencies(),
            "test_environment": self.check_test_environment(),
            "python_version": sys.version,
            "working_directory": os.getcwd(),
            "path": sys.path[:3]  # First 3 entries
        }
        
        return diagnostics

def create_debug_test_file():
    """Create a simple debug test file for troubleshooting."""
    debug_test_content = '''"""
Debug test file for troubleshooting rate limiting issues.
"""
import pytest
import asyncio
import time
import logging
from unittest.mock import patch, MagicMock

# Setup debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_rate_limiter_basic():
    """Basic test of rate limiter functionality."""
    from src.utils.rate_limiter import RateLimiter
    
    # Create rate limiter with very permissive limits for testing
    rate_limiter = RateLimiter({
        "test_service": {
            "requests": {"minute": 100, "hour": 1000},
            "cost": {"minute": 10.0, "hour": 100.0}
        }
    })
    
    # Test basic acquire
    result = asyncio.run(rate_limiter.acquire("test_service", cost=0.01))
    logger.debug(f"Basic acquire result: {result}")
    assert result is True
    
    # Check status
    status = rate_limiter.get_status("test_service")
    logger.debug(f"Status after acquire: {status}")
    
    # Should show 1 request
    assert "1/" in status["requests"]["minute"]

def test_cache_manager_basic():
    """Basic test of cache manager functionality."""
    from src.utils.cache_manager import CacheManager
    
    cache_manager = CacheManager()
    
    # Test basic caching
    prompt = "Test prompt"
    model = "test-model"
    response = "Test response"
    cost = 0.02
    
    cache_manager.cache_llm_response(prompt, model, response, cost)
    cached = cache_manager.get_llm_response(prompt, model)
    
    logger.debug(f"Cached response: {cached}")
    assert cached == response
    
    # Check stats
    stats = cache_manager.get_stats()
    logger.debug(f"Cache stats: {stats}")
    assert stats["total_entries"] >= 1

@pytest.mark.asyncio
async def test_rate_limiter_integration_debug():
    """Debug version of the failing integration test."""
    from src.utils.rate_limiter import RateLimiter
    from src.utils.cache_manager import CacheManager
    from src.utils.templates import TemplateManager
    import tempfile
    
    # Create components with debug logging
    with tempfile.TemporaryDirectory() as temp_dir:
        # Very permissive rate limits for debugging
        rate_limiter = RateLimiter({
            "test_llm": {
                "requests": {"minute": 100, "hour": 1000},
                "cost": {"minute": 10.0, "hour": 100.0}
            }
        })
        
        cache_manager = CacheManager()
        template_manager = TemplateManager(temp_dir)
        
        # Add test template
        template_manager.add_template(
            "test_prompt",
            "Classify: {{text}}",
            version="v1"
        )
        
        # Test 3 requests
        for i in range(3):
            logger.debug(f"\\n--- Request {i+1} ---")
            
            # Acquire rate limit
            acquired = await rate_limiter.acquire("test_llm", cost=0.02)
            logger.debug(f"Rate limit acquired: {acquired}")
            
            if acquired:
                # Use template
                prompt = template_manager.render_template(
                    "test_prompt",
                    {"text": f"test message {i}"},
                    version="v1"
                )
                logger.debug(f"Generated prompt: {prompt[:50]}...")
                
                # Simulate caching
                response = f"Response {i}"
                cache_manager.cache_llm_response(prompt, "gpt-4", response, 0.02)
                logger.debug(f"Cached response: {response}")
            
            # Check status
            status = rate_limiter.get_status("test_llm")
            logger.debug(f"Rate limiter status: {status}")
        
        # Final assertions
        final_status = rate_limiter.get_status("test_llm")
        logger.info(f"Final rate limiter status: {final_status}")
        
        cache_stats = cache_manager.get_stats()
        logger.info(f"Final cache stats: {cache_stats}")
        
        # This should pass with permissive limits
        assert cache_stats["total_entries"] >= 1
'''
    
    with open("debug_test.py", "w") as f:
        f.write(debug_test_content)
    
    print("Created debug_test.py - run with: python -m pytest debug_test.py -v -s")

def main():
    """Main debug interface."""
    config = DebugConfig(
        log_level="DEBUG",
        verbose=True,
        save_logs=True,
        coverage_report=True
    )
    
    # Setup logging
    debug_logger = DebugLogger(config)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting MLSys Debug Session")
    logger.info("="*50)
    
    # Run diagnostics
    diagnostics = SystemDiagnostics()
    diag_results = diagnostics.run_diagnostics()
    
    logger.info("System Diagnostics:")
    for key, value in diag_results.items():
        if key not in ["dependencies", "test_environment"]:
            logger.info(f"  {key}: {value}")
    
    # Check dependencies
    logger.info("\\nDependency Status:")
    for dep, status in diag_results["dependencies"].items():
        status_str = "✅" if status["installed"] else "❌"
        logger.info(f"  {dep}: {status_str}")
        if not status["installed"]:
            logger.error(f"    Error: {status.get('error', 'Unknown')}")
    
    # Run tests
    test_runner = TestRunner(config)
    
    print("\\n" + "="*60)
    print("DEBUG OPTIONS:")
    print("1. Run failing tests only")
    print("2. Run unit tests")
    print("3. Run integration tests")
    print("4. Create debug test file")
    print("5. Run all tests")
    print("="*60)
    
    choice = input("Enter choice (1-5): ").strip()
    
    if choice == "1":
        logger.info("Running failing tests...")
        results = test_runner.run_failing_tests()
        print(f"\\nResults: {results}")
    elif choice == "2":
        logger.info("Running unit tests...")
        result = test_runner.run_unit_tests()
        print(f"\\nUnit tests result: {result}")
    elif choice == "3":
        logger.info("Running integration tests...")
        result = test_runner.run_integration_tests()
        print(f"\\nIntegration tests result: {result}")
    elif choice == "4":
        create_debug_test_file()
        logger.info("Created debug_test.py")
    elif choice == "5":
        logger.info("Running all tests...")
        result = test_runner.run_single_test("tests/")
        print(f"\\nAll tests result: {result}")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main() 