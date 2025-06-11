# Forking Procedure Tests

> "Program testing can be used to show the presence of bugs, but never to show their absence!" — Edsger W. Dijkstra

This document explains the testing infrastructure for validating the mlsys forking procedure.

## Overview

The forking procedure is critical to the user experience—it transforms a generic template into a personalized project. These tests ensure that transformation works correctly across different scenarios.

## Test Components

### 1. Smoke Test (`scripts/test_forking_smoke.py`)

A lightweight test that quickly validates the core forking functionality.

**Usage:**
```bash
# Quick test with default name
make test-forking-smoke

# Test with custom project name
make test-forking-custom NAME=my-project

# Validate with debug output
make validate-forking
```

**What it tests:**
- ✅ mlsys script execution
- ✅ Directory renaming (`src/analysis_template` → `src/project_name`)
- ✅ pyproject.toml updates (project name, packages)
- ✅ Make command compatibility
- ✅ Project structure integrity

### 2. Comprehensive Tests (`tests/integration/test_forking_procedure.py`)

Full test suite with extensive coverage of edge cases and integration scenarios.

**Usage:**
```bash
# Run all forking tests
make test-forking-full

# Run specific test categories
pytest tests/integration/test_forking_procedure.py::TestForkingProcedure::test_mlsys_transformation_complex_name -v
```

**What it tests:**
- 🧪 Complex project names (hyphens, underscores, mixed)
- 📝 Documentation updates across multiple files
- 🔍 Import integrity after transformation
- ⚙️ Configuration file validation
- 🔄 Idempotency (running mlsys twice)
- 🚀 Full workflow integration (transform → install → test)

### 3. CI/CD Integration (`.github/workflows/forking-tests.yml`)

Automated testing across multiple Python versions and scenarios.

**Features:**
- 🐍 Multi-Python version testing (3.10, 3.11, 3.12)
- 📊 Custom project name matrix testing
- ⏰ Daily regression testing
- 🎯 Manual workflow dispatch with parameters
- 📈 Test reporting and artifact collection

## Test Validation Criteria

### Basic Validation
1. **Script Execution**: mlsys runs without errors
2. **Directory Structure**: Old directory removed, new directory created
3. **Configuration Updates**: pyproject.toml reflects new project name
4. **Make Compatibility**: `make help` works after transformation

### Advanced Validation
1. **Import Integrity**: Python can import the renamed package
2. **Complex Names**: Handles various naming conventions correctly
3. **Documentation**: Updates README, docs, and configuration files
4. **Error Handling**: Graceful failure when re-run or invalid input

## Running Tests Locally

### Quick Validation
```bash
# Fastest way to check if forking works
make test-forking-smoke

# Test with a specific project name
python3 scripts/test_forking_smoke.py financial-risk-model
```

### Development Testing
```bash
# Full test suite during development
make test-forking-full

# Keep temporary files for debugging
python3 scripts/test_forking_smoke.py debug-project --keep-temp
```

### Continuous Integration
```bash
# Simulate CI environment
pytest tests/integration/test_forking_procedure.py -v --tb=short
```

## Test Scenarios Covered

| Scenario | Test Type | Description |
|----------|-----------|-------------|
| **Basic Transform** | Smoke | Simple project name transformation |
| **Complex Names** | Comprehensive | Hyphens, underscores, mixed formats |
| **Documentation** | Comprehensive | README, docs, config file updates |
| **Import Validation** | Comprehensive | Python import system compatibility |
| **Configuration** | Both | pyproject.toml, Makefile, tool configs |
| **Idempotency** | Comprehensive | Running transformation multiple times |
| **Error Handling** | Comprehensive | Invalid inputs, missing files |
| **Integration** | Comprehensive | Full workflow with dependency installation |

## Interpreting Test Results

### Success Indicators
- ✅ All validation checks pass
- ✅ No TOML parsing errors (or graceful fallback)
- ✅ Project structure is correct
- ✅ Make commands work

### Common Issues

#### TOML Parsing Warnings
```
Warning: Could not parse pyproject.toml: Expected '=' after a key in a key/value pair
✅ Verified project name update via text search
```
**Explanation**: The TOML writer generates slightly non-standard syntax, but the transformation still works. The test falls back to text-based verification.

#### Directory Not Found
```
❌ Error: Project already appears to be initialized
```
**Explanation**: The mlsys script was already run in this directory. Start with a fresh copy of the template.

#### Import Errors
```
❌ Import failed after transformation: No module named 'project_name'
```
**Explanation**: The directory renaming didn't complete properly, or there are syntax errors in the renamed files.

## Extending the Tests

### Adding New Test Cases

1. **New Project Names**: Add to the test matrix in the GitHub workflow
2. **New Validation**: Extend the `verify_transformation` function
3. **Branch-Specific**: Add tests in `TestBranchSpecificForking`

### Custom Test Scenarios

```python
# Add to tests/integration/test_forking_procedure.py
def test_custom_scenario(self, temp_project_dir: Path):
    """Test custom forking scenario."""
    project_name = "my-custom-test"

    # Run transformation
    result = subprocess.run(
        [str(temp_project_dir / "mlsys"), project_name],
        cwd=temp_project_dir,
        capture_output=True,
        text=True,
        timeout=120
    )

    assert result.returncode == 0, f"Transformation failed: {result.stderr}"
    # Add custom validations here
```

## Monitoring and Maintenance

### Automated Monitoring
- 🕒 **Daily Tests**: Scheduled GitHub Actions run at 3 AM UTC
- 📊 **Matrix Testing**: Tests across Python versions and project names
- 🚨 **Failure Notifications**: Automatic alerts for broken functionality

### Manual Monitoring
```bash
# Weekly health check
make validate-forking

# Before releasing changes
make test-forking-full

# Test new project name patterns
make test-forking-custom NAME=new-pattern-test
```

### Updating Tests
When modifying the mlsys script:

1. **Run Smoke Test**: `make test-forking-smoke`
2. **Run Full Suite**: `make test-forking-full`
3. **Test Edge Cases**: Manual testing with complex names
4. **Update Documentation**: Reflect any new test requirements

## Troubleshooting

### Test Environment Issues
```bash
# Install test dependencies
pip install pytest tomli-w toml

# Check Python version
python3 --version  # Should be 3.10+

# Verify mlsys permissions
ls -la mlsys
chmod +x mlsys  # If not executable
```

### CI/CD Issues
- Check GitHub Actions logs for specific failure details
- Verify workflow permissions for artifact uploads
- Ensure all test dependencies are properly specified

### Performance Issues
- Smoke test should complete in <30 seconds
- Full test suite should complete in <5 minutes
- CI pipeline should complete in <15 minutes

## Best Practices

1. **Run Tests Early**: Always test forking changes before committing
2. **Use Descriptive Names**: Test with realistic project names
3. **Clean Environment**: Start with fresh template copies
4. **Monitor CI**: Watch for regressions in automated tests
5. **Document Changes**: Update tests when modifying mlsys behavior

---

**Need help?** Check the test logs, run with `--keep-temp` for debugging, or consult the GitHub Actions results for detailed failure information.
