# Security Scanning Guide

> "Security is not a product, but a process." — Bruce Schneier

<!-- NOTE: This file was moved from the project root to docs/development/ to keep all documentation inside the MkDocs tree. -->

This project includes comprehensive security scanning capabilities that can be run both locally and in CI/CD.

## 🚀 Quick Start

### 1. Install Security Tools
```bash
# Complete setup (includes Trivy)
make setup

# Or install just Trivy
make install-trivy
```

### 2. Run Security Scans

#### Local Filesystem Scan
```bash
# Quick scan (HIGH/CRITICAL issues only)
make trivy-fs-scan

# Comprehensive scan (all severities)
make trivy-fs-scan-all

# All security checks (Bandit + Trivy)
make security-scan-local
```

#### Docker Image Scan
```bash
# Build image first
make docker-build

# Scan the image
make trivy-image-scan IMAGE=mlops-template:latest

# Or use comprehensive scan
make trivy-image-scan-all IMAGE=mlops-template:latest
```

#### All Quality + Security Checks
```bash
# Recommended before pushing to GitHub
make all-checks
```

## 🔧 What Gets Scanned

### Filesystem Scanning (`trivy-fs-scan`)
- **Vulnerabilities**: Known CVEs in dependencies
- **Secrets**: Hardcoded passwords, API keys, tokens
- **Configuration**: Insecure settings in config files
- **License Issues**: Non-compliant licenses

### Code Scanning (`security-check`)
- **Code Vulnerabilities**: Bandit static analysis
- **Security Anti-patterns**: Dangerous function usage
- **Injection Risks**: SQL injection, command injection

### Docker Image Scanning (`trivy-image-scan`)
- **Base Image Vulnerabilities**: OS package CVEs
- **Layer Analysis**: Security issues in each layer
- **Runtime Security**: Container configuration issues

## 🎯 Integration with CI/CD

### GitHub Actions
The security scans automatically run in CI/CD:

1. **On every push**: Filesystem and code scanning
2. **After Docker build**: Image vulnerability scanning
3. **Results uploaded**: GitHub Security tab (SARIF format)

### Local Pre-commit
Set up pre-commit hooks to catch issues early:
```bash
make pre-commit
```

## 📊 Understanding Results

### Severity Levels
- **CRITICAL**: Immediate action required
- **HIGH**: Address before deployment
- **MEDIUM**: Fix in next iteration
- **LOW**: Informational, fix when convenient

### Common Issues & Fixes

#### Python Dependencies
```bash
# Update vulnerable packages
uv pip install package_name==secure_version

# Or update all packages
uv pip sync requirements.txt
```

#### Docker Base Images
```dockerfile
# Update base image in Dockerfile
FROM python:3.11-slim-bullseye  # Use latest secure version
```

#### Secrets in Code
```bash
# Use environment variables instead
export API_KEY=your_secret_key

# Or use .env file (add to .gitignore)
echo "API_KEY=your_secret_key" >> .env
```

## 🔒 Security Best Practices

1. **Regular Scanning**: Run `make security-scan-local` before each commit
2. **Update Dependencies**: Keep packages up-to-date
3. **Secure Secrets**: Never commit secrets, use environment variables
4. **Review Results**: Don't ignore security warnings
5. **Monitor CI**: Check GitHub Security tab regularly

## 📋 Troubleshooting

### Trivy Installation Issues
```bash
# Check if Trivy is installed
command -v trivy

# Reinstall if needed
make install-trivy

# Verify installation
make verify-setup
```

### False Positives
If you get false positives, you can create a `.trivyignore` file:
```bash
# Ignore specific CVE
CVE-2021-12345

# Ignore specific file
path/to/file.py
```

### GitHub Actions Failures
1. Check the Security tab for SARIF results
2. Review the Actions logs for specific errors
3. Run the same scan locally to debug

## 🚀 Advanced Usage

### Custom Trivy Configuration
Create `.trivy.yaml` for project-specific settings:
```yaml
vulnerability:
  type:
    - os
    - library
secret:
  config: trivy-secret.yaml
```

### Integration with IDEs
Many IDEs support Trivy and Bandit integration for real-time security feedback.

# Security Scanning
