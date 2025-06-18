#!/usr/bin/env python3
"""Enhanced Security Hardening Framework

This module provides comprehensive security hardening capabilities including:
- SBOM (Software Bill of Materials) generation
- Cosign verification patterns
- Enhanced vulnerability scanning
- Security policy enforcement
- Compliance reporting
"""

import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import yaml
import hashlib
from enum import Enum

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security compliance levels."""
    BASIC = "basic"
    ENHANCED = "enhanced"
    ENTERPRISE = "enterprise"
    CRITICAL = "critical"


class ScanType(Enum):
    """Types of security scans."""
    FILESYSTEM = "filesystem"
    CONTAINER = "container"
    DEPENDENCIES = "dependencies"
    SECRETS = "secrets"
    CONFIGURATION = "configuration"
    CODE = "code"


@dataclass
class SecurityFinding:
    """Represents a security finding."""
    scan_type: ScanType
    severity: str
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    cve_id: Optional[str] = None
    recommendation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityReport:
    """Comprehensive security report."""
    scan_timestamp: float
    project_path: str
    security_level: SecurityLevel
    findings: List[SecurityFinding] = field(default_factory=list)
    sbom: Optional[Dict[str, Any]] = None
    compliance_status: Dict[str, bool] = field(default_factory=dict)
    metrics: Dict[str, Union[int, float]] = field(default_factory=dict)
    
    @property
    def critical_findings(self) -> List[SecurityFinding]:
        """Get critical severity findings."""
        return [f for f in self.findings if f.severity.upper() == "CRITICAL"]
    
    @property
    def high_findings(self) -> List[SecurityFinding]:
        """Get high severity findings.""" 
        return [f for f in self.findings if f.severity.upper() == "HIGH"]
    
    @property
    def total_findings(self) -> int:
        """Total number of findings."""
        return len(self.findings)
    
    @property
    def risk_score(self) -> float:
        """Calculate risk score based on findings."""
        weights = {"CRITICAL": 10, "HIGH": 5, "MEDIUM": 2, "LOW": 1}
        score = sum(weights.get(f.severity.upper(), 0) for f in self.findings)
        return min(score / 10.0, 10.0)  # Normalize to 0-10 scale


class SecurityHardeningFramework:
    """Comprehensive security hardening framework."""
    
    def __init__(self, project_path: Path, security_level: SecurityLevel = SecurityLevel.ENHANCED):
        self.project_path = Path(project_path)
        self.security_level = security_level
        self.tools_config = self._load_tools_config()
        self.policies = self._load_security_policies()
        
        # Ensure required tools are available
        self._verify_security_tools()
    
    def _load_tools_config(self) -> Dict[str, Any]:
        """Load security tools configuration."""
        return {
            "trivy": {
                "binary": "trivy",
                "filesystem_args": ["fs", "--scanners", "vuln,secret,misconfig"],
                "image_args": ["image"],
                "output_format": "json"
            },
            "bandit": {
                "binary": "bandit",
                "args": ["-r", "-f", "json"],
                "config_file": ".bandit"
            },
            "safety": {
                "binary": "safety",
                "args": ["check", "--json"],
                "ignore_file": ".safety-ignore"
            },
            "cosign": {
                "binary": "cosign",
                "verify_args": ["verify"],
                "sign_args": ["sign"]
            },
            "syft": {
                "binary": "syft",
                "args": ["-o", "spdx-json"],
                "output_format": "spdx-json"
            }
        }
    
    def _load_security_policies(self) -> Dict[str, Any]:
        """Load security policies based on security level."""
        base_policies = {
            "max_critical_findings": 0,
            "max_high_findings": 5,
            "max_medium_findings": 20,
            "required_scans": [ScanType.FILESYSTEM, ScanType.DEPENDENCIES, ScanType.CODE],
            "sbom_required": True,
            "signature_verification": False,
            "compliance_frameworks": ["cis", "nist"]
        }
        
        level_overrides = {
            SecurityLevel.BASIC: {
                "max_high_findings": 10,
                "max_medium_findings": 50,
                "required_scans": [ScanType.FILESYSTEM, ScanType.CODE],
                "sbom_required": False
            },
            SecurityLevel.ENHANCED: {
                "signature_verification": True,
                "required_scans": [
                    ScanType.FILESYSTEM, ScanType.DEPENDENCIES, 
                    ScanType.CODE, ScanType.SECRETS
                ]
            },
            SecurityLevel.ENTERPRISE: {
                "max_high_findings": 2,
                "max_medium_findings": 10,
                "signature_verification": True,
                "required_scans": list(ScanType),
                "compliance_frameworks": ["cis", "nist", "sox", "pci"]
            },
            SecurityLevel.CRITICAL: {
                "max_high_findings": 0,
                "max_medium_findings": 5,
                "signature_verification": True,
                "required_scans": list(ScanType)
            }
        }
        
        policies = base_policies.copy()
        if self.security_level in level_overrides:
            policies.update(level_overrides[self.security_level])
        
        return policies
    
    def _verify_security_tools(self):
        """Verify that required security tools are available."""
        required_tools = ["trivy"]
        
        if self.security_level in [SecurityLevel.ENHANCED, SecurityLevel.ENTERPRISE, SecurityLevel.CRITICAL]:
            required_tools.extend(["bandit", "safety"])
        
        if self.policies.get("sbom_required"):
            required_tools.append("syft")
        
        if self.policies.get("signature_verification"):
            required_tools.append("cosign")
        
        missing_tools = []
        for tool in required_tools:
            if not self._tool_available(self.tools_config[tool]["binary"]):
                missing_tools.append(tool)
        
        if missing_tools:
            raise RuntimeError(f"Missing security tools: {missing_tools}")
    
    def _tool_available(self, tool_binary: str) -> bool:
        """Check if a security tool is available."""
        try:
            subprocess.run([tool_binary, "--version"], 
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def run_comprehensive_scan(self) -> SecurityReport:
        """Run comprehensive security scan based on security level."""
        logger.info(f"Starting comprehensive security scan at level: {self.security_level.value}")
        
        report = SecurityReport(
            scan_timestamp=time.time(),
            project_path=str(self.project_path),
            security_level=self.security_level
        )
        
        # Run required scans
        for scan_type in self.policies["required_scans"]:
            try:
                findings = self._run_scan_by_type(scan_type)
                report.findings.extend(findings)
                logger.info(f"Completed {scan_type.value} scan: {len(findings)} findings")
            except Exception as e:
                logger.error(f"Failed to run {scan_type.value} scan: {e}")
        
        # Generate SBOM if required
        if self.policies.get("sbom_required"):
            try:
                report.sbom = self.generate_sbom()
                logger.info("Generated SBOM")
            except Exception as e:
                logger.error(f"Failed to generate SBOM: {e}")
        
        # Check compliance
        report.compliance_status = self._check_compliance(report)
        
        # Calculate metrics
        report.metrics = self._calculate_metrics(report)
        
        logger.info(f"Security scan completed: {report.total_findings} findings, "
                   f"risk score: {report.risk_score:.1f}")
        
        return report
    
    def _run_scan_by_type(self, scan_type: ScanType) -> List[SecurityFinding]:
        """Run a specific type of security scan."""
        if scan_type == ScanType.FILESYSTEM:
            return self._run_trivy_filesystem_scan()
        elif scan_type == ScanType.CONTAINER:
            return self._run_trivy_container_scan()
        elif scan_type == ScanType.DEPENDENCIES:
            return self._run_dependency_scan()
        elif scan_type == ScanType.CODE:
            return self._run_code_scan()
        elif scan_type == ScanType.SECRETS:
            return self._run_secrets_scan()
        elif scan_type == ScanType.CONFIGURATION:
            return self._run_configuration_scan()
        else:
            return []
    
    def _run_trivy_filesystem_scan(self) -> List[SecurityFinding]:
        """Run Trivy filesystem vulnerability scan."""
        config = self.tools_config["trivy"]
        
        cmd = [config["binary"]] + config["filesystem_args"]
        if self.security_level in [SecurityLevel.ENTERPRISE, SecurityLevel.CRITICAL]:
            cmd.extend(["--severity", "CRITICAL,HIGH,MEDIUM"])
        else:
            cmd.extend(["--severity", "CRITICAL,HIGH"])
        
        cmd.extend(["-f", config["output_format"], str(self.project_path)])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            if result.returncode == 0 or result.stdout:
                scan_results = json.loads(result.stdout) if result.stdout else {"Results": []}
                return self._parse_trivy_results(scan_results, ScanType.FILESYSTEM)
            else:
                logger.warning(f"Trivy filesystem scan returned no results: {result.stderr}")
                return []
                
        except Exception as e:
            logger.error(f"Trivy filesystem scan failed: {e}")
            return []
    
    def _run_trivy_container_scan(self) -> List[SecurityFinding]:
        """Run Trivy container image scan."""
        # Look for Dockerfile or built images
        dockerfile_path = self.project_path / "Dockerfile"
        if not dockerfile_path.exists():
            logger.info("No Dockerfile found, skipping container scan")
            return []
        
        # For this implementation, we'll scan the Dockerfile itself
        # In practice, you'd scan built images
        config = self.tools_config["trivy"]
        
        cmd = [config["binary"], "config", "--severity", "CRITICAL,HIGH", 
               "-f", "json", str(dockerfile_path)]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            if result.returncode == 0 or result.stdout:
                scan_results = json.loads(result.stdout) if result.stdout else {"Results": []}
                return self._parse_trivy_results(scan_results, ScanType.CONTAINER)
            else:
                return []
                
        except Exception as e:
            logger.error(f"Trivy container scan failed: {e}")
            return []
    
    def _run_dependency_scan(self) -> List[SecurityFinding]:
        """Run dependency vulnerability scan using Safety."""
        if not self._tool_available("safety"):
            logger.warning("Safety not available, skipping dependency scan")
            return []
        
        config = self.tools_config["safety"]
        findings = []
        
        # Check for requirements files
        req_files = [
            "requirements.txt", "requirements-dev.txt", "requirements-prod.txt"
        ]
        
        for req_file in req_files:
            req_path = self.project_path / req_file
            if req_path.exists():
                cmd = [config["binary"]] + config["args"] + ["-r", str(req_path)]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                    
                    if result.stdout:
                        safety_results = json.loads(result.stdout)
                        findings.extend(self._parse_safety_results(safety_results, req_file))
                        
                except Exception as e:
                    logger.error(f"Safety scan failed for {req_file}: {e}")
        
        return findings
    
    def _run_code_scan(self) -> List[SecurityFinding]:
        """Run code security scan using Bandit."""
        if not self._tool_available("bandit"):
            logger.warning("Bandit not available, skipping code scan")
            return []
        
        config = self.tools_config["bandit"]
        src_paths = []
        
        # Find Python source directories
        for path in ["src", "app", "lib"]:
            src_path = self.project_path / path
            if src_path.exists() and src_path.is_dir():
                src_paths.append(str(src_path))
        
        if not src_paths:
            logger.info("No Python source directories found, skipping code scan")
            return []
        
        findings = []
        for src_path in src_paths:
            cmd = [config["binary"]] + config["args"] + [src_path]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                
                if result.stdout:
                    bandit_results = json.loads(result.stdout)
                    findings.extend(self._parse_bandit_results(bandit_results))
                    
            except Exception as e:
                logger.error(f"Bandit scan failed for {src_path}: {e}")
        
        return findings
    
    def _run_secrets_scan(self) -> List[SecurityFinding]:
        """Run secrets detection scan."""
        # Use Trivy's secret detection capability
        config = self.tools_config["trivy"]
        
        cmd = [config["binary"], "fs", "--scanners", "secret", 
               "--severity", "CRITICAL,HIGH,MEDIUM", 
               "-f", "json", str(self.project_path)]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            if result.returncode == 0 or result.stdout:
                scan_results = json.loads(result.stdout) if result.stdout else {"Results": []}
                return self._parse_trivy_results(scan_results, ScanType.SECRETS)
            else:
                return []
                
        except Exception as e:
            logger.error(f"Secrets scan failed: {e}")
            return []
    
    def _run_configuration_scan(self) -> List[SecurityFinding]:
        """Run configuration security scan."""
        # Use Trivy's misconfiguration detection
        config = self.tools_config["trivy"]
        
        cmd = [config["binary"], "fs", "--scanners", "misconfig",
               "--severity", "CRITICAL,HIGH,MEDIUM",
               "-f", "json", str(self.project_path)]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            if result.returncode == 0 or result.stdout:
                scan_results = json.loads(result.stdout) if result.stdout else {"Results": []}
                return self._parse_trivy_results(scan_results, ScanType.CONFIGURATION)
            else:
                return []
                
        except Exception as e:
            logger.error(f"Configuration scan failed: {e}")
            return []
    
    def _parse_trivy_results(self, results: Dict[str, Any], scan_type: ScanType) -> List[SecurityFinding]:
        """Parse Trivy scan results."""
        findings = []
        
        for result in results.get("Results", []):
            target = result.get("Target", "")
            
            # Parse vulnerabilities
            for vuln in result.get("Vulnerabilities", []):
                finding = SecurityFinding(
                    scan_type=scan_type,
                    severity=vuln.get("Severity", "UNKNOWN"),
                    title=vuln.get("Title", vuln.get("VulnerabilityID", "Unknown")),
                    description=vuln.get("Description", ""),
                    file_path=target,
                    cve_id=vuln.get("VulnerabilityID"),
                    recommendation=vuln.get("FixedVersion", "Update to latest version"),
                    metadata={
                        "package_name": vuln.get("PkgName"),
                        "installed_version": vuln.get("InstalledVersion"),
                        "fixed_version": vuln.get("FixedVersion"),
                        "references": vuln.get("References", [])
                    }
                )
                findings.append(finding)
            
            # Parse misconfigurations
            for misconfig in result.get("Misconfigurations", []):
                finding = SecurityFinding(
                    scan_type=scan_type,
                    severity=misconfig.get("Severity", "UNKNOWN"),
                    title=misconfig.get("Title", "Misconfiguration"),
                    description=misconfig.get("Description", ""),
                    file_path=target,
                    recommendation=misconfig.get("Resolution", "Fix configuration"),
                    metadata={
                        "type": misconfig.get("Type"),
                        "id": misconfig.get("ID"),
                        "avd_id": misconfig.get("AVDID")
                    }
                )
                findings.append(finding)
            
            # Parse secrets
            for secret in result.get("Secrets", []):
                finding = SecurityFinding(
                    scan_type=scan_type,
                    severity=secret.get("Severity", "HIGH"),
                    title=f"Secret detected: {secret.get('Title', 'Unknown')}",
                    description=f"Potential secret found in {target}",
                    file_path=target,
                    line_number=secret.get("StartLine"),
                    recommendation="Remove or encrypt the secret",
                    metadata={
                        "rule_id": secret.get("RuleID"),
                        "category": secret.get("Category"),
                        "match": secret.get("Match")
                    }
                )
                findings.append(finding)
        
        return findings
    
    def _parse_safety_results(self, results: List[Dict[str, Any]], req_file: str) -> List[SecurityFinding]:
        """Parse Safety scan results."""
        findings = []
        
        for vuln in results:
            finding = SecurityFinding(
                scan_type=ScanType.DEPENDENCIES,
                severity="HIGH",  # Safety typically reports high-severity vulnerabilities
                title=f"Vulnerable dependency: {vuln.get('package_name')}",
                description=vuln.get('advisory', ''),
                file_path=req_file,
                cve_id=vuln.get('cve'),
                recommendation=f"Update to version {vuln.get('safe_versions', 'latest')}",
                metadata={
                    "package_name": vuln.get('package_name'),
                    "installed_version": vuln.get('installed_version'),
                    "safe_versions": vuln.get('safe_versions'),
                    "vulnerability_id": vuln.get('vulnerability_id')
                }
            )
            findings.append(finding)
        
        return findings
    
    def _parse_bandit_results(self, results: Dict[str, Any]) -> List[SecurityFinding]:
        """Parse Bandit scan results."""
        findings = []
        
        for result in results.get("results", []):
            finding = SecurityFinding(
                scan_type=ScanType.CODE,
                severity=result.get("issue_severity", "MEDIUM"),
                title=result.get("test_name", "Security Issue"),
                description=result.get("issue_text", ""),
                file_path=result.get("filename"),
                line_number=result.get("line_number"),
                recommendation=result.get("more_info", "Review and fix the security issue"),
                metadata={
                    "test_id": result.get("test_id"),
                    "confidence": result.get("issue_confidence"),
                    "code": result.get("code")
                }
            )
            findings.append(finding)
        
        return findings
    
    def generate_sbom(self) -> Dict[str, Any]:
        """Generate Software Bill of Materials using Syft."""
        if not self._tool_available("syft"):
            raise RuntimeError("Syft not available for SBOM generation")
        
        config = self.tools_config["syft"]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = f.name
        
        try:
            cmd = [config["binary"]] + config["args"] + [str(self.project_path), "-o", f"json={output_file}"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            with open(output_file, 'r') as f:
                sbom_data = json.load(f)
            
            # Enhance SBOM with additional metadata
            enhanced_sbom = {
                "sbom_version": "1.0",
                "generation_timestamp": time.time(),
                "project_path": str(self.project_path),
                "tool": "syft",
                "data": sbom_data
            }
            
            return enhanced_sbom
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"SBOM generation failed: {e}")
        finally:
            # Clean up temporary file
            try:
                os.unlink(output_file)
            except FileNotFoundError:
                pass
    
    def verify_signatures(self, container_image: Optional[str] = None) -> Dict[str, Any]:
        """Verify container image signatures using Cosign."""
        if not self._tool_available("cosign"):
            raise RuntimeError("Cosign not available for signature verification")
        
        if not container_image:
            # Try to infer image name from project
            container_image = f"{self.project_path.name}:latest"
        
        config = self.tools_config["cosign"]
        
        try:
            cmd = [config["binary"]] + config["verify_args"] + [container_image]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            verification_result = {
                "image": container_image,
                "verified": result.returncode == 0,
                "timestamp": time.time(),
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None
            }
            
            return verification_result
            
        except Exception as e:
            return {
                "image": container_image,
                "verified": False,
                "timestamp": time.time(),
                "error": str(e)
            }
    
    def _check_compliance(self, report: SecurityReport) -> Dict[str, bool]:
        """Check compliance against security policies."""
        compliance = {}
        
        # Check finding thresholds
        critical_count = len(report.critical_findings)
        high_count = len(report.high_findings)
        medium_count = len([f for f in report.findings if f.severity.upper() == "MEDIUM"])
        
        compliance["critical_findings"] = critical_count <= self.policies["max_critical_findings"]
        compliance["high_findings"] = high_count <= self.policies["max_high_findings"]
        compliance["medium_findings"] = medium_count <= self.policies["max_medium_findings"]
        
        # Check required scans
        completed_scans = set(f.scan_type for f in report.findings)
        required_scans = set(self.policies["required_scans"])
        compliance["required_scans"] = required_scans.issubset(completed_scans)
        
        # Check SBOM requirement
        if self.policies.get("sbom_required"):
            compliance["sbom_generated"] = report.sbom is not None
        
        # Overall compliance
        compliance["overall"] = all(compliance.values())
        
        return compliance
    
    def _calculate_metrics(self, report: SecurityReport) -> Dict[str, Union[int, float]]:
        """Calculate security metrics."""
        findings_by_severity = {}
        findings_by_type = {}
        
        for finding in report.findings:
            # Count by severity
            severity = finding.severity.upper()
            findings_by_severity[severity] = findings_by_severity.get(severity, 0) + 1
            
            # Count by scan type
            scan_type = finding.scan_type.value
            findings_by_type[scan_type] = findings_by_type.get(scan_type, 0) + 1
        
        metrics = {
            "total_findings": report.total_findings,
            "risk_score": report.risk_score,
            "scan_duration": time.time() - report.scan_timestamp,
            **{f"{k.lower()}_findings": v for k, v in findings_by_severity.items()},
            **{f"{k}_findings": v for k, v in findings_by_type.items()}
        }
        
        return metrics
    
    def generate_security_report(self, report: SecurityReport, output_format: str = "json") -> str:
        """Generate formatted security report."""
        if output_format == "json":
            return self._generate_json_report(report)
        elif output_format == "html":
            return self._generate_html_report(report)
        elif output_format == "sarif":
            return self._generate_sarif_report(report)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _generate_json_report(self, report: SecurityReport) -> str:
        """Generate JSON security report."""
        report_data = {
            "scan_info": {
                "timestamp": report.scan_timestamp,
                "project_path": report.project_path,
                "security_level": report.security_level.value,
                "total_findings": report.total_findings,
                "risk_score": report.risk_score
            },
            "compliance": report.compliance_status,
            "metrics": report.metrics,
            "findings": [
                {
                    "scan_type": f.scan_type.value,
                    "severity": f.severity,
                    "title": f.title,
                    "description": f.description,
                    "file_path": f.file_path,
                    "line_number": f.line_number,
                    "cve_id": f.cve_id,
                    "recommendation": f.recommendation,
                    "metadata": f.metadata
                }
                for f in report.findings
            ]
        }
        
        if report.sbom:
            report_data["sbom"] = report.sbom
        
        return json.dumps(report_data, indent=2, default=str)
    
    def _generate_html_report(self, report: SecurityReport) -> str:
        """Generate HTML security report."""
        # Simplified HTML report template
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Security Report - {project_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e0e0e0; border-radius: 3px; }}
        .finding {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }}
        .critical {{ border-left-color: #d32f2f; }}
        .high {{ border-left-color: #ff9800; }}
        .medium {{ border-left-color: #ffc107; }}
        .low {{ border-left-color: #4caf50; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Security Report</h1>
        <p><strong>Project:</strong> {project_name}</p>
        <p><strong>Security Level:</strong> {security_level}</p>
        <p><strong>Scan Date:</strong> {scan_date}</p>
        <p><strong>Risk Score:</strong> {risk_score}/10</p>
    </div>
    
    <h2>Metrics</h2>
    <div class="metrics">
        <div class="metric">Total Findings: {total_findings}</div>
        <div class="metric">Critical: {critical_count}</div>
        <div class="metric">High: {high_count}</div>
        <div class="metric">Medium: {medium_count}</div>
    </div>
    
    <h2>Findings</h2>
    {findings_html}
</body>
</html>
        """
        
        findings_html = ""
        for finding in report.findings:
            severity_class = finding.severity.lower()
            findings_html += f"""
            <div class="finding {severity_class}">
                <h3>{finding.title}</h3>
                <p><strong>Severity:</strong> {finding.severity}</p>
                <p><strong>Type:</strong> {finding.scan_type.value}</p>
                {f'<p><strong>File:</strong> {finding.file_path}</p>' if finding.file_path else ''}
                {f'<p><strong>Line:</strong> {finding.line_number}</p>' if finding.line_number else ''}
                <p><strong>Description:</strong> {finding.description}</p>
                {f'<p><strong>Recommendation:</strong> {finding.recommendation}</p>' if finding.recommendation else ''}
            </div>
            """
        
        return html_template.format(
            project_name=Path(report.project_path).name,
            security_level=report.security_level.value,
            scan_date=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(report.scan_timestamp)),
            risk_score=report.risk_score,
            total_findings=report.total_findings,
            critical_count=len(report.critical_findings),
            high_count=len(report.high_findings),
            medium_count=len([f for f in report.findings if f.severity.upper() == "MEDIUM"]),
            findings_html=findings_html
        )
    
    def _generate_sarif_report(self, report: SecurityReport) -> str:
        """Generate SARIF format report for GitHub Security."""
        sarif_rules = {}
        sarif_results = []
        
        for finding in report.findings:
            rule_id = f"{finding.scan_type.value}_{finding.title.replace(' ', '_').lower()}"
            
            # Add rule if not exists
            if rule_id not in sarif_rules:
                sarif_rules[rule_id] = {
                    "id": rule_id,
                    "shortDescription": {"text": finding.title},
                    "fullDescription": {"text": finding.description},
                    "defaultConfiguration": {
                        "level": "error" if finding.severity.upper() in ["CRITICAL", "HIGH"] else "warning"
                    }
                }
            
            # Add result
            result = {
                "ruleId": rule_id,
                "message": {"text": finding.description},
                "level": "error" if finding.severity.upper() in ["CRITICAL", "HIGH"] else "warning"
            }
            
            if finding.file_path:
                result["locations"] = [{
                    "physicalLocation": {
                        "artifactLocation": {"uri": finding.file_path},
                        "region": {"startLine": finding.line_number or 1}
                    }
                }]
            
            sarif_results.append(result)
        
        sarif_report = {
            "version": "2.1.0",
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "runs": [{
                "tool": {
                    "driver": {
                        "name": "MLX Security Scanner",
                        "version": "1.0.0",
                        "informationUri": "https://github.com/mlx/security-scanner",
                        "rules": list(sarif_rules.values())
                    }
                },
                "results": sarif_results
            }]
        }
        
        return json.dumps(sarif_report, indent=2)
    
    def create_security_baseline(self) -> Dict[str, Any]:
        """Create security baseline for the project."""
        baseline_report = self.run_comprehensive_scan()
        
        baseline = {
            "created_at": time.time(),
            "project_path": str(self.project_path),
            "security_level": self.security_level.value,
            "baseline_metrics": baseline_report.metrics,
            "allowed_findings": [
                {
                    "scan_type": f.scan_type.value,
                    "severity": f.severity,
                    "title": f.title,
                    "file_path": f.file_path,
                    "checksum": hashlib.sha256(
                        f"{f.scan_type.value}{f.title}{f.file_path or ''}".encode()
                    ).hexdigest()[:16]
                }
                for f in baseline_report.findings
            ]
        }
        
        # Save baseline
        baseline_file = self.project_path / ".security_baseline.json"
        with open(baseline_file, 'w') as f:
            json.dump(baseline, f, indent=2)
        
        logger.info(f"Security baseline created with {len(baseline['allowed_findings'])} allowed findings")
        return baseline
    
    def compare_with_baseline(self, current_report: SecurityReport) -> Dict[str, Any]:
        """Compare current scan with security baseline."""
        baseline_file = self.project_path / ".security_baseline.json"
        
        if not baseline_file.exists():
            return {"status": "no_baseline", "message": "No security baseline found"}
        
        with open(baseline_file) as f:
            baseline = json.load(f)
        
        # Create checksums for current findings
        current_checksums = set()
        for f in current_report.findings:
            checksum = hashlib.sha256(
                f"{f.scan_type.value}{f.title}{f.file_path or ''}".encode()
            ).hexdigest()[:16]
            current_checksums.add(checksum)
        
        # Create checksums for baseline findings
        baseline_checksums = set(f["checksum"] for f in baseline["allowed_findings"])
        
        # Find new and resolved findings
        new_findings = current_checksums - baseline_checksums
        resolved_findings = baseline_checksums - current_checksums
        
        comparison = {
            "status": "compared",
            "baseline_date": baseline["created_at"],
            "current_date": current_report.scan_timestamp,
            "baseline_findings": len(baseline["allowed_findings"]),
            "current_findings": current_report.total_findings,
            "new_findings": len(new_findings),
            "resolved_findings": len(resolved_findings),
            "regression_detected": len(new_findings) > 0
        }
        
        return comparison


# CLI interface for security hardening
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Security Hardening Framework")
    parser.add_argument("command", choices=[
        "scan", "sbom", "verify", "baseline", "compare", "report"
    ])
    parser.add_argument("--project-path", default=".", help="Project path to scan")
    parser.add_argument("--security-level", choices=[l.value for l in SecurityLevel], 
                       default="enhanced", help="Security level")
    parser.add_argument("--output-format", choices=["json", "html", "sarif"], 
                       default="json", help="Output format")
    parser.add_argument("--output-file", help="Output file path")
    parser.add_argument("--container-image", help="Container image for verification")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    framework = SecurityHardeningFramework(
        Path(args.project_path), 
        SecurityLevel(args.security_level)
    )
    
    try:
        if args.command == "scan":
            report = framework.run_comprehensive_scan()
            output = framework.generate_security_report(report, args.output_format)
            
            if args.output_file:
                with open(args.output_file, 'w') as f:
                    f.write(output)
                print(f"Security report saved to: {args.output_file}")
            else:
                print(output)
        
        elif args.command == "sbom":
            sbom = framework.generate_sbom()
            output = json.dumps(sbom, indent=2)
            
            if args.output_file:
                with open(args.output_file, 'w') as f:
                    f.write(output)
                print(f"SBOM saved to: {args.output_file}")
            else:
                print(output)
        
        elif args.command == "verify":
            result = framework.verify_signatures(args.container_image)
            print(json.dumps(result, indent=2))
        
        elif args.command == "baseline":
            baseline = framework.create_security_baseline()
            print(f"Security baseline created with {len(baseline['allowed_findings'])} findings")
        
        elif args.command == "compare":
            report = framework.run_comprehensive_scan()
            comparison = framework.compare_with_baseline(report)
            print(json.dumps(comparison, indent=2))
        
        elif args.command == "report":
            report = framework.run_comprehensive_scan()
            output = framework.generate_security_report(report, args.output_format)
            
            if args.output_file:
                with open(args.output_file, 'w') as f:
                    f.write(output)
                print(f"Report saved to: {args.output_file}")
            else:
                print(output)
    
    except Exception as e:
        logger.error(f"Command failed: {e}")
        exit(1) 