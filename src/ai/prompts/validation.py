"""Prompt validation utilities."""

import logging
import re
from dataclasses import dataclass
from enum import Enum

from .base import PromptTemplate


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""

    severity: ValidationSeverity
    message: str
    location: str | None = None
    suggestion: str | None = None


@dataclass
class ValidationResult:
    """Result of prompt validation."""

    is_valid: bool
    issues: list[ValidationIssue]
    quality_score: float
    recommendations: list[str]


class PromptValidator:
    """Validator for prompt quality and best practices."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Common problematic patterns
        self.problematic_patterns = {
            r"\b(always|never|everyone|nobody)\b": "Avoid absolute statements",
            r"\b(obviously|clearly|simply)\b": "Avoid assuming clarity",
            r"\?{2,}": "Avoid multiple question marks",
            r"!{2,}": "Avoid multiple exclamation marks",
            r"[A-Z]{4,}": "Avoid excessive capitalization",
        }

        # Bias indicators
        self.bias_indicators = {
            "gender": [
                r"\b(he|him|his)\b(?!\s+(or|/)\s+(she|her))",
                r"\b(she|her|hers)\b(?!\s+(or|/)\s+(he|him))",
                r"\b(guys|dudes|bros)\b",
                r"\b(girls|ladies)\b",
            ],
            "cultural": [
                r"\b(normal|typical|standard)\s+(person|people|family)\b",
                r"\b(exotic|foreign|weird)\b",
            ],
            "age": [
                r"\b(young|old)\s+(people|person)\b",
                r"\b(millennials|boomers|gen-z)\b",
            ],
        }

    def validate_prompt(self, prompt: str) -> ValidationResult:
        """Perform comprehensive validation of a prompt."""
        issues = []

        # Basic validation
        issues.extend(self._validate_basic_structure(prompt))
        issues.extend(self._validate_clarity(prompt))
        issues.extend(self._validate_length(prompt))
        issues.extend(self._validate_language_quality(prompt))
        issues.extend(self._validate_bias(prompt))
        issues.extend(self._validate_security(prompt))

        # Calculate quality score
        quality_score = self._calculate_quality_score(prompt, issues)

        # Generate recommendations
        recommendations = self._generate_recommendations(prompt, issues)

        # Determine if valid (no errors)
        is_valid = not any(
            issue.severity == ValidationSeverity.ERROR for issue in issues
        )

        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            quality_score=quality_score,
            recommendations=recommendations,
        )

    def validate_template(self, template: PromptTemplate) -> ValidationResult:
        """Validate a prompt template."""
        issues = []

        # Validate template structure
        issues.extend(self._validate_template_structure(template))

        # Validate base prompt
        base_result = self.validate_prompt(template.template)
        issues.extend(base_result.issues)

        # Template-specific validations
        issues.extend(self._validate_template_variables(template))
        issues.extend(self._validate_template_examples(template))

        quality_score = self._calculate_quality_score(template.template, issues)
        recommendations = self._generate_recommendations(template.template, issues)

        is_valid = not any(
            issue.severity == ValidationSeverity.ERROR for issue in issues
        )

        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            quality_score=quality_score,
            recommendations=recommendations,
        )

    def _validate_basic_structure(self, prompt: str) -> list[ValidationIssue]:
        """Validate basic prompt structure."""
        issues = []

        if not prompt.strip():
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message="Prompt is empty",
                    suggestion="Add content to the prompt",
                )
            )
            return issues

        # Check for clear task/instruction
        if not any(
            word in prompt.lower()
            for word in ["please", "write", "generate", "create", "explain", "analyze"]
        ):
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="Prompt lacks clear action verbs",
                    suggestion="Add clear instructions like 'Please write', 'Generate', or 'Explain'",
                )
            )

        # Check for question structure
        if "?" not in prompt and not any(
            word in prompt.lower() for word in ["what", "how", "why", "when", "where"]
        ):
            if not any(
                word in prompt.lower()
                for word in ["please", "write", "generate", "create"]
            ):
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message="Prompt could benefit from question format or clear instruction",
                        suggestion="Consider rephrasing as a question or adding 'Please' to instructions",
                    )
                )

        return issues

    def _validate_clarity(self, prompt: str) -> list[ValidationIssue]:
        """Validate prompt clarity."""
        issues = []

        # Check for ambiguous words
        ambiguous_words = ["thing", "stuff", "something", "anything", "it"]
        for word in ambiguous_words:
            if f" {word} " in prompt.lower():
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Ambiguous word '{word}' found",
                        suggestion=f"Replace '{word}' with more specific terms",
                    )
                )

        # Check for overly complex sentences
        sentences = [s.strip() for s in prompt.split(".") if s.strip()]
        for sentence in sentences:
            word_count = len(sentence.split())
            if word_count > 40:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Very long sentence ({word_count} words)",
                        location=sentence[:50] + "...",
                        suggestion="Break into shorter sentences for clarity",
                    )
                )

        return issues

    def _validate_length(self, prompt: str) -> list[ValidationIssue]:
        """Validate prompt length."""
        issues = []

        word_count = len(prompt.split())
        char_count = len(prompt)

        if word_count < 5:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Prompt is very short ({word_count} words)",
                    suggestion="Consider adding more context or instructions",
                )
            )
        elif word_count > 1000:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Prompt is very long ({word_count} words)",
                    suggestion="Consider breaking into smaller prompts or using templates",
                )
            )

        # Estimate token count (rough approximation)
        estimated_tokens = char_count // 4
        if estimated_tokens > 2000:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Prompt may exceed token limits (~{estimated_tokens} tokens)",
                    suggestion="Shorten prompt or use summarization",
                )
            )

        return issues

    def _validate_language_quality(self, prompt: str) -> list[ValidationIssue]:
        """Validate language quality and style."""
        issues = []

        # Check problematic patterns
        for pattern, message in self.problematic_patterns.items():
            matches = re.finditer(pattern, prompt, re.IGNORECASE)
            for match in matches:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=message,
                        location=match.group(),
                        suggestion="Consider rewording for better clarity",
                    )
                )

        # Check for passive voice (simple heuristic)
        passive_indicators = re.findall(r"\b(is|are|was|were|been)\s+\w+ed\b", prompt)
        if len(passive_indicators) > 3:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message="Multiple instances of passive voice detected",
                    suggestion="Consider using active voice for clearer instructions",
                )
            )

        return issues

    def _validate_bias(self, prompt: str) -> list[ValidationIssue]:
        """Check for potential bias in prompt."""
        issues = []

        for bias_type, patterns in self.bias_indicators.items():
            for pattern in patterns:
                matches = re.finditer(pattern, prompt, re.IGNORECASE)
                for match in matches:
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message=f"Potential {bias_type} bias detected",
                            location=match.group(),
                            suggestion=f"Consider more inclusive language for {bias_type}",
                        )
                    )

        return issues

    def _validate_security(self, prompt: str) -> list[ValidationIssue]:
        """Check for potential security issues."""
        issues = []

        # Check for injection patterns
        injection_patterns = [
            r"ignore\s+(previous|all)\s+instructions",
            r"forget\s+(everything|all)",
            r"act\s+as\s+(if|though)",
            r"pretend\s+(to\s+be|you\s+are)",
            r"jailbreak",
            r"developer\s+mode",
        ]

        for pattern in injection_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message="Potential prompt injection detected",
                        suggestion="Remove instructions that could override system behavior",
                    )
                )

        # Check for sensitive information requests
        sensitive_patterns = [
            r"password",
            r"api\s+key",
            r"secret",
            r"token",
            r"credit\s+card",
            r"ssn|social\s+security",
        ]

        for pattern in sensitive_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message="Prompt requests potentially sensitive information",
                        suggestion="Ensure appropriate handling of sensitive data",
                    )
                )

        return issues

    def _validate_template_structure(
        self, template: PromptTemplate
    ) -> list[ValidationIssue]:
        """Validate template-specific structure."""
        issues = []

        if not template.name:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message="Template missing name",
                    suggestion="Add a descriptive name to the template",
                )
            )

        if not template.variables and "{" in template.template:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="Template contains variables but none are declared",
                    suggestion="Declare variables in the template",
                )
            )

        return issues

    def _validate_template_variables(
        self, template: PromptTemplate
    ) -> list[ValidationIssue]:
        """Validate template variables."""
        issues = []

        # Find variables in template
        template_vars = set(re.findall(r"\{([^}]+)\}", template.template))
        declared_vars = set(template.variables)

        # Check for undeclared variables
        undeclared = template_vars - declared_vars
        for var in undeclared:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Variable '{var}' used but not declared",
                    suggestion=f"Add '{var}' to template variables",
                )
            )

        # Check for declared but unused variables
        unused = declared_vars - template_vars
        for var in unused:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Variable '{var}' declared but not used",
                    suggestion=f"Remove '{var}' from variables or use in template",
                )
            )

        return issues

    def _validate_template_examples(
        self, template: PromptTemplate
    ) -> list[ValidationIssue]:
        """Validate template examples."""
        issues = []

        if template.prompt_type.value == "few_shot" and not template.examples:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="Few-shot template has no examples",
                    suggestion="Add examples to demonstrate the expected format",
                )
            )

        for i, example in enumerate(template.examples):
            if not example.input or not example.output:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Example {i + 1} missing input or output",
                        suggestion="Ensure all examples have both input and output",
                    )
                )

        return issues

    def _calculate_quality_score(
        self, prompt: str, issues: list[ValidationIssue]
    ) -> float:
        """Calculate overall quality score (0-1)."""
        base_score = 1.0

        # Deduct points for issues
        for issue in issues:
            if issue.severity == ValidationSeverity.ERROR:
                base_score -= 0.3
            elif issue.severity == ValidationSeverity.WARNING:
                base_score -= 0.1
            else:  # INFO
                base_score -= 0.01

        # Add points for good practices
        word_count = len(prompt.split())
        if 10 <= word_count <= 200:  # Good length
            base_score += 0.1

        if any(word in prompt.lower() for word in ["please", "specific", "detailed"]):
            base_score += 0.1

        return max(0.0, min(1.0, base_score))

    def _generate_recommendations(
        self, prompt: str, issues: list[ValidationIssue]
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Collect suggestions from issues
        for issue in issues:
            if issue.suggestion:
                recommendations.append(issue.suggestion)

        # Add general recommendations
        if len(prompt.split()) < 10:
            recommendations.append("Add more context and specific instructions")

        if "?" not in prompt and "please" not in prompt.lower():
            recommendations.append("Make instructions more polite and clear")

        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)

        return unique_recommendations[:5]  # Limit to top 5
