#!/usr/bin/env python3
"""
🎯 MLX AI Response Evaluation System

Production-ready evaluation framework for Mlx Platform Foundation AI responses.
Repository infrastructure code for evaluating AI assistant quality.

Key Features:
- 5-dimensional scoring criteria (MLX Accuracy, Actionability, Context Awareness, Production Readiness, UX)
- Comprehensive benchmark dataset covering all MLX capabilities
- Automated regression testing and performance tracking
- A/B testing framework for prompt optimization
- Real-time analytics dashboard with trend analysis
"""

import json
import logging
import re
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class EvaluationCriteria:
    """MLX AI Response Evaluation Criteria (Production Grade)"""

    # Mlx Platform Accuracy (30%)
    mlx_platform_accuracy: float = 0.0
    command_accuracy: float = 0.0
    framework_integration: float = 0.0
    platform_specificity: float = 0.0

    # Actionability (25%)
    actionability: float = 0.0
    executable_commands: float = 0.0
    step_by_step_clarity: float = 0.0
    parameter_specificity: float = 0.0

    # Context Awareness (20%)
    context_awareness: float = 0.0
    project_state_utilization: float = 0.0
    framework_cross_references: float = 0.0
    personalization: float = 0.0

    # Production Readiness (15%)
    production_readiness: float = 0.0
    error_handling: float = 0.0
    security_considerations: float = 0.0
    monitoring_guidance: float = 0.0

    # User Experience (10%)
    user_experience: float = 0.0
    clarity_formatting: float = 0.0
    tone_professionalism: float = 0.0
    appropriate_detail_level: float = 0.0

    def calculate_weighted_score(self) -> float:
        """Calculate final weighted score (0-100)"""
        return (
            (self.mlx_platform_accuracy * 0.30)
            + (self.actionability * 0.25)
            + (self.context_awareness * 0.20)
            + (self.production_readiness * 0.15)
            + (self.user_experience * 0.10)
        ) * 100


@dataclass
class AIResponseEvaluation:
    """Complete AI response evaluation result"""

    evaluation_id: str
    timestamp: datetime
    user_query: str
    ai_response: str
    project_context: dict[str, Any]

    # Evaluation scores
    criteria: EvaluationCriteria
    final_score: float = 0.0
    grade: str = ""

    # Detailed analysis
    command_accuracy_details: dict[str, Any] = field(default_factory=dict)
    framework_coverage: list[str] = field(default_factory=list)
    missing_elements: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    improvement_areas: list[str] = field(default_factory=list)

    # Performance metrics
    response_time: float = 0.0
    token_usage: dict[str, int] = field(default_factory=dict)
    cost_estimate: float = 0.0

    def __post_init__(self):
        self.final_score = self.criteria.calculate_weighted_score()
        self.grade = self._calculate_grade()

    def _calculate_grade(self) -> str:
        """Convert numeric score to letter grade"""
        if self.final_score >= 95:
            return "A+"
        elif self.final_score >= 90:
            return "A"
        elif self.final_score >= 85:
            return "A-"
        elif self.final_score >= 80:
            return "B+"
        elif self.final_score >= 75:
            return "B"
        elif self.final_score >= 70:
            return "B-"
        elif self.final_score >= 65:
            return "C+"
        elif self.final_score >= 60:
            return "C"
        else:
            return "F"


class MLXCommandValidator:
    """Validates MLX-specific commands and syntax"""

    def __init__(self):
        # mlx command patterns from codebase analysis
        self.valid_commands = {
            "mlx assistant golden-repos": [
                "list",
                "create",
                "validate",
                "create-all",
                "validate-all",
            ],
            "mlx assistant security": [
                "scan",
                "sbom",
                "verify",
                "baseline",
                "compare",
                "report",
            ],
            "mlx assistant plugins": ["create", "validate", "list", "info"],
            "mlx assistant glossary": ["view", "search", "validate-naming"],
            "mlx assistant": [
                "doctor",
                "analyze",
                "quick-start",
                "ask",
                "ai-analyze",
                "ai-workflow",
            ],
        }

        self.security_levels = ["basic", "enhanced", "enterprise", "critical"]
        self.repo_specs = [
            "minimal",
            "standard",
            "advanced",
            "plugin_heavy",
            "performance",
        ]
        self.plugin_types = [
            "ml_framework",
            "data_processor",
            "model_provider",
            "deployment",
            "monitoring",
            "security",
            "utility",
        ]

    def validate_command_accuracy(self, response: str) -> dict[str, Any]:
        """Validate mlx command accuracy in AI response"""
        results = {
            "total_commands": 0,
            "correct_commands": 0,
            "incorrect_commands": [],
            "missing_parameters": [],
            "correct_security_levels": 0,
            "correct_repo_specs": 0,
            "framework_coverage": set(),
        }

        # Extract MLX commands from response
        mlx_commands = re.findall(
            r"mlx\s+assistant\s+[\w\-]+(?:\s+[\w\-]+)*(?:\s+--[\w\-]+\s+[\w\-]+)*",
            response,
        )

        for cmd in mlx_commands:
            results["total_commands"] += 1
            cmd_parts = cmd.split()

            # Check base command structure
            if (
                len(cmd_parts) >= 3
                and cmd_parts[0] == "mlx"
                and cmd_parts[1] == "assistant"
            ):
                framework = cmd_parts[2]

                # Validate framework
                base_cmd = f"mlx assistant {framework}"
                if base_cmd in self.valid_commands:
                    results["framework_coverage"].add(framework)

                    # Validate subcommand
                    if len(cmd_parts) > 3:
                        subcommand = cmd_parts[3]
                        if subcommand in self.valid_commands[base_cmd]:
                            results["correct_commands"] += 1

                            # Validate parameters
                            self._validate_parameters(
                                cmd, framework, subcommand, results
                            )
                        else:
                            results["incorrect_commands"].append(
                                f"Invalid subcommand: {cmd}"
                            )
                    else:
                        results["correct_commands"] += 1
                else:
                    results["incorrect_commands"].append(f"Invalid framework: {cmd}")
            else:
                results["incorrect_commands"].append(
                    f"Invalid command structure: {cmd}"
                )

        # Calculate accuracy percentage
        if results["total_commands"] > 0:
            results["accuracy_percentage"] = (
                results["correct_commands"] / results["total_commands"]
            ) * 100
        else:
            results["accuracy_percentage"] = 0

        results["framework_coverage"] = list(results["framework_coverage"])
        return results

    def _validate_parameters(
        self, cmd: str, framework: str, subcommand: str, results: dict[str, Any]
    ):
        """Validate command parameters"""
        if framework == "security" and subcommand == "scan":
            if "--level" in cmd:
                level_match = re.search(r"--level\s+([\w\-]+)", cmd)
                if level_match and level_match.group(1) in self.security_levels:
                    results["correct_security_levels"] += 1
                else:
                    results["missing_parameters"].append(
                        f"Invalid security level in: {cmd}"
                    )

        elif framework == "golden-repos" and subcommand == "create":
            # Check for repo spec parameter
            spec_found = False
            for spec in self.repo_specs:
                if spec in cmd:
                    results["correct_repo_specs"] += 1
                    spec_found = True
                    break
            if not spec_found:
                results["missing_parameters"].append(f"Missing repo spec in: {cmd}")


class MLXFrameworkAnalyzer:
    """Analyzes framework integration and coverage"""

    def __init__(self):
        self.frameworks = {
            "golden_repos": {
                "keywords": [
                    "golden",
                    "repository",
                    "template",
                    "reference",
                    "component",
                ],
                "commands": ["create", "validate", "extract"],
                "concepts": ["minimal", "standard", "advanced", "testing"],
            },
            "security": {
                "keywords": ["security", "scan", "vulnerability", "hardening", "sbom"],
                "commands": ["scan", "verify", "baseline", "report"],
                "concepts": ["enhanced", "enterprise", "critical", "compliance"],
            },
            "plugins": {
                "keywords": ["plugin", "ecosystem", "development", "validation"],
                "commands": ["create", "validate", "list"],
                "concepts": ["ml_framework", "data_processor", "utility"],
            },
            "glossary": {
                "keywords": ["glossary", "terminology", "naming", "standards"],
                "commands": ["view", "search", "validate-naming"],
                "concepts": ["documentation", "conventions", "definitions"],
            },
        }

    def analyze_framework_integration(
        self, response: str, query: str
    ) -> dict[str, Any]:
        """Analyze how well response integrates MLX frameworks"""
        analysis = {
            "frameworks_mentioned": [],
            "integration_quality": 0.0,
            "cross_references": 0,
            "framework_completeness": {},
        }

        response_lower = response.lower()
        query.lower()

        for framework_name, framework_data in self.frameworks.items():
            mentioned = False
            completeness_score = 0

            # Check keyword presence
            keyword_matches = sum(
                1 for keyword in framework_data["keywords"] if keyword in response_lower
            )
            if keyword_matches > 0:
                mentioned = True
                completeness_score += (
                    min(keyword_matches / len(framework_data["keywords"]), 1.0) * 40
                )

            # Check command usage
            command_matches = sum(
                1 for cmd in framework_data["commands"] if cmd in response_lower
            )
            if command_matches > 0:
                completeness_score += (
                    min(command_matches / len(framework_data["commands"]), 1.0) * 30
                )

            # Check concept understanding
            concept_matches = sum(
                1 for concept in framework_data["concepts"] if concept in response_lower
            )
            if concept_matches > 0:
                completeness_score += (
                    min(concept_matches / len(framework_data["concepts"]), 1.0) * 30
                )

            if mentioned:
                analysis["frameworks_mentioned"].append(framework_name)
                analysis["framework_completeness"][framework_name] = completeness_score

        # Calculate integration quality
        if analysis["frameworks_mentioned"]:
            avg_completeness = statistics.mean(
                analysis["framework_completeness"].values()
            )
            framework_coverage = len(analysis["frameworks_mentioned"]) / len(
                self.frameworks
            )
            analysis["integration_quality"] = (
                avg_completeness + framework_coverage * 100
            ) / 2

        # Count cross-references (mentions of multiple frameworks)
        if len(analysis["frameworks_mentioned"]) > 1:
            analysis["cross_references"] = len(analysis["frameworks_mentioned"]) - 1

        return analysis


class ProductionReadinessAnalyzer:
    """Analyzes production readiness aspects of AI responses"""

    def analyze_production_readiness(self, response: str, query: str) -> dict[str, Any]:
        """Analyze production readiness indicators"""
        analysis = {
            "error_handling_mentioned": False,
            "security_considerations": False,
            "monitoring_guidance": False,
            "best_practices": [],
            "production_patterns": [],
            "completeness_score": 0.0,
        }

        response_lower = response.lower()

        # Error handling indicators
        error_patterns = [
            "error",
            "exception",
            "fail",
            "fallback",
            "retry",
            "timeout",
            "validation",
            "check",
            "verify",
            "handle",
        ]
        if any(pattern in response_lower for pattern in error_patterns):
            analysis["error_handling_mentioned"] = True
            analysis["completeness_score"] += 30

        # Security considerations
        security_patterns = [
            "security",
            "secure",
            "vulnerability",
            "authentication",
            "authorization",
            "encrypt",
            "permission",
            "access",
            "credential",
        ]
        if any(pattern in response_lower for pattern in security_patterns):
            analysis["security_considerations"] = True
            analysis["completeness_score"] += 35

        # Monitoring and maintenance
        monitoring_patterns = [
            "monitor",
            "log",
            "track",
            "metric",
            "alert",
            "dashboard",
            "health",
            "status",
            "maintenance",
            "update",
        ]
        if any(pattern in response_lower for pattern in monitoring_patterns):
            analysis["monitoring_guidance"] = True
            analysis["completeness_score"] += 35

        # Best practices
        best_practice_patterns = [
            "best practice",
            "recommended",
            "convention",
            "standard",
            "guideline",
            "principle",
            "pattern",
        ]
        for pattern in best_practice_patterns:
            if pattern in response_lower:
                analysis["best_practices"].append(pattern)

        return analysis


class AIResponseEvaluator:
    """Production-grade AI response evaluation system"""

    def __init__(self, config_path: Path | None = None):
        self.config = self._load_config(config_path)
        self.command_validator = MLXCommandValidator()
        self.framework_analyzer = MLXFrameworkAnalyzer()
        self.production_analyzer = ProductionReadinessAnalyzer()

        # Initialize evaluation storage
        self.evaluations_dir = Path("data/evaluations")
        self.evaluations_dir.mkdir(parents=True, exist_ok=True)

        # Performance tracking
        self.evaluation_history = []
        self.benchmark_results = {}

    def _load_config(self, config_path: Path | None) -> dict[str, Any]:
        """Load evaluation configuration"""
        default_config = {
            "scoring_weights": {
                "mlx_platform_accuracy": 0.30,
                "actionability": 0.25,
                "context_awareness": 0.20,
                "production_readiness": 0.15,
                "user_experience": 0.10,
            },
            "performance_thresholds": {
                "command_accuracy": 0.90,
                "response_time": 2.0,
                "user_satisfaction": 0.85,
                "framework_coverage": 0.75,
            },
            "benchmark_categories": [
                "security_workflows",
                "plugin_development",
                "golden_repos",
                "troubleshooting",
                "optimization",
                "integration",
            ],
        }

        if config_path and config_path.exists():
            with open(config_path) as f:
                user_config = json.load(f)
                default_config.update(user_config)

        return default_config

    async def evaluate_response(
        self,
        user_query: str,
        ai_response: str,
        project_context: dict[str, Any] = None,
        response_time: float = 0.0,
        token_usage: dict[str, int] = None,
    ) -> AIResponseEvaluation:
        """Comprehensive AI response evaluation"""

        evaluation_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        logger.info(
            f"Starting evaluation {evaluation_id} for query: {user_query[:100]}..."
        )

        # Initialize evaluation
        evaluation = AIResponseEvaluation(
            evaluation_id=evaluation_id,
            timestamp=datetime.now(),
            user_query=user_query,
            ai_response=ai_response,
            project_context=project_context or {},
            criteria=EvaluationCriteria(),
            response_time=response_time,
            token_usage=token_usage or {},
        )

        # Perform detailed analysis
        await self._evaluate_mlx_platform_accuracy(evaluation)
        await self._evaluate_actionability(evaluation)
        await self._evaluate_context_awareness(evaluation)
        await self._evaluate_production_readiness(evaluation)
        await self._evaluate_user_experience(evaluation)

        # Calculate final scores
        evaluation.final_score = evaluation.criteria.calculate_weighted_score()
        evaluation.grade = evaluation._calculate_grade()

        # Generate insights
        await self._generate_evaluation_insights(evaluation)

        # Store evaluation
        await self._store_evaluation(evaluation)

        processing_time = time.time() - start_time
        logger.info(
            f"Evaluation {evaluation_id} completed in {processing_time:.2f}s - Grade: {evaluation.grade}"
        )

        return evaluation

    async def _evaluate_mlx_platform_accuracy(self, evaluation: AIResponseEvaluation):
        """Evaluate mlx platform-specific accuracy (30% weight)"""

        # Command accuracy analysis
        cmd_analysis = self.command_validator.validate_command_accuracy(
            evaluation.ai_response
        )
        evaluation.command_accuracy_details = cmd_analysis
        evaluation.criteria.command_accuracy = cmd_analysis["accuracy_percentage"] / 100

        # Framework integration analysis
        framework_analysis = self.framework_analyzer.analyze_framework_integration(
            evaluation.ai_response, evaluation.user_query
        )
        evaluation.framework_coverage = framework_analysis["frameworks_mentioned"]
        evaluation.criteria.framework_integration = (
            framework_analysis["integration_quality"] / 100
        )

        # Platform specificity (MLX-specific vs generic advice)
        mlx_keywords = [
            "mlx",
            "golden",
            "security hardening",
            "plugin ecosystem",
            "glossary",
        ]
        specificity_score = sum(
            1
            for keyword in mlx_keywords
            if keyword.lower() in evaluation.ai_response.lower()
        )
        evaluation.criteria.platform_specificity = min(
            specificity_score / len(mlx_keywords), 1.0
        )

        # Overall mlx platform accuracy
        evaluation.criteria.mlx_platform_accuracy = (
            evaluation.criteria.command_accuracy * 0.5
            + evaluation.criteria.framework_integration * 0.3
            + evaluation.criteria.platform_specificity * 0.2
        )

    async def _evaluate_actionability(self, evaluation: AIResponseEvaluation):
        """Evaluate actionability of the response (25% weight)"""

        response = evaluation.ai_response

        # Executable commands
        command_count = len(re.findall(r"mlx\s+assistant\s+[\w\-]+", response))
        bash_commands = len(re.findall(r"```bash\n(.*?)```", response, re.DOTALL))
        total_commands = command_count + bash_commands
        evaluation.criteria.executable_commands = min(
            total_commands / 3, 1.0
        )  # Expect ~3 commands for good actionability

        # Step-by-step clarity
        step_indicators = len(
            re.findall(
                r"(\d+\.|step \d+|first|second|third|then|next|finally)",
                response.lower(),
            )
        )
        evaluation.criteria.step_by_step_clarity = min(step_indicators / 5, 1.0)

        # Parameter specificity
        parameter_patterns = [
            r"--level \w+",
            r"--name \w+",
            r"--type \w+",
            r"--output \w+",
        ]
        parameter_count = sum(
            len(re.findall(pattern, response)) for pattern in parameter_patterns
        )
        evaluation.criteria.parameter_specificity = min(parameter_count / 3, 1.0)

        # Overall actionability
        evaluation.criteria.actionability = (
            evaluation.criteria.executable_commands * 0.4
            + evaluation.criteria.step_by_step_clarity * 0.4
            + evaluation.criteria.parameter_specificity * 0.2
        )

    async def _evaluate_context_awareness(self, evaluation: AIResponseEvaluation):
        """Evaluate context awareness (20% weight)"""

        context = evaluation.project_context
        response = evaluation.ai_response

        # Project state utilization
        context_usage = 0.0
        if context:
            context_keys = [
                "is_mlx_project",
                "has_components",
                "plugins_available",
                "security_status",
            ]
            mentioned_contexts = sum(
                1
                for key in context_keys
                if str(context.get(key, "")).lower() in response.lower()
            )
            context_usage = mentioned_contexts / len(context_keys)
        evaluation.criteria.project_state_utilization = context_usage

        # Framework cross-references
        cross_ref_patterns = [
            "after.*security.*scan",
            "before.*plugin.*create",
            "with.*golden.*repo",
            "integrate.*framework",
            "combine.*with",
        ]
        cross_refs = sum(
            1 for pattern in cross_ref_patterns if re.search(pattern, response.lower())
        )
        evaluation.criteria.framework_cross_references = min(cross_refs / 2, 1.0)

        # Personalization (user-specific recommendations)
        personal_indicators = [
            "your project",
            "based on",
            "considering",
            "in your case",
            "for you",
        ]
        personalization = sum(
            1 for indicator in personal_indicators if indicator in response.lower()
        )
        evaluation.criteria.personalization = min(personalization / 3, 1.0)

        # Overall context awareness
        evaluation.criteria.context_awareness = (
            evaluation.criteria.project_state_utilization * 0.4
            + evaluation.criteria.framework_cross_references * 0.3
            + evaluation.criteria.personalization * 0.3
        )

    async def _evaluate_production_readiness(self, evaluation: AIResponseEvaluation):
        """Evaluate production readiness (15% weight)"""

        prod_analysis = self.production_analyzer.analyze_production_readiness(
            evaluation.ai_response, evaluation.user_query
        )

        evaluation.criteria.error_handling = (
            1.0 if prod_analysis["error_handling_mentioned"] else 0.0
        )
        evaluation.criteria.security_considerations = (
            1.0 if prod_analysis["security_considerations"] else 0.0
        )
        evaluation.criteria.monitoring_guidance = (
            1.0 if prod_analysis["monitoring_guidance"] else 0.0
        )

        # Overall production readiness
        evaluation.criteria.production_readiness = (
            prod_analysis["completeness_score"] / 100
        )

    async def _evaluate_user_experience(self, evaluation: AIResponseEvaluation):
        """Evaluate user experience (10% weight)"""

        response = evaluation.ai_response

        # Clarity and formatting
        formatting_score = 0.0
        if "```" in response:
            formatting_score += 0.3  # Code blocks
        if re.search(r"\*\*.*?\*\*", response):
            formatting_score += 0.2  # Bold text
        if re.search(r"^\d+\.", response, re.MULTILINE):
            formatting_score += 0.2  # Numbered lists
        if re.search(r"^[•\-\*]", response, re.MULTILINE):
            formatting_score += 0.2  # Bullet points
        if len(response.split("\n\n")) > 2:
            formatting_score += 0.1  # Paragraphs
        evaluation.criteria.clarity_formatting = min(formatting_score, 1.0)

        # Tone and professionalism
        professional_indicators = [
            "recommend",
            "suggest",
            "consider",
            "ensure",
            "verify",
        ]
        unprofessional_indicators = ["just", "simply", "easy", "obvious", "trivial"]
        professional_score = sum(
            1 for indicator in professional_indicators if indicator in response.lower()
        )
        unprofessional_score = sum(
            1
            for indicator in unprofessional_indicators
            if indicator in response.lower()
        )
        tone_score = max(0, (professional_score - unprofessional_score) / 5)
        evaluation.criteria.tone_professionalism = min(tone_score, 1.0)

        # Appropriate detail level (not too verbose, not too brief)
        word_count = len(response.split())
        if 100 <= word_count <= 500:
            detail_score = 1.0
        elif 50 <= word_count < 100 or 500 < word_count <= 800:
            detail_score = 0.7
        else:
            detail_score = 0.3
        evaluation.criteria.appropriate_detail_level = detail_score

        # Overall user experience
        evaluation.criteria.user_experience = (
            evaluation.criteria.clarity_formatting * 0.4
            + evaluation.criteria.tone_professionalism * 0.3
            + evaluation.criteria.appropriate_detail_level * 0.3
        )

    async def _generate_evaluation_insights(self, evaluation: AIResponseEvaluation):
        """Generate insights and recommendations"""

        # Identify strengths
        if evaluation.criteria.command_accuracy > 0.9:
            evaluation.strengths.append("✅ Excellent mlx command accuracy")
        if evaluation.criteria.framework_integration > 0.8:
            evaluation.strengths.append("✅ Strong framework integration")
        if evaluation.criteria.actionability > 0.8:
            evaluation.strengths.append("✅ Highly actionable recommendations")
        if evaluation.criteria.production_readiness > 0.7:
            evaluation.strengths.append("✅ Production-ready guidance")

        # Identify improvement areas
        if evaluation.criteria.command_accuracy < 0.7:
            evaluation.improvement_areas.append("❌ Command accuracy needs improvement")
            evaluation.missing_elements.append("Correct mlx command syntax")
        if evaluation.criteria.framework_integration < 0.6:
            evaluation.improvement_areas.append("❌ Lacks mlx framework integration")
            evaluation.missing_elements.append("Framework-specific features")
        if evaluation.criteria.context_awareness < 0.5:
            evaluation.improvement_areas.append("❌ Poor context awareness")
            evaluation.missing_elements.append("Project-specific recommendations")
        if evaluation.criteria.production_readiness < 0.5:
            evaluation.improvement_areas.append("❌ Insufficient production guidance")
            evaluation.missing_elements.append(
                "Error handling and security considerations"
            )

    async def _store_evaluation(self, evaluation: AIResponseEvaluation):
        """Store evaluation results"""

        # Save detailed evaluation
        eval_file = self.evaluations_dir / f"eval_{evaluation.evaluation_id}.json"
        with open(eval_file, "w") as f:
            json.dump(asdict(evaluation), f, indent=2, default=str)

        # Update evaluation history
        self.evaluation_history.append(
            {
                "evaluation_id": evaluation.evaluation_id,
                "timestamp": evaluation.timestamp.isoformat(),
                "final_score": evaluation.final_score,
                "grade": evaluation.grade,
                "query_category": self._categorize_query(evaluation.user_query),
            }
        )

    def _categorize_query(self, query: str) -> str:
        """Categorize user query for analytics"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["security", "scan", "vulnerability"]):
            return "security"
        elif any(word in query_lower for word in ["plugin", "development", "create"]):
            return "plugin_development"
        elif any(word in query_lower for word in ["golden", "repository", "template"]):
            return "golden_repos"
        elif any(
            word in query_lower
            for word in ["troubleshoot", "error", "problem", "issue"]
        ):
            return "troubleshooting"
        elif any(
            word in query_lower for word in ["optimize", "performance", "improve"]
        ):
            return "optimization"
        else:
            return "general"

    def generate_evaluation_report(self, evaluation: AIResponseEvaluation) -> str:
        """Generate formatted evaluation report"""

        report = f"""
🎯 **MLX AI Response Evaluation Report**

**Evaluation ID:** {evaluation.evaluation_id}
**Timestamp:** {evaluation.timestamp.strftime("%Y-%m-%d %H:%M:%S")}
**Final Score:** {evaluation.final_score:.1f}/100 (Grade: {evaluation.grade})

## 📊 **Detailed Scoring**

### Mlx Platform Accuracy (30% weight): {evaluation.criteria.mlx_platform_accuracy * 100:.1f}%
- Command Accuracy: {evaluation.criteria.command_accuracy * 100:.1f}%
- Framework Integration: {evaluation.criteria.framework_integration * 100:.1f}%
- Platform Specificity: {evaluation.criteria.platform_specificity * 100:.1f}%

### Actionability (25% weight): {evaluation.criteria.actionability * 100:.1f}%
- Executable Commands: {evaluation.criteria.executable_commands * 100:.1f}%
- Step-by-Step Clarity: {evaluation.criteria.step_by_step_clarity * 100:.1f}%
- Parameter Specificity: {evaluation.criteria.parameter_specificity * 100:.1f}%

### Context Awareness (20% weight): {evaluation.criteria.context_awareness * 100:.1f}%
- Project State Utilization: {evaluation.criteria.project_state_utilization * 100:.1f}%
- Framework Cross-References: {evaluation.criteria.framework_cross_references * 100:.1f}%
- Personalization: {evaluation.criteria.personalization * 100:.1f}%

### Production Readiness (15% weight): {evaluation.criteria.production_readiness * 100:.1f}%
- Error Handling: {evaluation.criteria.error_handling * 100:.1f}%
- Security Considerations: {evaluation.criteria.security_considerations * 100:.1f}%
- Monitoring Guidance: {evaluation.criteria.monitoring_guidance * 100:.1f}%

### User Experience (10% weight): {evaluation.criteria.user_experience * 100:.1f}%
- Clarity & Formatting: {evaluation.criteria.clarity_formatting * 100:.1f}%
- Tone & Professionalism: {evaluation.criteria.tone_professionalism * 100:.1f}%
- Appropriate Detail Level: {evaluation.criteria.appropriate_detail_level * 100:.1f}%

## 🎯 **Command Analysis**
- Total Commands Found: {evaluation.command_accuracy_details.get("total_commands", 0)}
- Correct Commands: {evaluation.command_accuracy_details.get("correct_commands", 0)}
- Command Accuracy: {evaluation.command_accuracy_details.get("accuracy_percentage", 0):.1f}%
- Frameworks Covered: {", ".join(evaluation.framework_coverage) if evaluation.framework_coverage else "None"}

## ✅ **Strengths**
{chr(10).join(f"- {strength}" for strength in evaluation.strengths)}

## ❌ **Areas for Improvement**
{chr(10).join(f"- {area}" for area in evaluation.improvement_areas)}

## 🔍 **Missing Elements**
{chr(10).join(f"- {element}" for element in evaluation.missing_elements)}

## ⚡ **Performance Metrics**
- Response Time: {evaluation.response_time:.2f}s
- Token Usage: {evaluation.token_usage.get("total_tokens", "N/A")}
- Estimated Cost: ${evaluation.cost_estimate:.4f}

---
**Query:** {evaluation.user_query}

**Response Preview:** {evaluation.ai_response[:200]}...
"""
        return report.strip()


# Export main classes
__all__ = [
    "AIResponseEvaluator",
    "AIResponseEvaluation",
    "EvaluationCriteria",
    "MLXCommandValidator",
    "MLXFrameworkAnalyzer",
    "ProductionReadinessAnalyzer",
]
