#!/usr/bin/env python3
"""
🎯 MLX AI Response Benchmark Dataset Generator

Creates comprehensive test datasets for evaluating AI assistant responses
across all Mlx Platform Foundation capabilities and use cases.

Features:
- Category-based test scenarios (Security, Plugins, Golden Repos, etc.)
- Difficulty levels (Basic, Intermediate, Advanced, Expert)
- Real-world usage patterns and edge cases
- Expected response quality benchmarks
- Regression testing scenarios
"""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class BenchmarkScenario:
    """A single benchmark test scenario"""

    scenario_id: str
    category: str
    difficulty: str  # basic, intermediate, advanced, expert
    user_query: str
    expected_commands: List[str] = field(default_factory=list)
    expected_frameworks: List[str] = field(default_factory=list)
    project_context: Dict[str, Any] = field(default_factory=dict)
    success_criteria: Dict[str, float] = field(
        default_factory=dict
    )  # min scores for each dimension
    description: str = ""
    tags: List[str] = field(default_factory=list)


class BenchmarkDatasetGenerator:
    """Generates comprehensive benchmark datasets for MLX AI evaluation"""

    def __init__(self):
        self.scenarios = []
        self.categories = [
            "security_workflows",
            "plugin_development",
            "golden_repos",
            "troubleshooting",
            "optimization",
            "integration",
            "workflow_automation",
            "configuration",
        ]

        # Success criteria templates for different difficulty levels
        self.success_criteria_templates = {
            "basic": {
                "command_accuracy": 0.80,
                "framework_integration": 0.60,
                "actionability": 0.70,
                "production_readiness": 0.50,
                "user_experience": 0.70,
            },
            "intermediate": {
                "command_accuracy": 0.85,
                "framework_integration": 0.70,
                "actionability": 0.80,
                "production_readiness": 0.65,
                "user_experience": 0.75,
            },
            "advanced": {
                "command_accuracy": 0.90,
                "framework_integration": 0.80,
                "actionability": 0.85,
                "production_readiness": 0.75,
                "user_experience": 0.80,
            },
            "expert": {
                "command_accuracy": 0.95,
                "framework_integration": 0.90,
                "actionability": 0.90,
                "production_readiness": 0.85,
                "user_experience": 0.85,
            },
        }

    def generate_security_scenarios(self) -> List[BenchmarkScenario]:
        """Generate security workflow test scenarios"""
        scenarios = []

        # Basic security scenarios
        scenarios.extend(
            [
                BenchmarkScenario(
                    scenario_id="sec_001",
                    category="security_workflows",
                    difficulty="basic",
                    user_query="How do I set up security scanning?",
                    expected_commands=[
                        "mlx assistant security scan",
                        "mlx assistant security scan --level enhanced",
                    ],
                    expected_frameworks=["security"],
                    project_context={"is_mlx_project": True, "has_components": False},
                    success_criteria=self.success_criteria_templates["basic"],
                    description="Basic security scanning setup",
                    tags=["security", "scanning", "setup"],
                ),
                BenchmarkScenario(
                    scenario_id="sec_002",
                    category="security_workflows",
                    difficulty="basic",
                    user_query="What security levels are available?",
                    expected_commands=[
                        "mlx assistant security scan --level basic",
                        "mlx assistant security scan --level enhanced",
                        "mlx assistant security scan --level enterprise",
                        "mlx assistant security scan --level critical",
                    ],
                    expected_frameworks=["security"],
                    success_criteria=self.success_criteria_templates["basic"],
                    description="Security levels explanation",
                    tags=["security", "levels", "information"],
                ),
            ]
        )

        # Intermediate security scenarios
        scenarios.extend(
            [
                BenchmarkScenario(
                    scenario_id="sec_101",
                    category="security_workflows",
                    difficulty="intermediate",
                    user_query="Set up comprehensive security scanning for a production mlx project with SBOM generation",
                    expected_commands=[
                        "mlx assistant security scan --level enterprise",
                        "mlx assistant security sbom",
                        "mlx assistant security baseline",
                    ],
                    expected_frameworks=["security"],
                    project_context={
                        "is_mlx_project": True,
                        "has_components": True,
                        "plugins_available": 3,
                    },
                    success_criteria=self.success_criteria_templates["intermediate"],
                    description="Production security setup with SBOM",
                    tags=["security", "production", "sbom", "enterprise"],
                ),
                BenchmarkScenario(
                    scenario_id="sec_102",
                    category="security_workflows",
                    difficulty="intermediate",
                    user_query="How do I integrate security scanning into my CI/CD pipeline?",
                    expected_commands=[
                        "mlx assistant security scan --level enhanced",
                        "mlx assistant security baseline",
                    ],
                    expected_frameworks=["security"],
                    success_criteria=self.success_criteria_templates["intermediate"],
                    description="CI/CD security integration",
                    tags=["security", "cicd", "automation", "integration"],
                ),
            ]
        )

        # Advanced security scenarios
        scenarios.extend(
            [
                BenchmarkScenario(
                    scenario_id="sec_201",
                    category="security_workflows",
                    difficulty="advanced",
                    user_query="Implement a complete security hardening framework with vulnerability tracking, compliance reporting, and automated remediation for a multi-component mlx project",
                    expected_commands=[
                        "mlx assistant security scan --level critical",
                        "mlx assistant security sbom",
                        "mlx assistant security verify",
                        "mlx assistant security compare",
                        "mlx assistant security report",
                    ],
                    expected_frameworks=["security", "plugins", "golden_repos"],
                    project_context={
                        "is_mlx_project": True,
                        "has_components": True,
                        "plugins_available": 5,
                        "security_status": "enterprise",
                    },
                    success_criteria=self.success_criteria_templates["advanced"],
                    description="Complete security hardening framework",
                    tags=[
                        "security",
                        "compliance",
                        "automation",
                        "enterprise",
                        "multi-component",
                    ],
                )
            ]
        )

        return scenarios

    def generate_plugin_scenarios(self) -> List[BenchmarkScenario]:
        """Generate plugin development test scenarios"""
        scenarios = []

        # Basic plugin scenarios
        scenarios.extend(
            [
                BenchmarkScenario(
                    scenario_id="plug_001",
                    category="plugin_development",
                    difficulty="basic",
                    user_query="How do I create a new MLX plugin?",
                    expected_commands=[
                        "mlx assistant plugins create --name my-plugin --type ml_framework"
                    ],
                    expected_frameworks=["plugins"],
                    success_criteria=self.success_criteria_templates["basic"],
                    description="Basic plugin creation",
                    tags=["plugin", "create", "basic"],
                ),
                BenchmarkScenario(
                    scenario_id="plug_002",
                    category="plugin_development",
                    difficulty="basic",
                    user_query="List all available plugin types",
                    expected_commands=["mlx assistant plugins list"],
                    expected_frameworks=["plugins"],
                    success_criteria=self.success_criteria_templates["basic"],
                    description="Plugin types listing",
                    tags=["plugin", "list", "types"],
                ),
            ]
        )

        # Intermediate plugin scenarios
        scenarios.extend(
            [
                BenchmarkScenario(
                    scenario_id="plug_101",
                    category="plugin_development",
                    difficulty="intermediate",
                    user_query="Create a data processing plugin with validation and testing setup",
                    expected_commands=[
                        "mlx assistant plugins create --name data-processor --type data_processor",
                        "mlx assistant plugins validate",
                    ],
                    expected_frameworks=["plugins"],
                    success_criteria=self.success_criteria_templates["intermediate"],
                    description="Data processing plugin with validation",
                    tags=["plugin", "data_processor", "validation", "testing"],
                )
            ]
        )

        # Advanced plugin scenarios
        scenarios.extend(
            [
                BenchmarkScenario(
                    scenario_id="plug_201",
                    category="plugin_development",
                    difficulty="advanced",
                    user_query="Develop a complete plugin ecosystem with multiple interconnected plugins, validation framework, and automated testing pipeline",
                    expected_commands=[
                        "mlx assistant plugins create --name core-ml --type ml_framework",
                        "mlx assistant plugins create --name data-pipeline --type data_processor",
                        "mlx assistant plugins create --name monitoring --type monitoring",
                        "mlx assistant plugins validate",
                        "mlx assistant security scan",
                    ],
                    expected_frameworks=["plugins", "security"],
                    project_context={"is_mlx_project": True, "has_components": True},
                    success_criteria=self.success_criteria_templates["advanced"],
                    description="Complete plugin ecosystem development",
                    tags=["plugin", "ecosystem", "validation", "automation", "testing"],
                )
            ]
        )

        return scenarios

    def generate_golden_repos_scenarios(self) -> List[BenchmarkScenario]:
        """Generate golden repository test scenarios"""
        scenarios = []

        # Basic scenarios
        scenarios.extend(
            [
                BenchmarkScenario(
                    scenario_id="gold_001",
                    category="golden_repos",
                    difficulty="basic",
                    user_query="Create a minimal golden repository for testing",
                    expected_commands=["mlx assistant golden-repos create minimal"],
                    expected_frameworks=["golden_repos"],
                    success_criteria=self.success_criteria_templates["basic"],
                    description="Minimal golden repository creation",
                    tags=["golden_repos", "minimal", "testing"],
                ),
                BenchmarkScenario(
                    scenario_id="gold_002",
                    category="golden_repos",
                    difficulty="basic",
                    user_query="What golden repository specifications are available?",
                    expected_commands=["mlx assistant golden-repos list"],
                    expected_frameworks=["golden_repos"],
                    success_criteria=self.success_criteria_templates["basic"],
                    description="Golden repository specifications listing",
                    tags=["golden_repos", "list", "specifications"],
                ),
            ]
        )

        # Intermediate scenarios
        scenarios.extend(
            [
                BenchmarkScenario(
                    scenario_id="gold_101",
                    category="golden_repos",
                    difficulty="intermediate",
                    user_query="Set up a standard golden repository with component extraction and validation",
                    expected_commands=[
                        "mlx assistant golden-repos create standard",
                        "mlx assistant golden-repos validate standard",
                    ],
                    expected_frameworks=["golden_repos"],
                    success_criteria=self.success_criteria_templates["intermediate"],
                    description="Standard repository with validation",
                    tags=["golden_repos", "standard", "validation", "components"],
                )
            ]
        )

        # Advanced scenarios
        scenarios.extend(
            [
                BenchmarkScenario(
                    scenario_id="gold_201",
                    category="golden_repos",
                    difficulty="advanced",
                    user_query="Create a complete golden repository testing suite with multiple specifications, validation, and performance benchmarking",
                    expected_commands=[
                        "mlx assistant golden-repos create-all",
                        "mlx assistant golden-repos validate-all",
                        "mlx assistant golden-repos create performance",
                    ],
                    expected_frameworks=["golden_repos", "security"],
                    project_context={"is_mlx_project": True},
                    success_criteria=self.success_criteria_templates["advanced"],
                    description="Complete golden repository testing suite",
                    tags=[
                        "golden_repos",
                        "testing",
                        "performance",
                        "validation",
                        "comprehensive",
                    ],
                )
            ]
        )

        return scenarios

    def generate_troubleshooting_scenarios(self) -> List[BenchmarkScenario]:
        """Generate troubleshooting test scenarios"""
        scenarios = []

        scenarios.extend(
            [
                BenchmarkScenario(
                    scenario_id="trouble_001",
                    category="troubleshooting",
                    difficulty="basic",
                    user_query="MLX assistant commands are not working, how do I troubleshoot?",
                    expected_commands=["mlx assistant doctor"],
                    expected_frameworks=["golden_repos", "security", "plugins"],
                    success_criteria=self.success_criteria_templates["basic"],
                    description="Basic troubleshooting with health check",
                    tags=["troubleshooting", "health", "doctor"],
                ),
                BenchmarkScenario(
                    scenario_id="trouble_002",
                    category="troubleshooting",
                    difficulty="intermediate",
                    user_query="My plugin validation is failing with permission errors, what should I check?",
                    expected_commands=[
                        "mlx assistant doctor",
                        "mlx assistant security scan",
                        "mlx assistant plugins validate",
                    ],
                    expected_frameworks=["plugins", "security"],
                    success_criteria=self.success_criteria_templates["intermediate"],
                    description="Plugin validation troubleshooting",
                    tags=["troubleshooting", "plugins", "permissions", "validation"],
                ),
            ]
        )

        return scenarios

    def generate_integration_scenarios(self) -> List[BenchmarkScenario]:
        """Generate framework integration test scenarios"""
        scenarios = []

        scenarios.extend(
            [
                BenchmarkScenario(
                    scenario_id="integ_001",
                    category="integration",
                    difficulty="intermediate",
                    user_query="How do I integrate security scanning with plugin development workflow?",
                    expected_commands=[
                        "mlx assistant plugins create --name secure-plugin --type security",
                        "mlx assistant security scan --level enhanced",
                        "mlx assistant plugins validate",
                    ],
                    expected_frameworks=["plugins", "security"],
                    success_criteria=self.success_criteria_templates["intermediate"],
                    description="Security and plugin integration",
                    tags=["integration", "security", "plugins", "workflow"],
                ),
                BenchmarkScenario(
                    scenario_id="integ_002",
                    category="integration",
                    difficulty="advanced",
                    user_query="Create a complete MLX workflow that integrates all frameworks: golden repos, security, plugins, and glossary",
                    expected_commands=[
                        "mlx assistant golden-repos create advanced",
                        "mlx assistant security scan --level enterprise",
                        "mlx assistant plugins create --name integrated-plugin --type ml_framework",
                        "mlx assistant glossary view",
                        "mlx assistant plugins validate",
                    ],
                    expected_frameworks=[
                        "golden_repos",
                        "security",
                        "plugins",
                        "glossary",
                    ],
                    project_context={"is_mlx_project": True, "has_components": True},
                    success_criteria=self.success_criteria_templates["advanced"],
                    description="Complete framework integration",
                    tags=["integration", "comprehensive", "all_frameworks", "workflow"],
                ),
            ]
        )

        return scenarios

    def generate_edge_case_scenarios(self) -> List[BenchmarkScenario]:
        """Generate edge case and error handling scenarios"""
        scenarios = []

        scenarios.extend(
            [
                BenchmarkScenario(
                    scenario_id="edge_001",
                    category="troubleshooting",
                    difficulty="intermediate",
                    user_query="What happens if I run security scan on a non-mlx project?",
                    expected_commands=["mlx assistant doctor", "mlx create project"],
                    expected_frameworks=["security"],
                    project_context={"is_mlx_project": False},
                    success_criteria=self.success_criteria_templates["intermediate"],
                    description="Non-mlx project handling",
                    tags=["edge_case", "non_mlx", "error_handling"],
                ),
                BenchmarkScenario(
                    scenario_id="edge_002",
                    category="configuration",
                    difficulty="advanced",
                    user_query="How do I handle conflicting plugin dependencies in a complex mlx project?",
                    expected_commands=[
                        "mlx assistant plugins list",
                        "mlx assistant plugins validate",
                        "mlx assistant doctor",
                    ],
                    expected_frameworks=["plugins"],
                    project_context={
                        "is_mlx_project": True,
                        "has_components": True,
                        "plugins_available": 10,
                    },
                    success_criteria=self.success_criteria_templates["advanced"],
                    description="Complex dependency resolution",
                    tags=["edge_case", "dependencies", "conflicts", "complex"],
                ),
            ]
        )

        return scenarios

    def generate_benchmark_dataset(
        self, include_categories: Optional[List[str]] = None
    ) -> List[BenchmarkScenario]:
        """Generate complete benchmark dataset"""
        all_scenarios = []

        # Generate scenarios by category
        generators = {
            "security_workflows": self.generate_security_scenarios,
            "plugin_development": self.generate_plugin_scenarios,
            "golden_repos": self.generate_golden_repos_scenarios,
            "troubleshooting": self.generate_troubleshooting_scenarios,
            "integration": self.generate_integration_scenarios,
        }

        # Add edge cases to troubleshooting
        edge_cases = self.generate_edge_case_scenarios()

        categories_to_include = include_categories or self.categories

        for category in categories_to_include:
            if category in generators:
                scenarios = generators[category]()
                all_scenarios.extend(scenarios)

        # Add edge cases
        all_scenarios.extend(edge_cases)

        # Shuffle for variety in testing
        random.shuffle(all_scenarios)

        self.scenarios = all_scenarios
        return all_scenarios

    def save_benchmark_dataset(self, output_path: Path):
        """Save benchmark dataset to JSON file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dataset = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_scenarios": len(self.scenarios),
                "categories": list(set(s.category for s in self.scenarios)),
                "difficulty_levels": list(set(s.difficulty for s in self.scenarios)),
            },
            "scenarios": [asdict(scenario) for scenario in self.scenarios],
        }

        with open(output_path, "w") as f:
            json.dump(dataset, f, indent=2)

        print(f"✅ Benchmark dataset saved to {output_path}")
        print(f"📊 Total scenarios: {len(self.scenarios)}")

        # Print category breakdown
        category_counts = {}
        for scenario in self.scenarios:
            category_counts[scenario.category] = (
                category_counts.get(scenario.category, 0) + 1
            )

        print("📋 Category breakdown:")
        for category, count in sorted(category_counts.items()):
            print(f"  - {category}: {count} scenarios")

    def load_benchmark_dataset(self, input_path: Path) -> List[BenchmarkScenario]:
        """Load benchmark dataset from JSON file"""
        with open(input_path) as f:
            data = json.load(f)

        scenarios = []
        for scenario_data in data["scenarios"]:
            scenario = BenchmarkScenario(**scenario_data)
            scenarios.append(scenario)

        self.scenarios = scenarios
        return scenarios

    def get_scenarios_by_category(self, category: str) -> List[BenchmarkScenario]:
        """Get scenarios filtered by category"""
        return [s for s in self.scenarios if s.category == category]

    def get_scenarios_by_difficulty(self, difficulty: str) -> List[BenchmarkScenario]:
        """Get scenarios filtered by difficulty"""
        return [s for s in self.scenarios if s.difficulty == difficulty]


if __name__ == "__main__":
    # Generate and save benchmark dataset
    generator = BenchmarkDatasetGenerator()
    scenarios = generator.generate_benchmark_dataset()

    output_path = Path("data/benchmarks/mlx_ai_benchmark_dataset.json")
    generator.save_benchmark_dataset(output_path)
