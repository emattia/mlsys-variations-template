#!/usr/bin/env python3
"""
🎯 MLX AI Evaluation System - Main CLI Interface
Production-ready evaluation system for Mlx Platform Foundation AI responses.
Repository infrastructure for comprehensive AI assistant quality assessment.

Usage:
    mlx-eval run --query "How do I set up security scanning?"
    mlx-eval benchmark --category security
    mlx-eval dashboard
    mlx-eval analyze --category plugin_development
    mlx-eval export --format json

Features:
- Single AI response evaluation with detailed analysis
- Benchmark dataset evaluation across multiple scenarios
- Interactive analytics dashboard with rich visualization
- Performance tracking and regression detection
- Comprehensive reporting and export capabilities
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Optional, List
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

# Local imports
from .ai_response_evaluator import AIResponseEvaluator, AIResponseEvaluation
from .benchmark_generator import BenchmarkDatasetGenerator, BenchmarkScenario
from .analytics_dashboard import AnalyticsDashboard

console = Console()
app = typer.Typer(
    name="mlx-eval",
    help="🎯 MLX AI Evaluation System - Production-grade AI assistant quality assessment",
    add_completion=False,
)

# Global evaluator instance
evaluator = None


def get_evaluator() -> AIResponseEvaluator:
    """Get or create evaluator instance"""
    global evaluator
    if evaluator is None:
        evaluator = AIResponseEvaluator()
    return evaluator


@app.command("run")
def evaluate_single_response(
    query: str = typer.Option(
        ..., "--query", "-q", help="User query to evaluate AI response for"
    ),
    response: Optional[str] = typer.Option(
        None,
        "--response",
        "-r",
        help="AI response to evaluate (will prompt if not provided)",
    ),
    context_file: Optional[str] = typer.Option(
        None, "--context", "-c", help="JSON file with project context"
    ),
    output_format: str = typer.Option(
        "console", "--format", "-f", help="Output format: console, json, report"
    ),
    save_result: bool = typer.Option(
        True, "--save/--no-save", help="Save evaluation result"
    ),
):
    """
    🔍 Evaluate a single AI response for quality and accuracy

    Examples:
        mlx-eval run -q "How do I set up security scanning?"
        mlx-eval run -q "Create a plugin" -r "Use mlx assistant plugins create" --format json
    """
    # Get AI response if not provided
    if not response:
        console.print(f"\n[cyan]Query:[/cyan] {query}")
        console.print("\n[yellow]Please provide the AI response to evaluate:[/yellow]")
        response = typer.prompt("AI Response", type=str)

    # Load project context if provided
    project_context = {}
    if context_file:
        try:
            with open(context_file) as f:
                project_context = json.load(f)
        except Exception as e:
            console.print(f"[red]Error loading context file: {e}[/red]")
            return

    # Run evaluation
    console.print("\n🔄 [yellow]Running evaluation...[/yellow]")
    start_time = time.time()
    evaluation = asyncio.run(
        get_evaluator().evaluate_response(
            user_query=query,
            ai_response=response,
            project_context=project_context,
            response_time=0.0,  # We don't have actual response time
        )
    )
    processing_time = time.time() - start_time

    # Display results
    if output_format == "console":
        _display_evaluation_console(evaluation, processing_time)
    elif output_format == "json":
        _display_evaluation_json(evaluation)
    elif output_format == "report":
        report = get_evaluator().generate_evaluation_report(evaluation)
        console.print(report)

    if save_result:
        console.print(
            f"\n✅ [green]Evaluation saved with ID: {evaluation.evaluation_id}[/green]"
        )


@app.command("benchmark")
def run_benchmark_evaluation(
    category: Optional[str] = typer.Option(
        None, "--category", "-c", help="Benchmark category to run"
    ),
    difficulty: Optional[str] = typer.Option(
        None,
        "--difficulty",
        "-d",
        help="Difficulty level: basic, intermediate, advanced, expert",
    ),
    dataset_file: Optional[str] = typer.Option(
        None, "--dataset", help="Custom benchmark dataset file"
    ),
    limit: int = typer.Option(
        10, "--limit", "-l", help="Limit number of scenarios to run"
    ),
    output_dir: str = typer.Option(
        "data/benchmark_results", "--output", "-o", help="Output directory for results"
    ),
):
    """
    🧪 Run benchmark evaluation across multiple test scenarios

    Examples:
        mlx-eval benchmark --category security --limit 5
        mlx-eval benchmark --difficulty advanced
        mlx-eval benchmark --dataset custom_scenarios.json
    """
    # Load or generate benchmark dataset
    generator = BenchmarkDatasetGenerator()
    if dataset_file:
        scenarios = generator.load_benchmark_dataset(Path(dataset_file))
        console.print(f"📊 Loaded {len(scenarios)} scenarios from {dataset_file}")
    else:
        scenarios = generator.generate_benchmark_dataset()
        console.print(f"📊 Generated {len(scenarios)} benchmark scenarios")

    # Filter scenarios
    if category:
        scenarios = [s for s in scenarios if s.category == category]
        console.print(
            f"🔍 Filtered to {len(scenarios)} scenarios for category: {category}"
        )
    if difficulty:
        scenarios = [s for s in scenarios if s.difficulty == difficulty]
        console.print(
            f"🔍 Filtered to {len(scenarios)} scenarios for difficulty: {difficulty}"
        )

    # Limit scenarios
    scenarios = scenarios[:limit]
    if not scenarios:
        console.print("[red]No scenarios found matching the criteria![/red]")
        return

    console.print(
        f"\n🚀 Running benchmark evaluation on {len(scenarios)} scenarios...\n"
    )

    # Run benchmark with progress tracking
    results = []
    with Progress() as progress:
        task = progress.add_task("Evaluating scenarios...", total=len(scenarios))
        for i, scenario in enumerate(scenarios, 1):
            # Simulate AI response (in real usage, this would come from the AI assistant)
            mock_response = _generate_mock_response(scenario)

            # Run evaluation
            evaluation = asyncio.run(
                get_evaluator().evaluate_response(
                    user_query=scenario.user_query,
                    ai_response=mock_response,
                    project_context=scenario.project_context,
                    response_time=1.5,  # Mock response time
                )
            )
            results.append(
                {
                    "scenario": scenario,
                    "evaluation": evaluation,
                    "passed": _check_scenario_pass(evaluation, scenario),
                }
            )
            progress.update(task, advance=1)

            # Show progress
            if i % 5 == 0:
                passed = sum(1 for r in results if r["passed"])
                console.print(
                    f"  Progress: {i}/{len(scenarios)} | Passed: {passed}/{i} ({passed / i * 100:.1f}%)"
                )

    # Generate benchmark report
    _display_benchmark_results(results, output_dir)


@app.command("dashboard")
def show_analytics_dashboard():
    """
    📊 Display interactive analytics dashboard with performance metrics
    """
    dashboard = AnalyticsDashboard()
    dashboard.load_evaluation_data()
    dashboard.display_overview_dashboard()


@app.command("analyze")
def analyze_performance(
    category: Optional[str] = typer.Option(
        None, "--category", "-c", help="Analyze specific category"
    ),
    time_range: int = typer.Option(30, "--days", "-d", help="Time range in days"),
    export_path: Optional[str] = typer.Option(
        None, "--export", "-e", help="Export analysis to file"
    ),
):
    """
    🔍 Analyze performance trends and detailed metrics

    Examples:
        mlx-eval analyze --category security
        mlx-eval analyze --days 7 --export weekly_analysis.json
    """
    dashboard = AnalyticsDashboard()
    dashboard.load_evaluation_data()
    dashboard.display_detailed_analysis(category)
    if export_path:
        dashboard.export_metrics_report(Path(export_path))


@app.command("generate-dataset")
def generate_benchmark_dataset(
    categories: Optional[List[str]] = typer.Option(
        None, "--categories", "-c", help="Categories to include"
    ),
    output_file: str = typer.Option(
        "data/benchmarks/mlx_ai_benchmark_dataset.json",
        "--output",
        "-o",
        help="Output file path",
    ),
    shuffle: bool = typer.Option(
        True, "--shuffle/--no-shuffle", help="Shuffle scenarios"
    ),
):
    """
    📝 Generate benchmark dataset for testing

    Examples:
        mlx-eval generate-dataset --categories security plugin_development
        mlx-eval generate-dataset --output custom_benchmarks.json
    """
    generator = BenchmarkDatasetGenerator()
    _ = generator.generate_benchmark_dataset(categories)
    generator.save_benchmark_dataset(Path(output_file))


@app.command("export")
def export_evaluation_data(
    format_type: str = typer.Option(
        "json", "--format", "-f", help="Export format: json, csv, html"
    ),
    output_path: str = typer.Option(
        "reports/mlx_evaluation_export",
        "--output",
        "-o",
        help="Output path (without extension)",
    ),
    category: Optional[str] = typer.Option(
        None, "--category", "-c", help="Filter by category"
    ),
    time_range: int = typer.Option(30, "--days", "-d", help="Time range in days"),
):
    """
    📤 Export evaluation data and analytics reports

    Examples:
        mlx-eval export --format json --category security
        mlx-eval export --format html --days 7
    """
    dashboard = AnalyticsDashboard()
    dashboard.load_evaluation_data()
    if format_type == "json":
        output_file = Path(f"{output_path}.json")
        dashboard.export_metrics_report(output_file)
    elif format_type == "html":
        _export_html_report(dashboard, f"{output_path}.html")
    elif format_type == "csv":
        _export_csv_data(dashboard, f"{output_path}.csv")


@app.command("status")
def show_system_status():
    """
    ⚡ Show system status and health check
    """
    console.print("\n🎯 [bold cyan]MLX AI Evaluation System Status[/bold cyan]\n")

    # Check data directories
    data_dir = Path("data/evaluations")
    benchmarks_dir = Path("data/benchmarks")
    reports_dir = Path("reports")
    table = Table(title="📊 System Health", show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="white")

    # Data directories
    table.add_row(
        "Evaluation Data",
        "✅ Available" if data_dir.exists() else "❌ Missing",
        f"{len(list(data_dir.glob('eval_*.json')))} evaluations"
        if data_dir.exists()
        else "Directory not found",
    )
    table.add_row(
        "Benchmark Data",
        "✅ Available" if benchmarks_dir.exists() else "❌ Missing",
        f"{len(list(benchmarks_dir.glob('*.json')))} datasets"
        if benchmarks_dir.exists()
        else "Directory not found",
    )
    table.add_row(
        "Reports",
        "✅ Available" if reports_dir.exists() else "❌ Missing",
        f"{len(list(reports_dir.glob('*')))} files"
        if reports_dir.exists()
        else "Directory not found",
    )

    # System components
    try:
        _ = AIResponseEvaluator()
        table.add_row("AI Evaluator", "✅ Ready", "All components loaded")
    except Exception as e:
        table.add_row("AI Evaluator", "❌ Error", str(e))

    try:
        _ = AnalyticsDashboard()
        table.add_row("Analytics Dashboard", "✅ Ready", "Dashboard available")
    except Exception as e:
        table.add_row("Analytics Dashboard", "❌ Error", str(e))

    console.print(table)

    # Show recent activity
    if data_dir.exists():
        recent_files = sorted(
            data_dir.glob("eval_*.json"), key=lambda x: x.stat().st_mtime, reverse=True
        )[:5]
        if recent_files:
            console.print("\n📈 [bold]Recent Evaluations[/bold]")
            for i, file in enumerate(recent_files, 1):
                mtime = file.stat().st_mtime
                from datetime import datetime

                time_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                console.print(f"  {i}. {file.name} - {time_str}")


def _display_evaluation_console(
    evaluation: AIResponseEvaluation, processing_time: float
):
    """Display evaluation results in console format"""
    # Header
    header = (
        f"🎯 Evaluation Results - {evaluation.grade} ({evaluation.final_score:.1f}/100)"
    )
    console.print(
        Panel(
            header,
            style="bold green" if evaluation.final_score >= 80 else "bold yellow",
        )
    )

    # Detailed scores table
    table = Table(title="📊 Detailed Scoring", show_header=True)
    table.add_column("Dimension", style="cyan")
    table.add_column("Score", style="green")
    table.add_column("Weight", style="yellow")
    table.add_column("Weighted", style="magenta")
    criteria = evaluation.criteria
    weights = {
        "mlx_platform_accuracy": 30,
        "actionability": 25,
        "context_awareness": 20,
        "production_readiness": 15,
        "user_experience": 10,
    }
    for attr, weight in weights.items():
        score = getattr(criteria, attr) * 100
        weighted = score * (weight / 100)
        table.add_row(
            attr.replace("_", " ").title(),
            f"{score:.1f}%",
            f"{weight}%",
            f"{weighted:.1f}",
        )
    console.print(table)

    # Strengths and improvements
    if evaluation.strengths:
        console.print("\n✅ [bold green]Strengths[/bold green]")
        for strength in evaluation.strengths:
            console.print(f"  • {strength}")
    if evaluation.improvement_areas:
        console.print("\n❌ [bold red]Areas for Improvement[/bold red]")
        for area in evaluation.improvement_areas:
            console.print(f"  • {area}")

    # Command analysis
    cmd_details = evaluation.command_accuracy_details
    if cmd_details.get("total_commands", 0) > 0:
        console.print("\n🎯 [bold]Command Analysis[/bold]")
        console.print(f"  • Total Commands: {cmd_details['total_commands']}")
        console.print(f"  • Correct Commands: {cmd_details['correct_commands']}")
        console.print(f"  • Accuracy: {cmd_details.get('accuracy_percentage', 0):.1f}%")
        if cmd_details.get("framework_coverage"):
            console.print(
                f"  • Frameworks: {', '.join(cmd_details['framework_coverage'])}"
            )
    console.print(f"\n⚡ Processing time: {processing_time:.2f}s")


def _display_evaluation_json(evaluation: AIResponseEvaluation):
    """Display evaluation results in JSON format"""
    from dataclasses import asdict

    result = asdict(evaluation)
    console.print(json.dumps(result, indent=2, default=str))


def _generate_mock_response(scenario: BenchmarkScenario) -> str:
    """Generate a mock AI response for benchmarking (in real usage, this would come from the AI assistant)"""
    # Simple mock response generator based on scenario
    if scenario.category == "security_workflows":
        if "scan" in scenario.user_query.lower():
            return f"To set up security scanning, use: {scenario.expected_commands[0] if scenario.expected_commands else 'mlx assistant security scan --level enhanced'}"
        elif "levels" in scenario.user_query.lower():
            return "MLX supports these security levels: basic, enhanced, enterprise, and critical. Use --level parameter to specify."
        else:
            return f"For security setup, run: {scenario.expected_commands[0] if scenario.expected_commands else 'mlx assistant security scan'}"
    elif scenario.category == "plugin_development":
        if "create" in scenario.user_query.lower():
            return f"Create a plugin using: {scenario.expected_commands[0] if scenario.expected_commands else 'mlx assistant plugins create --name my-plugin --type ml_framework'}"
        elif "list" in scenario.user_query.lower():
            return "Available plugin types: ml_framework, data_processor, model_provider, deployment, monitoring, security, utility"
        else:
            return f"For plugin operations, use: {scenario.expected_commands[0] if scenario.expected_commands else 'mlx assistant plugins'}"
    elif scenario.category == "golden_repos":
        if "create" in scenario.user_query.lower():
            return f"Create a golden repository: {scenario.expected_commands[0] if scenario.expected_commands else 'mlx assistant golden-repos create standard'}"
        else:
            return f"For golden repos, use: {scenario.expected_commands[0] if scenario.expected_commands else 'mlx assistant golden-repos'}"
    else:
        # Generic response
        cmd = (
            scenario.expected_commands[0]
            if scenario.expected_commands
            else "mlx assistant"
        )
        return f"You can achieve this using: {cmd}. This integrates with MLX frameworks for comprehensive functionality."


def _check_scenario_pass(
    evaluation: AIResponseEvaluation, scenario: BenchmarkScenario
) -> bool:
    """Check if evaluation passes the scenario success criteria"""
    criteria = scenario.success_criteria

    # Check each criterion
    for criterion, min_score in criteria.items():
        actual_score = getattr(evaluation.criteria, criterion, 0)
        if actual_score < min_score:
            return False
    return True


def _display_benchmark_results(results: List[Dict], output_dir: str):
    """Display benchmark evaluation results"""
    total_scenarios = len(results)
    passed_scenarios = sum(1 for r in results if r["passed"])
    pass_rate = (passed_scenarios / total_scenarios) * 100

    # Summary
    console.print("\n🎯 [bold cyan]Benchmark Evaluation Complete[/bold cyan]\n")
    console.print(f"📊 Total Scenarios: {total_scenarios}")
    console.print(f"✅ Passed: {passed_scenarios}")
    console.print(f"❌ Failed: {total_scenarios - passed_scenarios}")
    console.print(f"📈 Pass Rate: {pass_rate:.1f}%")

    # Category breakdown
    category_stats = {}
    for result in results:
        category = result["scenario"].category
        if category not in category_stats:
            category_stats[category] = {"total": 0, "passed": 0}
        category_stats[category]["total"] += 1
        if result["passed"]:
            category_stats[category]["passed"] += 1

    console.print("\n📋 [bold]Results by Category[/bold]")
    table = Table(show_header=True)
    table.add_column("Category", style="cyan")
    table.add_column("Passed", style="green")
    table.add_column("Total", style="white")
    table.add_column("Pass Rate", style="yellow")
    for category, stats in category_stats.items():
        pass_rate = (stats["passed"] / stats["total"]) * 100
        table.add_row(
            category.replace("_", " ").title(),
            str(stats["passed"]),
            str(stats["total"]),
            f"{pass_rate:.1f}%",
        )
    console.print(table)

    # Save detailed results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / f"benchmark_results_{int(time.time())}.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "summary": {
                    "total_scenarios": total_scenarios,
                    "passed_scenarios": passed_scenarios,
                    "pass_rate": pass_rate,
                    "category_stats": category_stats,
                },
                "detailed_results": [
                    {
                        "scenario_id": r["scenario"].scenario_id,
                        "category": r["scenario"].category,
                        "difficulty": r["scenario"].difficulty,
                        "query": r["scenario"].user_query,
                        "passed": r["passed"],
                        "final_score": r["evaluation"].final_score,
                        "grade": r["evaluation"].grade,
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
        )
    console.print(f"\n✅ [green]Detailed results saved to: {results_file}[/green]")


def _export_html_report(dashboard: AnalyticsDashboard, output_path: str):
    """Export HTML report (placeholder implementation)"""
    console.print(
        "[yellow]HTML export not yet implemented. Exporting as JSON instead.[/yellow]"
    )
    json_path = output_path.replace(".html", ".json")
    dashboard.export_metrics_report(Path(json_path))


def _export_csv_data(dashboard: AnalyticsDashboard, output_path: str):
    """Export CSV data (placeholder implementation)"""
    console.print(
        "[yellow]CSV export not yet implemented. Exporting as JSON instead.[/yellow]"
    )
    json_path = output_path.replace(".csv", ".json")
    dashboard.export_metrics_report(Path(json_path))


if __name__ == "__main__":
    app()
