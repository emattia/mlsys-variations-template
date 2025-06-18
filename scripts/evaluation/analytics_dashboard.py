#!/usr/bin/env python3
"""
📊 MLX AI Evaluation Analytics Dashboard

Real-time analytics and performance insights for MLX AI assistant evaluation.
Repository infrastructure for tracking AI quality metrics and trends.

Features:
- Performance trend analysis and visualization
- A/B testing results comparison
- Regression detection and alerting
- Interactive CLI dashboard with rich formatting
- Export capabilities for reporting
"""

import json
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import pandas as pd

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn
from rich.tree import Tree
from rich.align import Align
import typer

console = Console()

class AnalyticsDashboard:
    """Real-time analytics dashboard for MLX AI evaluation metrics"""
    
    def __init__(self, data_dir: Path = Path("data/evaluations")):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.evaluations = []
        self.metrics_cache = {}
        
    def load_evaluation_data(self) -> List[Dict[str, Any]]:
        """Load all evaluation data from storage"""
        evaluations = []
        
        for eval_file in self.data_dir.glob("eval_*.json"):
            try:
                with open(eval_file) as f:
                    evaluation = json.load(f)
                    evaluations.append(evaluation)
            except Exception as e:
                console.print(f"[red]Error loading {eval_file}: {e}[/red]")
        
        # Sort by timestamp
        evaluations.sort(key=lambda x: x.get('timestamp', ''))
        self.evaluations = evaluations
        return evaluations
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        if not self.evaluations:
            return {}
        
        metrics = {
            "total_evaluations": len(self.evaluations),
            "date_range": {
                "start": self.evaluations[0].get('timestamp', ''),
                "end": self.evaluations[-1].get('timestamp', '')
            },
            "overall_performance": {},
            "category_performance": defaultdict(list),
            "trend_analysis": {},
            "quality_distribution": defaultdict(int),
            "framework_coverage": defaultdict(int),
            "command_accuracy_trend": [],
            "recent_performance": {},
            "regression_alerts": []
        }
        
        # Aggregate scores by dimension
        dimension_scores = defaultdict(list)
        category_scores = defaultdict(lambda: defaultdict(list))
        recent_scores = defaultdict(list)  # Last 7 days
        
        recent_cutoff = datetime.now() - timedelta(days=7)
        
        for eval_data in self.evaluations:
            final_score = eval_data.get('final_score', 0)
            grade = eval_data.get('grade', 'F')
            category = eval_data.get('query_category', 'general')
            eval_time = datetime.fromisoformat(eval_data.get('timestamp', ''))
            
            # Overall metrics
            dimension_scores['final_score'].append(final_score)
            metrics['quality_distribution'][grade] += 1
            
            # Category metrics
            category_scores[category]['final_score'].append(final_score)
            
            # Framework coverage
            frameworks = eval_data.get('framework_coverage', [])
            for framework in frameworks:
                metrics['framework_coverage'][framework] += 1
            
            # Recent performance
            if eval_time >= recent_cutoff:
                recent_scores['final_score'].append(final_score)
            
            # Individual criteria scores
            criteria = eval_data.get('criteria', {})
            for key, value in criteria.items():
                if isinstance(value, (int, float)) and key != 'final_score':
                    dimension_scores[key].append(value * 100)  # Convert to percentage
                    category_scores[category][key].append(value * 100)
                    if eval_time >= recent_cutoff:
                        recent_scores[key].append(value * 100)
            
            # Command accuracy trend
            cmd_details = eval_data.get('command_accuracy_details', {})
            cmd_accuracy = cmd_details.get('accuracy_percentage', 0)
            metrics['command_accuracy_trend'].append({
                'timestamp': eval_data.get('timestamp'),
                'accuracy': cmd_accuracy,
                'category': category
            })
        
        # Calculate overall performance
        for dimension, scores in dimension_scores.items():
            if scores:
                metrics['overall_performance'][dimension] = {
                    'mean': statistics.mean(scores),
                    'median': statistics.median(scores),
                    'std': statistics.stdev(scores) if len(scores) > 1 else 0,
                    'min': min(scores),
                    'max': max(scores),
                    'count': len(scores)
                }
        
        # Category performance
        for category, dimensions in category_scores.items():
            metrics['category_performance'][category] = {}
            for dimension, scores in dimensions.items():
                if scores:
                    metrics['category_performance'][category][dimension] = {
                        'mean': statistics.mean(scores),
                        'count': len(scores)
                    }
        
        # Recent performance (last 7 days)
        for dimension, scores in recent_scores.items():
            if scores:
                metrics['recent_performance'][dimension] = {
                    'mean': statistics.mean(scores),
                    'count': len(scores),
                    'trend': self._calculate_trend(dimension_scores[dimension], scores)
                }
        
        # Regression detection
        metrics['regression_alerts'] = self._detect_regressions(dimension_scores)
        
        self.metrics_cache = metrics
        return metrics
    
    def _calculate_trend(self, all_scores: List[float], recent_scores: List[float]) -> str:
        """Calculate if recent performance is improving/declining"""
        if len(all_scores) < 10 or len(recent_scores) < 3:
            return "insufficient_data"
        
        overall_mean = statistics.mean(all_scores)
        recent_mean = statistics.mean(recent_scores)
        
        improvement_threshold = overall_mean * 0.05  # 5% improvement
        decline_threshold = overall_mean * 0.05      # 5% decline
        
        if recent_mean > overall_mean + improvement_threshold:
            return "improving"
        elif recent_mean < overall_mean - decline_threshold:
            return "declining"
        else:
            return "stable"
    
    def _detect_regressions(self, dimension_scores: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        """Detect performance regressions"""
        alerts = []
        
        for dimension, scores in dimension_scores.items():
            if len(scores) < 10:
                continue
            
            # Check last 5 vs previous average
            recent_scores = scores[-5:]
            historical_scores = scores[:-5]
            
            if len(historical_scores) > 0:
                historical_mean = statistics.mean(historical_scores)
                recent_mean = statistics.mean(recent_scores)
                
                # Alert if recent performance is significantly worse
                decline_threshold = historical_mean * 0.10  # 10% decline
                if recent_mean < historical_mean - decline_threshold:
                    alerts.append({
                        'dimension': dimension,
                        'historical_mean': historical_mean,
                        'recent_mean': recent_mean,
                        'decline_percentage': ((historical_mean - recent_mean) / historical_mean) * 100,
                        'severity': 'high' if recent_mean < historical_mean - (decline_threshold * 2) else 'medium'
                    })
        
        return alerts
    
    def display_overview_dashboard(self):
        """Display comprehensive overview dashboard"""
        metrics = self.calculate_performance_metrics()
        
        if not metrics:
            console.print(Panel("[yellow]No evaluation data found. Run some evaluations first![/yellow]"))
            return
        
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        layout["left"].split_column(
            Layout(name="performance", ratio=1),
            Layout(name="categories", ratio=1)
        )
        
        layout["right"].split_column(
            Layout(name="trends", ratio=1),
            Layout(name="alerts", ratio=1)
        )
        
        # Header
        header_text = f"🎯 MLX AI Evaluation Analytics Dashboard\n📊 Total Evaluations: {metrics['total_evaluations']} | 📅 Data Range: {metrics['date_range']['start'][:10]} to {metrics['date_range']['end'][:10]}"
        layout["header"] = Panel(Align.center(header_text), style="bold blue")
        
        # Performance Overview
        layout["performance"] = self._create_performance_panel(metrics)
        
        # Category Performance
        layout["categories"] = self._create_category_panel(metrics)
        
        # Trends
        layout["trends"] = self._create_trends_panel(metrics)
        
        # Alerts
        layout["alerts"] = self._create_alerts_panel(metrics)
        
        # Footer
        footer_text = f"🔄 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 💡 Use 'mlx-eval' commands for detailed analysis"
        layout["footer"] = Panel(Align.center(footer_text), style="dim")
        
        console.print(layout)
    
    def _create_performance_panel(self, metrics: Dict[str, Any]) -> Panel:
        """Create overall performance metrics panel"""
        table = Table(title="📈 Overall Performance", show_header=True)
        table.add_column("Dimension", style="cyan")
        table.add_column("Mean", style="green")
        table.add_column("Trend", style="yellow")
        table.add_column("Grade", style="magenta")
        
        overall = metrics.get('overall_performance', {})
        recent = metrics.get('recent_performance', {})
        
        for dimension, stats in overall.items():
            if dimension == 'final_score':
                continue
                
            mean_score = stats['mean']
            grade = self._score_to_grade(mean_score)
            
            trend_info = recent.get(dimension, {})
            trend = trend_info.get('trend', 'unknown')
            trend_icon = {"improving": "📈", "declining": "📉", "stable": "➡️"}.get(trend, "❓")
            
            table.add_row(
                dimension.replace('_', ' ').title(),
                f"{mean_score:.1f}%",
                f"{trend_icon} {trend}",
                grade
            )
        
        return Panel(table, border_style="green")
    
    def _create_category_panel(self, metrics: Dict[str, Any]) -> Panel:
        """Create category performance panel"""
        table = Table(title="📋 Performance by Category", show_header=True)
        table.add_column("Category", style="cyan")
        table.add_column("Evaluations", style="white")
        table.add_column("Avg Score", style="green") 
        table.add_column("Grade", style="magenta")
        
        category_perf = metrics.get('category_performance', {})
        
        for category, performance in category_perf.items():
            final_score_data = performance.get('final_score', {})
            if final_score_data:
                avg_score = final_score_data['mean']
                count = final_score_data['count']
                grade = self._score_to_grade(avg_score)
                
                table.add_row(
                    category.replace('_', ' ').title(),
                    str(count),
                    f"{avg_score:.1f}%",
                    grade
                )
        
        return Panel(table, border_style="blue")
    
    def _create_trends_panel(self, metrics: Dict[str, Any]) -> Panel:
        """Create trends analysis panel"""
        tree = Tree("📊 Trend Analysis")
        
        framework_coverage = metrics.get('framework_coverage', {})
        if framework_coverage:
            frameworks_tree = tree.add("🧩 Framework Coverage")
            for framework, count in sorted(framework_coverage.items(), key=lambda x: x[1], reverse=True):
                frameworks_tree.add(f"{framework}: {count} evaluations")
        
        quality_dist = metrics.get('quality_distribution', {})
        if quality_dist:
            grades_tree = tree.add("🎯 Grade Distribution")
            total_evals = sum(quality_dist.values())
            for grade in ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'F']:
                count = quality_dist.get(grade, 0)
                if count > 0:
                    percentage = (count / total_evals) * 100
                    grades_tree.add(f"{grade}: {count} ({percentage:.1f}%)")
        
        return Panel(tree, title="📈 Trends", border_style="yellow")
    
    def _create_alerts_panel(self, metrics: Dict[str, Any]) -> Panel:
        """Create regression alerts panel"""
        alerts = metrics.get('regression_alerts', [])
        
        if not alerts:
            content = "[green]✅ No performance regressions detected![/green]"
        else:
            table = Table(show_header=True)
            table.add_column("Dimension", style="cyan")
            table.add_column("Decline", style="red")
            table.add_column("Severity", style="yellow")
            
            for alert in alerts:
                severity_icon = "🚨" if alert['severity'] == 'high' else "⚠️"
                table.add_row(
                    alert['dimension'].replace('_', ' ').title(),
                    f"{alert['decline_percentage']:.1f}%",
                    f"{severity_icon} {alert['severity']}"
                )
            content = table
        
        return Panel(content, title="🚨 Regression Alerts", border_style="red")
    
    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        if score >= 95: return "A+"
        elif score >= 90: return "A"
        elif score >= 85: return "A-"
        elif score >= 80: return "B+"
        elif score >= 75: return "B"
        elif score >= 70: return "B-"
        elif score >= 65: return "C+"
        elif score >= 60: return "C"
        else: return "F"
    
    def display_detailed_analysis(self, category: Optional[str] = None):
        """Display detailed analysis for specific category or overall"""
        metrics = self.calculate_performance_metrics()
        
        if category:
            console.print(f"\n🔍 [bold cyan]Detailed Analysis: {category.title()}[/bold cyan]\n")
            self._display_category_details(metrics, category)
        else:
            console.print(f"\n🔍 [bold cyan]Detailed Analysis: Overall Performance[/bold cyan]\n")
            self._display_overall_details(metrics)
    
    def _display_category_details(self, metrics: Dict[str, Any], category: str):
        """Display detailed analysis for specific category"""
        category_data = metrics.get('category_performance', {}).get(category)
        
        if not category_data:
            console.print(f"[red]No data found for category: {category}[/red]")
            return
        
        table = Table(title=f"Performance Breakdown: {category.title()}", show_header=True)
        table.add_column("Dimension", style="cyan")
        table.add_column("Mean Score", style="green")
        table.add_column("Evaluations", style="white")
        table.add_column("Grade", style="magenta")
        
        for dimension, stats in category_data.items():
            mean_score = stats['mean']
            count = stats['count']
            grade = self._score_to_grade(mean_score)
            
            table.add_row(
                dimension.replace('_', ' ').title(),
                f"{mean_score:.1f}%",
                str(count),
                grade
            )
        
        console.print(table)
        
        # Show recent evaluations for this category
        recent_evals = [e for e in self.evaluations[-10:] if e.get('query_category') == category]
        if recent_evals:
            console.print(f"\n📋 [bold]Recent Evaluations ({len(recent_evals)})[/bold]")
            for i, eval_data in enumerate(recent_evals[-5:], 1):
                timestamp = eval_data.get('timestamp', '')[:19]
                score = eval_data.get('final_score', 0)
                grade = eval_data.get('grade', 'F')
                query = eval_data.get('user_query', '')[:60] + "..." if len(eval_data.get('user_query', '')) > 60 else eval_data.get('user_query', '')
                
                console.print(f"  {i}. [{timestamp}] {score:.1f}% ({grade}) - {query}")
    
    def _display_overall_details(self, metrics: Dict[str, Any]):
        """Display detailed overall analysis"""
        overall = metrics.get('overall_performance', {})
        
        # Performance table
        table = Table(title="📊 Comprehensive Performance Analysis", show_header=True)
        table.add_column("Dimension", style="cyan")
        table.add_column("Mean", style="green")
        table.add_column("Median", style="blue")
        table.add_column("Std Dev", style="yellow")
        table.add_column("Range", style="magenta")
        table.add_column("Count", style="white")
        
        for dimension, stats in overall.items():
            table.add_row(
                dimension.replace('_', ' ').title(),
                f"{stats['mean']:.1f}%",
                f"{stats['median']:.1f}%",
                f"{stats['std']:.1f}",
                f"{stats['min']:.1f}-{stats['max']:.1f}%",
                str(stats['count'])
            )
        
        console.print(table)
        
        # Command accuracy trend
        cmd_trend = metrics.get('command_accuracy_trend', [])
        if cmd_trend:
            console.print(f"\n📈 [bold]Command Accuracy Trend (Last 10)[/bold]")
            for i, trend_point in enumerate(cmd_trend[-10:], 1):
                timestamp = trend_point.get('timestamp', '')[:16]
                accuracy = trend_point.get('accuracy', 0)
                category = trend_point.get('category', 'unknown')
                console.print(f"  {i}. [{timestamp}] {accuracy:.1f}% - {category}")
    
    def export_metrics_report(self, output_path: Path):
        """Export comprehensive metrics report"""
        metrics = self.calculate_performance_metrics()
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_evaluations": metrics.get('total_evaluations', 0),
                "date_range": metrics.get('date_range', {}),
                "overall_grade": self._score_to_grade(metrics.get('overall_performance', {}).get('final_score', {}).get('mean', 0))
            },
            "performance_metrics": metrics.get('overall_performance', {}),
            "category_performance": dict(metrics.get('category_performance', {})),
            "recent_performance": metrics.get('recent_performance', {}),
            "trend_analysis": {
                "framework_coverage": dict(metrics.get('framework_coverage', {})),
                "quality_distribution": dict(metrics.get('quality_distribution', {})),
                "command_accuracy_trend": metrics.get('command_accuracy_trend', [])
            },
            "regression_alerts": metrics.get('regression_alerts', [])
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        console.print(f"✅ [green]Metrics report exported to: {output_path}[/green]")

# CLI interface
app = typer.Typer(help="MLX AI Evaluation Analytics Dashboard")

@app.command("dashboard")
def show_dashboard():
    """Show the interactive analytics dashboard"""
    dashboard = AnalyticsDashboard()
    dashboard.load_evaluation_data()
    dashboard.display_overview_dashboard()

@app.command("analyze")
def analyze_performance(
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Analyze specific category")
):
    """Show detailed performance analysis"""
    dashboard = AnalyticsDashboard()
    dashboard.load_evaluation_data()
    dashboard.display_detailed_analysis(category)

@app.command("export")
def export_report(
    output: str = typer.Option("reports/mlx_ai_metrics_report.json", "--output", "-o", help="Output file path")
):
    """Export comprehensive metrics report"""
    dashboard = AnalyticsDashboard()
    dashboard.load_evaluation_data()
    dashboard.export_metrics_report(Path(output))

if __name__ == "__main__":
    app() 