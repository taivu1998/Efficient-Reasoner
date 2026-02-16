"""
Phase 3: Pareto Frontier Visualization

Generates the "Money Plot" showing the compute-accuracy trade-off:
- X-axis: Average Tokens per Query (Computational Cost)
- Y-axis: Accuracy (Performance)

The goal is for the GRPO model to be in the top-left (high accuracy, low cost).
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")


@dataclass
class ModelPoint:
    """A point on the Pareto frontier."""
    name: str
    accuracy: float
    avg_tokens: float
    avg_tool_calls: float
    color: str = "blue"
    marker: str = "o"


def load_benchmark_results(results_dir: str) -> Dict[str, dict]:
    """Load benchmark results from JSON files."""
    results = {}

    summary_path = os.path.join(results_dir, "benchmark_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            results = json.load(f)
    else:
        # Load individual result files
        for filename in os.listdir(results_dir):
            if filename.endswith("_results.json"):
                filepath = os.path.join(results_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    results[data['model_name']] = {
                        'accuracy': data['accuracy'],
                        'avg_tokens': data['avg_tokens'],
                        'avg_tool_calls': data['avg_tool_calls']
                    }

    return results


def compute_pareto_frontier(points: List[ModelPoint]) -> List[ModelPoint]:
    """
    Compute the Pareto frontier for accuracy vs tokens.
    Pareto optimal: No other point has both higher accuracy AND lower tokens.
    """
    pareto_points = []

    for point in points:
        is_dominated = False
        for other in points:
            if other.name == point.name:
                continue
            # Check if 'other' dominates 'point'
            # Dominated if other has higher/equal accuracy AND lower/equal tokens
            # (with at least one strict inequality)
            if (other.accuracy >= point.accuracy and
                other.avg_tokens <= point.avg_tokens and
                (other.accuracy > point.accuracy or other.avg_tokens < point.avg_tokens)):
                is_dominated = True
                break

        if not is_dominated:
            pareto_points.append(point)

    # Sort by tokens for drawing the frontier line
    pareto_points.sort(key=lambda p: p.avg_tokens)
    return pareto_points


def plot_pareto_frontier(
    results: Dict[str, dict],
    output_path: str = "eval/pareto_frontier.png",
    title: str = "EfficientReasoning: Compute-Accuracy Trade-off",
    show_frontier: bool = True,
    figsize: Tuple[int, int] = (10, 8)
) -> Optional[str]:
    """
    Generate the Pareto frontier plot.

    Args:
        results: Dictionary of model results
        output_path: Path to save the plot
        title: Plot title
        show_frontier: Whether to draw the Pareto frontier line
        figsize: Figure size

    Returns:
        Path to saved figure, or None if matplotlib not available
    """
    if not HAS_MATPLOTLIB:
        print("Cannot generate plot: matplotlib not installed")
        return None

    # Define colors and markers for different model types
    model_styles = {
        'base': {'color': '#ff6b6b', 'marker': 's', 'label': 'Base Model (Zero-Shot)'},
        'sft': {'color': '#4ecdc4', 'marker': '^', 'label': 'SFT Model (Always Search)'},
        'grpo': {'color': '#45b7d1', 'marker': 'o', 'label': 'GRPO Model (Efficient)'},
        'default': {'color': '#95a5a6', 'marker': 'D', 'label': 'Other'}
    }

    # Create model points
    points = []
    for name, data in results.items():
        # Determine style based on name
        name_lower = name.lower()
        if 'base' in name_lower or 'zero' in name_lower:
            style = model_styles['base']
        elif 'sft' in name_lower:
            style = model_styles['sft']
        elif 'grpo' in name_lower or 'efficient' in name_lower:
            style = model_styles['grpo']
        else:
            style = model_styles['default']

        points.append(ModelPoint(
            name=name,
            accuracy=data['accuracy'],
            avg_tokens=data['avg_tokens'],
            avg_tool_calls=data.get('avg_tool_calls', 0),
            color=style['color'],
            marker=style['marker']
        ))

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each point
    for point in points:
        ax.scatter(
            point.avg_tokens,
            point.accuracy * 100,  # Convert to percentage
            c=point.color,
            marker=point.marker,
            s=200,
            edgecolors='black',
            linewidths=1.5,
            zorder=5
        )
        # Add label
        ax.annotate(
            point.name,
            (point.avg_tokens, point.accuracy * 100),
            xytext=(10, 5),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold'
        )

    # Draw Pareto frontier
    if show_frontier and len(points) > 1:
        pareto_points = compute_pareto_frontier(points)
        if len(pareto_points) > 1:
            frontier_x = [p.avg_tokens for p in pareto_points]
            frontier_y = [p.accuracy * 100 for p in pareto_points]
            ax.plot(
                frontier_x, frontier_y,
                'k--', linewidth=2, alpha=0.5,
                label='Pareto Frontier', zorder=1
            )
            # Fill area under frontier
            ax.fill_between(
                frontier_x, frontier_y, 0,
                alpha=0.1, color='green',
                label='Efficient Region'
            )

    # Styling
    ax.set_xlabel('Average Tokens per Query', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Set axis limits with padding
    if points:
        x_vals = [p.avg_tokens for p in points]
        y_vals = [p.accuracy * 100 for p in points]
        x_margin = (max(x_vals) - min(x_vals)) * 0.1 or 50
        y_margin = (max(y_vals) - min(y_vals)) * 0.1 or 5

        ax.set_xlim(min(x_vals) - x_margin, max(x_vals) + x_margin)
        ax.set_ylim(max(0, min(y_vals) - y_margin), min(100, max(y_vals) + y_margin))

    # Grid
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Legend
    handles = []
    for style_name, style in model_styles.items():
        if any(style['color'] == p.color for p in points):
            handles.append(mpatches.Patch(
                color=style['color'],
                label=style['label']
            ))

    if handles:
        ax.legend(handles=handles, loc='lower right', fontsize=10)

    # Add annotation for optimal region
    ax.annotate(
        'Optimal\nRegion',
        xy=(0.1, 0.9),
        xycoords='axes fraction',
        fontsize=12,
        fontweight='bold',
        color='green',
        alpha=0.7
    )

    # Arrow pointing to optimal region
    ax.annotate(
        '',
        xy=(0.05, 0.95),
        xytext=(0.15, 0.85),
        xycoords='axes fraction',
        arrowprops=dict(
            arrowstyle='->',
            color='green',
            alpha=0.5,
            lw=2
        )
    )

    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Pareto frontier plot saved to: {output_path}")

    plt.close()
    return output_path


def generate_report(
    results: Dict[str, dict],
    output_path: str = "eval/benchmark_report.md"
) -> str:
    """
    Generate a markdown report of the benchmark results.

    Args:
        results: Dictionary of model results
        output_path: Path to save the report

    Returns:
        Path to saved report
    """
    report = []
    report.append("# EfficientReasoning Benchmark Report\n")
    report.append("## Results Summary\n")
    report.append("| Model | Accuracy | Avg Tokens | Avg Tool Calls |")
    report.append("|-------|----------|------------|----------------|")

    for name, data in sorted(results.items(), key=lambda x: -x[1]['accuracy']):
        report.append(
            f"| {name} | {data['accuracy']:.1%} | "
            f"{data['avg_tokens']:.1f} | {data.get('avg_tool_calls', 0):.2f} |"
        )

    report.append("\n## Analysis\n")

    # Find best models
    if results:
        best_accuracy = max(results.items(), key=lambda x: x[1]['accuracy'])
        most_efficient = min(results.items(), key=lambda x: x[1]['avg_tokens'])

        report.append(f"- **Highest Accuracy**: {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.1%})")
        report.append(f"- **Most Efficient**: {most_efficient[0]} ({most_efficient[1]['avg_tokens']:.1f} tokens)")

        # Check for Pareto improvements
        if len(results) > 1:
            report.append("\n### Pareto Analysis\n")
            points = [
                ModelPoint(
                    name=name,
                    accuracy=data['accuracy'],
                    avg_tokens=data['avg_tokens'],
                    avg_tool_calls=data.get('avg_tool_calls', 0)
                )
                for name, data in results.items()
            ]
            pareto = compute_pareto_frontier(points)
            pareto_names = [p.name for p in pareto]
            report.append(f"Models on Pareto frontier: {', '.join(pareto_names)}")

    report.append("\n---\n")
    report.append("*Generated by EfficientReasoning benchmark suite*")

    # Save report
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"Report saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate Pareto frontier plot")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="eval/results",
        help="Directory containing benchmark results"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval/pareto_frontier.png",
        help="Output path for the plot"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="EfficientReasoning: Compute-Accuracy Trade-off",
        help="Plot title"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Also generate markdown report"
    )
    args = parser.parse_args()

    # Load results
    print(f"Loading results from {args.results_dir}...")
    results = load_benchmark_results(args.results_dir)

    if not results:
        print("No benchmark results found. Run benchmark.py first.")
        return

    print(f"Found results for {len(results)} models")

    # Generate plot
    plot_pareto_frontier(
        results=results,
        output_path=args.output,
        title=args.title
    )

    # Generate report if requested
    if args.report:
        report_path = args.output.replace('.png', '_report.md')
        generate_report(results, report_path)


if __name__ == "__main__":
    main()
