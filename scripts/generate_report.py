#!/usr/bin/env python3
"""
Generate benchmark reports from results.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


def load_benchmark_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all benchmark result files from directory."""
    results = []
    
    for result_file in results_dir.glob("benchmark_*.json"):
        try:
            with open(result_file) as f:
                data = json.load(f)
                data['filename'] = result_file.name
                results.append(data)
        except Exception as e:
            print(f"Error loading {result_file}: {e}")
    
    return results


def generate_accuracy_comparison(results: List[Dict[str, Any]], output_dir: Path):
    """Generate accuracy comparison chart."""
    plt.figure(figsize=(12, 8))
    
    all_strategies = set()
    for result in results:
        all_strategies.update(result.get('strategies', {}).keys())
    
    strategy_accuracies = {strategy: [] for strategy in all_strategies}
    
    for result in results:
        for strategy in all_strategies:
            if strategy in result.get('strategies', {}):
                summary = result['strategies'][strategy].get('summary', {})
                accuracy = summary.get('accuracy', 0.0)
                strategy_accuracies[strategy].append(accuracy)
            else:
                strategy_accuracies[strategy].append(0.0)
    
    # Create box plot
    data_to_plot = [accuracies for accuracies in strategy_accuracies.values() if any(acc > 0 for acc in accuracies)]
    labels = [strategy for strategy, accuracies in strategy_accuracies.items() if any(acc > 0 for acc in accuracies)]
    
    plt.boxplot(data_to_plot, labels=labels)
    plt.title('Strategy Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_performance_metrics(results: List[Dict[str, Any]], output_dir: Path):
    """Generate performance metrics table."""
    data = []
    
    for result in results:
        model = result.get('model', 'unknown')
        timestamp = datetime.fromtimestamp(result.get('timestamp', 0))
        
        for strategy_name, strategy_data in result.get('strategies', {}).items():
            if 'summary' in strategy_data:
                summary = strategy_data['summary']
                data.append({
                    'Timestamp': timestamp,
                    'Model': model,
                    'Strategy': strategy_name,
                    'Accuracy': summary.get('accuracy', 0.0),
                    'Avg_Confidence': summary.get('avg_confidence', 0.0),
                    'Avg_Time': summary.get('avg_execution_time', 0.0),
                    'Total_Tokens': summary.get('total_tokens', 0),
                    'Failed_Problems': summary.get('failed_problems', 0)
                })
    
    if data:
        df = pd.DataFrame(data)
        
        # Summary statistics
        summary_stats = df.groupby(['Model', 'Strategy']).agg({
            'Accuracy': ['mean', 'std', 'max'],
            'Avg_Confidence': 'mean',
            'Avg_Time': 'mean',
            'Total_Tokens': 'mean'
        }).round(4)
        
        # Save to CSV
        df.to_csv(output_dir / 'detailed_results.csv', index=False)
        summary_stats.to_csv(output_dir / 'summary_stats.csv')
        
        return df
    
    return None


def generate_trend_analysis(results: List[Dict[str, Any]], output_dir: Path):
    """Generate trend analysis over time."""
    if len(results) < 2:
        print("Not enough results for trend analysis")
        return
    
    plt.figure(figsize=(14, 10))
    
    # Sort results by timestamp
    results.sort(key=lambda x: x.get('timestamp', 0))
    
    timestamps = [datetime.fromtimestamp(r.get('timestamp', 0)) for r in results]
    
    # Track accuracy trends for each strategy
    strategy_trends = {}
    
    for result in results:
        for strategy_name, strategy_data in result.get('strategies', {}).items():
            if 'summary' in strategy_data:
                if strategy_name not in strategy_trends:
                    strategy_trends[strategy_name] = []
                accuracy = strategy_data['summary'].get('accuracy', 0.0)
                strategy_trends[strategy_name].append(accuracy)
    
    # Plot trends
    for strategy_name, accuracies in strategy_trends.items():
        if len(accuracies) == len(timestamps):  # Only plot if we have data for all runs
            plt.plot(timestamps, accuracies, marker='o', label=strategy_name)
    
    plt.title('Strategy Performance Trends Over Time')
    plt.xlabel('Date')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_dir / 'performance_trends.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_html_report(results: List[Dict[str, Any]], output_dir: Path, df: pd.DataFrame = None):
    """Generate comprehensive HTML report."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ThinkMesh Benchmark Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }}
            .metric {{ background: #f8f9fa; padding: 20px; margin: 10px 0; border-radius: 5px; }}
            .metric h3 {{ margin-top: 0; color: #2980b9; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; }}
            .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <h1>ThinkMesh Benchmark Report</h1>
        <p class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Executive Summary</h2>
        <div class="metric">
            <h3>Benchmark Overview</h3>
            <p><strong>Total Benchmark Runs:</strong> {len(results)}</p>
            <p><strong>Models Tested:</strong> {len(set(r.get('model', 'unknown') for r in results))}</p>
            <p><strong>Strategies Tested:</strong> {len(set().union(*(r.get('strategies', {}).keys() for r in results)))}</p>
        </div>
    """
    
    if df is not None:
        # Best performing strategies
        best_strategies = df.groupby('Strategy')['Accuracy'].mean().sort_values(ascending=False)
        html_content += f"""
        <div class="metric">
            <h3>Top Performing Strategies</h3>
            <ol>
        """
        for strategy, accuracy in best_strategies.head().items():
            html_content += f"<li><strong>{strategy}:</strong> {accuracy:.1%} average accuracy</li>"
        
        html_content += """
            </ol>
        </div>
        """
        
        # Performance statistics
        html_content += """
        <div class="metric">
            <h3>Performance Statistics</h3>
            <table>
                <tr>
                    <th>Strategy</th>
                    <th>Avg Accuracy</th>
                    <th>Avg Confidence</th>
                    <th>Avg Time (sec)</th>
                    <th>Avg Tokens</th>
                </tr>
        """
        
        strategy_stats = df.groupby('Strategy').agg({
            'Accuracy': 'mean',
            'Avg_Confidence': 'mean', 
            'Avg_Time': 'mean',
            'Total_Tokens': 'mean'
        }).round(3)
        
        for strategy, row in strategy_stats.iterrows():
            html_content += f"""
                <tr>
                    <td>{strategy}</td>
                    <td>{row['Accuracy']:.1%}</td>
                    <td>{row['Avg_Confidence']:.3f}</td>
                    <td>{row['Avg_Time']:.2f}</td>
                    <td>{int(row['Total_Tokens'])}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </div>
        """
    
    # Add charts
    html_content += """
        <h2>Visualizations</h2>
        <h3>Strategy Accuracy Comparison</h3>
        <img src="accuracy_comparison.png" alt="Accuracy Comparison">
        
        <h3>Performance Trends</h3>
        <img src="performance_trends.png" alt="Performance Trends">
    """
    
    # Recent results
    if results:
        latest_result = max(results, key=lambda x: x.get('timestamp', 0))
        html_content += f"""
        <h2>Latest Benchmark Results</h2>
        <div class="metric">
            <h3>Run Details</h3>
            <p><strong>Model:</strong> {latest_result.get('model', 'unknown')}</p>
            <p><strong>Dataset:</strong> {latest_result.get('dataset', 'unknown')}</p>
            <p><strong>Problems:</strong> {latest_result.get('num_problems', 'unknown')}</p>
            <p><strong>Timestamp:</strong> {datetime.fromtimestamp(latest_result.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <h3>Strategy Results</h3>
        <table>
            <tr>
                <th>Strategy</th>
                <th>Accuracy</th>
                <th>Confidence</th>
                <th>Execution Time</th>
                <th>Status</th>
            </tr>
        """
        
        for strategy_name, strategy_data in latest_result.get('strategies', {}).items():
            if 'summary' in strategy_data:
                summary = strategy_data['summary']
                exec_time = strategy_data.get('execution_time', 0)
                html_content += f"""
                    <tr>
                        <td>{strategy_name}</td>
                        <td>{summary.get('accuracy', 0):.1%}</td>
                        <td>{summary.get('avg_confidence', 0):.3f}</td>
                        <td>{exec_time:.1f}s</td>
                        <td>✓ Success</td>
                    </tr>
                """
            else:
                error = strategy_data.get('error', 'Unknown error')
                html_content += f"""
                    <tr>
                        <td>{strategy_name}</td>
                        <td>-</td>
                        <td>-</td>
                        <td>-</td>
                        <td>✗ Failed: {error}</td>
                    </tr>
                """
        
        html_content += """
            </table>
        """
    
    html_content += """
        </body>
    </html>
    """
    
    with open(output_dir / 'benchmark_report.html', 'w') as f:
        f.write(html_content)


def main():
    """Generate comprehensive benchmark report."""
    parser = argparse.ArgumentParser(description="Generate benchmark reports")
    parser.add_argument("results_dir", help="Directory containing benchmark results")
    parser.add_argument("--output-dir", default="reports", help="Output directory for reports")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return 1
    
    print(f"Loading benchmark results from: {results_dir}")
    results = load_benchmark_results(results_dir)
    
    if not results:
        print("No benchmark results found")
        return 1
    
    print(f"Loaded {len(results)} benchmark results")
    
    try:
        # Generate charts
        print("Generating accuracy comparison...")
        generate_accuracy_comparison(results, output_dir)
        
        print("Generating performance metrics...")
        df = generate_performance_metrics(results, output_dir)
        
        print("Generating trend analysis...")
        generate_trend_analysis(results, output_dir)
        
        print("Generating HTML report...")
        generate_html_report(results, output_dir, df)
        
        print(f"\nReports generated in: {output_dir}")
        print("Files created:")
        for file in output_dir.glob("*"):
            print(f"  - {file.name}")
        
    except ImportError as e:
        print(f"Missing dependencies for report generation: {e}")
        print("Install with: pip install matplotlib pandas")
        return 1
    
    except Exception as e:
        print(f"Error generating reports: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
