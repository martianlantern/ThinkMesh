#!/usr/bin/env python3
"""
Comprehensive benchmark runner for ThinkMesh.
"""
import asyncio
import argparse
import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add tests directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

from thinkmesh import ThinkConfig, ModelSpec, StrategySpec
from benchmarks.gsm8k_utils import (
    create_gsm8k_sample_dataset, run_gsm8k_benchmark,
    load_gsm8k_problems, BenchmarkSummary
)


def get_model_configs() -> Dict[str, ModelSpec]:
    """Get available model configurations."""
    return {
        "small_cpu": ModelSpec(
            backend="transformers",
            model_name="microsoft/DialoGPT-small",
            max_tokens=256,
            temperature=0.7,
            extra={"device": "cpu", "batch_size": 4}
        ),
        "small_gpu": ModelSpec(
            backend="transformers",
            model_name="microsoft/DialoGPT-small",
            max_tokens=256,
            temperature=0.7,
            extra={"device": "cuda:0", "dtype": "float16", "batch_size": 8}
        ),
        "medium_gpu": ModelSpec(
            backend="transformers", 
            model_name="microsoft/DialoGPT-medium",
            max_tokens=512,
            temperature=0.7,
            extra={"device": "cuda:0", "dtype": "float16", "batch_size": 12}
        ),
        "large_gpu": ModelSpec(
            backend="transformers",
            model_name="microsoft/DialoGPT-large", 
            max_tokens=512,
            temperature=0.7,
            extra={"device": "cuda:0", "dtype": "float16", "batch_size": 16}
        )
    }


def get_strategy_configs() -> Dict[str, StrategySpec]:
    """Get available strategy configurations."""
    return {
        "self_consistency_small": StrategySpec(
            name="self_consistency",
            parallel=4,
            max_steps=1
        ),
        "self_consistency_large": StrategySpec(
            name="self_consistency",
            parallel=16,
            max_steps=1
        ),
        "deepconf_small": StrategySpec(
            name="deepconf",
            parallel=6,
            max_steps=2,
            deepconf={"k": 3, "tau_low": -1.2, "tau_ent": 2.5, "realloc_top_p": 0.4}
        ),
        "deepconf_large": StrategySpec(
            name="deepconf",
            parallel=12,
            max_steps=2,
            deepconf={"k": 5, "tau_low": -1.0, "tau_ent": 2.0, "realloc_top_p": 0.5}
        ),
        "debate": StrategySpec(
            name="debate",
            parallel=4,
            max_steps=2,
            debate={"rounds": 2}
        ),
        "tree": StrategySpec(
            name="tree",
            parallel=8,
            max_steps=3,
            tree={"branches": 4, "depth": 2}
        ),
        "graph": StrategySpec(
            name="graph",
            parallel=6,
            max_steps=2
        )
    }


async def run_benchmark_suite(
    model_name: str,
    strategy_names: List[str],
    dataset: str = "sample",
    dataset_path: Optional[str] = None,
    num_problems: int = 10,
    output_dir: str = "benchmark_results",
    max_concurrent: int = 1
) -> Dict[str, Any]:
    """
    Run a comprehensive benchmark suite.
    
    Args:
        model_name: Name of model config to use
        strategy_names: List of strategy names to test
        dataset: Dataset to use ("sample" or "gsm8k")
        dataset_path: Path to GSM8K dataset file
        num_problems: Number of problems to test
        output_dir: Output directory for results
        max_concurrent: Maximum concurrent executions
        
    Returns:
        Dictionary with all benchmark results
    """
    print(f"Running ThinkMesh Benchmark Suite")
    print(f"Model: {model_name}")
    print(f"Strategies: {', '.join(strategy_names)}")
    print(f"Dataset: {dataset}")
    print(f"Problems: {num_problems}")
    print("="*60)
    
    # Get configurations
    model_configs = get_model_configs()
    strategy_configs = get_strategy_configs()
    
    if model_name not in model_configs:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_configs.keys())}")
    
    model_spec = model_configs[model_name]
    
    # Load problems
    if dataset == "sample":
        problems = create_gsm8k_sample_dataset()[:num_problems]
    elif dataset == "gsm8k":
        if not dataset_path:
            raise ValueError("GSM8K dataset path required for 'gsm8k' dataset")
        problems = load_gsm8k_problems(dataset_path, limit=num_problems)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    print(f"Loaded {len(problems)} problems")
    
    # Run benchmarks
    all_results = {
        "timestamp": time.time(),
        "model": model_name,
        "dataset": dataset,
        "num_problems": len(problems),
        "strategies": {}
    }
    
    for strategy_name in strategy_names:
        if strategy_name not in strategy_configs:
            print(f"Warning: Unknown strategy '{strategy_name}', skipping")
            continue
        
        strategy_spec = strategy_configs[strategy_name]
        
        config = ThinkConfig(
            model=model_spec,
            strategy=strategy_spec,
            budgets={"wall_clock_s": 600, "tokens": 20000}
        )
        
        print(f"\nRunning {strategy_name}...")
        print(f"  Parallel: {strategy_spec.parallel}")
        print(f"  Max steps: {strategy_spec.max_steps}")
        
        try:
            start_time = time.time()
            
            def progress_callback(current: int, total: int, result):
                accuracy = sum(1 for r in all_results["strategies"].get(strategy_name, {}).get("results", []) + [result] if r.is_correct) / current
                print(f"  Progress: {current}/{total} - Running accuracy: {accuracy:.1%}")
            
            summary = await run_gsm8k_benchmark(
                problems=problems,
                config=config,
                max_concurrent=max_concurrent,
                progress_callback=progress_callback
            )
            
            execution_time = time.time() - start_time
            
            print(f"  ✓ Completed in {execution_time:.1f}s")
            print(f"  Accuracy: {summary.accuracy:.1%}")
            print(f"  Avg confidence: {summary.avg_confidence:.3f}")
            print(f"  Total tokens: {summary.total_tokens}")
            
            all_results["strategies"][strategy_name] = {
                "summary": summary.to_dict(),
                "execution_time": execution_time
            }
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            all_results["strategies"][strategy_name] = {
                "error": str(e)
            }
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"benchmark_{model_name}_{dataset}_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    successful_strategies = [
        name for name, result in all_results["strategies"].items()
        if "summary" in result
    ]
    
    if successful_strategies:
        print(f"{'Strategy':>20} {'Accuracy':>10} {'Confidence':>10} {'Tokens':>8} {'Time':>6}")
        print("-" * 60)
        
        for strategy_name in successful_strategies:
            result = all_results["strategies"][strategy_name]
            summary = result["summary"]
            exec_time = result["execution_time"]
            
            print(f"{strategy_name:>20} {summary['accuracy']:>9.1%} "
                  f"{summary['avg_confidence']:>9.3f} {summary['total_tokens']:>7} "
                  f"{exec_time:>5.1f}s")
        
        # Find best performing strategy
        best_strategy = max(successful_strategies, 
                           key=lambda s: all_results["strategies"][s]["summary"]["accuracy"])
        best_accuracy = all_results["strategies"][best_strategy]["summary"]["accuracy"]
        
        print("-" * 60)
        print(f"Best performing strategy: {best_strategy} ({best_accuracy:.1%} accuracy)")
    
    failed_strategies = [
        name for name, result in all_results["strategies"].items()
        if "error" in result
    ]
    
    if failed_strategies:
        print(f"\nFailed strategies: {', '.join(failed_strategies)}")
    
    return all_results


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(description="Run ThinkMesh benchmarks")
    
    parser.add_argument("--model", default="small_cpu", 
                       choices=list(get_model_configs().keys()),
                       help="Model configuration to use")
    
    parser.add_argument("--strategies", nargs="+", 
                       default=["self_consistency_small", "deepconf_small"],
                       choices=list(get_strategy_configs().keys()),
                       help="Strategy configurations to test")
    
    parser.add_argument("--dataset", default="sample", 
                       choices=["sample", "gsm8k"],
                       help="Dataset to use")
    
    parser.add_argument("--dataset-path", type=str,
                       help="Path to GSM8K dataset file (required for --dataset=gsm8k)")
    
    parser.add_argument("--num-problems", type=int, default=10,
                       help="Number of problems to test")
    
    parser.add_argument("--output-dir", default="benchmark_results",
                       help="Output directory for results")
    
    parser.add_argument("--max-concurrent", type=int, default=1,
                       help="Maximum concurrent executions")
    
    parser.add_argument("--quick", action="store_true",
                       help="Run quick benchmark with fewer problems")
    
    args = parser.parse_args()
    
    if args.quick:
        args.num_problems = min(args.num_problems, 3)
        args.strategies = args.strategies[:2]
        print("Quick mode: Limited to 3 problems and 2 strategies")
    
    # Run benchmarks
    try:
        results = asyncio.run(run_benchmark_suite(
            model_name=args.model,
            strategy_names=args.strategies,
            dataset=args.dataset,
            dataset_path=args.dataset_path,
            num_problems=args.num_problems,
            output_dir=args.output_dir,
            max_concurrent=args.max_concurrent
        ))
        
        print("\n✓ Benchmark suite completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠ Benchmark interrupted by user")
    except Exception as e:
        print(f"\n✗ Benchmark failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
