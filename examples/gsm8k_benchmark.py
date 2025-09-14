#!/usr/bin/env python3
"""
Example: Running GSM8K benchmarks with different strategies.
"""
import asyncio
import time
from pathlib import Path
from thinkmesh import ThinkConfig, ModelSpec, StrategySpec
from tests.benchmarks.gsm8k_utils import (
    create_gsm8k_sample_dataset,
    run_gsm8k_benchmark,
    save_benchmark_results
)


async def run_quick_gsm8k_comparison():
    """Compare different strategies on a small GSM8K sample."""
    print("=== Quick GSM8K Strategy Comparison ===")
    
    # Use sample dataset for quick testing
    problems = create_gsm8k_sample_dataset()[:3]  # Just 3 problems
    print(f"Testing on {len(problems)} GSM8K problems")
    
    # Define strategies to compare
    strategies = [
        ("Self-Consistency", StrategySpec(name="self_consistency", parallel=4)),
        ("DeepConf", StrategySpec(name="deepconf", parallel=6, deepconf={"k": 3})),
        ("Debate", StrategySpec(name="debate", parallel=3, debate={"rounds": 2})),
        ("Tree", StrategySpec(name="tree", parallel=4, tree={"branches": 2, "depth": 2})),
    ]
    
    results = {}
    
    for strategy_name, strategy_spec in strategies:
        print(f"\n--- Testing {strategy_name} ---")
        
        config = ThinkConfig(
            model=ModelSpec(
                backend="transformers",
                model_name="microsoft/DialoGPT-small",  # Small model for quick testing
                max_tokens=256,
                temperature=0.7,
                extra={"device": "cpu"}  # Use CPU for compatibility
            ),
            strategy=strategy_spec,
            budgets={"wall_clock_s": 180, "tokens": 2000}
        )
        
        try:
            start_time = time.time()
            
            def progress_callback(current, total, result):
                print(f"  Progress: {current}/{total} - "
                      f"Problem: {result.problem_id} - "
                      f"Correct: {'✓' if result.is_correct else '✗'}")
            
            summary = await run_gsm8k_benchmark(
                problems=problems,
                config=config,
                progress_callback=progress_callback
            )
            
            execution_time = time.time() - start_time
            results[strategy_name] = {
                "summary": summary,
                "execution_time": execution_time
            }
            
            print(f"  Results: {summary.accuracy:.1%} accuracy, "
                  f"{summary.avg_confidence:.3f} avg confidence, "
                  f"{execution_time:.1f}s total time")
            
        except Exception as e:
            print(f"  Error: {e}")
            results[strategy_name] = {"error": str(e)}
    
    # Print comparison summary
    print("\n=== Comparison Summary ===")
    print(f"{'Strategy':>15} {'Accuracy':>10} {'Confidence':>10} {'Time':>8}")
    print("-" * 50)
    
    for strategy_name, result in results.items():
        if "summary" in result:
            summary = result["summary"]
            exec_time = result["execution_time"]
            print(f"{strategy_name:>15} {summary.accuracy:>9.1%} "
                  f"{summary.avg_confidence:>9.3f} {exec_time:>7.1f}s")
        else:
            print(f"{strategy_name:>15} {'ERROR':>10} {'-':>10} {'-':>8}")
    
    return results


async def run_gpu_gsm8k_benchmark():
    """Run GSM8K benchmark with GPU acceleration."""
    print("\n=== GPU Accelerated GSM8K Benchmark ===")
    
    import torch
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU benchmark")
        return
    
    problems = create_gsm8k_sample_dataset()[:5]  # 5 problems for GPU test
    
    config = ThinkConfig(
        model=ModelSpec(
            backend="transformers",
            model_name="microsoft/DialoGPT-medium",
            max_tokens=512,
            temperature=0.7,
            extra={
                "device": "cuda:0",
                "dtype": "float16",
                "batch_size": 8
            }
        ),
        strategy=StrategySpec(
            name="deepconf",
            parallel=8,
            max_steps=2,
            deepconf={
                "k": 5,
                "tau_low": -1.0,
                "tau_ent": 2.0,
                "realloc_top_p": 0.4
            }
        ),
        budgets={"wall_clock_s": 300, "tokens": 5000}
    )
    
    print(f"Running DeepConf on {len(problems)} problems with GPU acceleration...")
    
    try:
        start_time = time.time()
        
        def progress_callback(current, total, result):
            print(f"GPU Progress: {current}/{total} - "
                  f"Correct: {'✓' if result.is_correct else '✗'} - "
                  f"Confidence: {result.confidence:.3f}")
        
        summary = await run_gsm8k_benchmark(
            problems=problems,
            config=config,
            max_concurrent=2,  # Allow some concurrency for GPU
            progress_callback=progress_callback
        )
        
        total_time = time.time() - start_time
        
        print(f"\nGPU Benchmark Results:")
        print(f"  Model: {config.model.model_name}")
        print(f"  Strategy: {config.strategy.name}")
        print(f"  Problems: {len(problems)}")
        print(f"  Accuracy: {summary.accuracy:.1%}")
        print(f"  Average confidence: {summary.avg_confidence:.3f}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Time per problem: {total_time / len(problems):.1f}s")
        print(f"  Total tokens: {summary.total_tokens}")
        print(f"  Tokens per second: {summary.total_tokens / total_time:.0f}")
        
        # Show individual results
        print(f"\nIndividual Results:")
        for result in summary.results:
            status = "✓" if result.is_correct else "✗"
            print(f"  {status} {result.predicted_answer} (correct: {result.correct_answer}) "
                  f"conf: {result.confidence:.3f}")
        
        return summary
        
    except Exception as e:
        print(f"GPU benchmark failed: {e}")
        return None


async def run_a100_performance_test():
    """Run performance test specifically designed for A100 GPU."""
    print("\n=== A100 Performance Test ===")
    
    import torch
    if not torch.cuda.is_available():
        print("CUDA not available, skipping A100 test")
        return
    
    # Check if we have an A100 (>70GB memory indicates A100 80GB)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if gpu_memory < 70:
        print(f"GPU has {gpu_memory:.1f}GB memory, A100 test requires >70GB")
        return
    
    print(f"Detected A100 with {gpu_memory:.1f}GB memory")
    
    problems = create_gsm8k_sample_dataset()  # All 5 sample problems
    
    # High-performance A100 configuration
    config = ThinkConfig(
        model=ModelSpec(
            backend="transformers",
            model_name="microsoft/DialoGPT-large",  # Larger model
            max_tokens=512,
            temperature=0.7,
            extra={
                "device": "cuda:0",
                "dtype": "float16",
                "batch_size": 16  # Large batch for A100
            }
        ),
        strategy=StrategySpec(
            name="deepconf",
            parallel=16,  # High parallelism
            max_steps=2,
            deepconf={
                "k": 5,
                "tau_low": -0.8,
                "tau_ent": 1.8,
                "realloc_top_p": 0.5
            }
        ),
        budgets={"wall_clock_s": 600, "tokens": 10000}
    )
    
    try:
        print(f"Running A100 performance test on {len(problems)} problems...")
        print(f"Configuration: {config.strategy.parallel} parallel, batch_size={config.model.extra['batch_size']}")
        
        # Monitor GPU memory
        torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.memory_allocated() / (1024**3)
        
        start_time = time.time()
        
        def progress_callback(current, total, result):
            current_memory = torch.cuda.memory_allocated() / (1024**3)
            print(f"A100 Progress: {current}/{total} - "
                  f"Correct: {'✓' if result.is_correct else '✗'} - "
                  f"GPU Memory: {current_memory:.1f}GB")
        
        summary = await run_gsm8k_benchmark(
            problems=problems,
            config=config,
            max_concurrent=4,  # A100 can handle more concurrency
            progress_callback=progress_callback
        )
        
        total_time = time.time() - start_time
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
        
        print(f"\nA100 Performance Results:")
        print(f"  Model: {config.model.model_name}")
        print(f"  Parallel processes: {config.strategy.parallel}")
        print(f"  Batch size: {config.model.extra['batch_size']}")
        print(f"  Total problems: {len(problems)}")
        print(f"  Accuracy: {summary.accuracy:.1%}")
        print(f"  Average confidence: {summary.avg_confidence:.3f}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Throughput: {len(problems) / total_time:.2f} problems/sec")
        print(f"  Token throughput: {summary.total_tokens / total_time:.0f} tokens/sec")
        print(f"  Peak GPU memory: {peak_memory:.1f}GB")
        print(f"  Memory efficiency: {summary.total_tokens / peak_memory:.0f} tokens/GB")
        
        return summary
        
    except torch.cuda.OutOfMemoryError:
        print("A100 test exceeded GPU memory capacity")
        return None
    except Exception as e:
        print(f"A100 performance test failed: {e}")
        return None


async def save_benchmark_comparison():
    """Run comprehensive benchmark and save results."""
    print("\n=== Saving Benchmark Results ===")
    
    # Run quick comparison
    results = await run_quick_gsm8k_comparison()
    
    # Save results for later analysis
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)
    
    for strategy_name, result in results.items():
        if "summary" in result:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = output_dir / f"gsm8k_{strategy_name.lower().replace(' ', '_')}_{timestamp}.json"
            
            try:
                save_benchmark_results(result["summary"], filename)
                print(f"Saved {strategy_name} results to {filename}")
            except Exception as e:
                print(f"Error saving {strategy_name} results: {e}")
    
    print(f"\nResults saved to {output_dir}")


async def main():
    """Run all GSM8K benchmark examples."""
    print("ThinkMesh GSM8K Benchmark Examples")
    print("=" * 50)
    
    # Quick strategy comparison
    await run_quick_gsm8k_comparison()
    
    # GPU benchmark if available
    await run_gpu_gsm8k_benchmark()
    
    # A100 performance test if available
    await run_a100_performance_test()
    
    # Save results
    await save_benchmark_comparison()
    
    print("\n" + "=" * 50)
    print("GSM8K benchmark examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
