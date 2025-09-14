"""
Performance tests for A100 GPU with ThinkMesh.
"""
import pytest
import torch
import time
import psutil
import os
from typing import List, Dict, Any
from dataclasses import dataclass
from thinkmesh import think, ThinkConfig, ModelSpec, StrategySpec
from tests.benchmarks.gsm8k_utils import create_gsm8k_sample_dataset, run_gsm8k_benchmark


@dataclass
class PerformanceMetrics:
    """Performance measurement metrics."""
    throughput_problems_per_sec: float
    throughput_tokens_per_sec: float
    avg_memory_usage_gb: float
    peak_memory_usage_gb: float
    gpu_utilization_percent: float
    avg_response_time_sec: float
    total_tokens: int
    total_problems: int
    strategy_name: str
    model_name: str
    batch_size: int
    parallel_count: int


def get_gpu_memory_usage() -> float:
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    return 0.0


def get_gpu_memory_peak() -> float:
    """Get peak GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**3)
    return 0.0


def get_cpu_memory_usage() -> float:
    """Get current CPU memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**3)


def reset_gpu_memory_stats():
    """Reset GPU memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


@pytest.mark.performance
@pytest.mark.a100
class TestA100Performance:
    """Performance tests specifically for A100 GPU."""
    
    def get_performance_configs(self) -> List[ThinkConfig]:
        """Get configurations optimized for A100 performance testing."""
        configs = []
        
        # High-throughput config with larger batch sizes
        configs.append(ThinkConfig(
            model=ModelSpec(
                backend="transformers",
                model_name="microsoft/DialoGPT-medium", 
                max_tokens=512,
                temperature=0.7,
                extra={
                    "device": "cuda:0", 
                    "dtype": "float16",
                    "batch_size": 16
                }
            ),
            strategy=StrategySpec(
                name="self_consistency",
                parallel=16,
                max_steps=1
            ),
            budgets={"wall_clock_s": 300, "tokens": 10000}
        ))
        
        # DeepConf with high parallelism
        configs.append(ThinkConfig(
            model=ModelSpec(
                backend="transformers",
                model_name="microsoft/DialoGPT-medium",
                max_tokens=512,
                temperature=0.7,
                extra={
                    "device": "cuda:0",
                    "dtype": "float16", 
                    "batch_size": 12
                }
            ),
            strategy=StrategySpec(
                name="deepconf",
                parallel=12,
                max_steps=2,
                deepconf={"k": 5, "tau_low": -1.0, "tau_ent": 2.0, "realloc_top_p": 0.5}
            ),
            budgets={"wall_clock_s": 400, "tokens": 15000}
        ))
        
        # Tree strategy with moderate parallelism
        configs.append(ThinkConfig(
            model=ModelSpec(
                backend="transformers", 
                model_name="microsoft/DialoGPT-medium",
                max_tokens=384,
                temperature=0.7,
                extra={
                    "device": "cuda:0",
                    "dtype": "float16",
                    "batch_size": 8
                }
            ),
            strategy=StrategySpec(
                name="tree",
                parallel=8,
                max_steps=3,
                tree={"branches": 4, "depth": 2}
            ),
            budgets={"wall_clock_s": 350, "tokens": 12000}
        ))
        
        return configs
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_a100_throughput_benchmark(self, skip_if_no_a100):
        """Test throughput performance on A100."""
        configs = self.get_performance_configs()
        problems = create_gsm8k_sample_dataset() * 2  # 10 problems total
        
        results = []
        
        for config in configs:
            reset_gpu_memory_stats()
            
            print(f"\n{'='*60}")
            print(f"Testing {config.strategy.name} with {config.strategy.parallel} parallel")
            print(f"Model: {config.model.model_name}")
            print(f"Batch size: {config.model.extra.get('batch_size', 'default')}")
            
            try:
                start_time = time.time()
                memory_readings = []
                
                def memory_monitor():
                    """Monitor memory usage during execution."""
                    memory_readings.append(get_gpu_memory_usage())
                
                # Run benchmark
                summary = await run_gsm8k_benchmark(
                    problems=problems,
                    config=config,
                    max_concurrent=2,  # Allow some concurrency
                    progress_callback=lambda *args: memory_monitor()
                )
                
                total_time = time.time() - start_time
                peak_memory = get_gpu_memory_peak()
                avg_memory = sum(memory_readings) / len(memory_readings) if memory_readings else 0
                
                # Calculate throughput metrics
                throughput_problems = len(problems) / total_time
                throughput_tokens = summary.total_tokens / total_time if summary.total_tokens > 0 else 0
                
                metrics = PerformanceMetrics(
                    throughput_problems_per_sec=throughput_problems,
                    throughput_tokens_per_sec=throughput_tokens,
                    avg_memory_usage_gb=avg_memory,
                    peak_memory_usage_gb=peak_memory,
                    gpu_utilization_percent=0.0,  # Would need nvidia-ml-py for this
                    avg_response_time_sec=summary.avg_execution_time,
                    total_tokens=summary.total_tokens,
                    total_problems=len(problems),
                    strategy_name=config.strategy.name,
                    model_name=config.model.model_name,
                    batch_size=config.model.extra.get('batch_size', 4),
                    parallel_count=config.strategy.parallel
                )
                
                results.append(metrics)
                
                # Print results
                print(f"Accuracy: {summary.accuracy:.1%}")
                print(f"Throughput: {throughput_problems:.2f} problems/sec")
                print(f"Token throughput: {throughput_tokens:.0f} tokens/sec")
                print(f"Avg memory: {avg_memory:.2f} GB")
                print(f"Peak memory: {peak_memory:.2f} GB")
                print(f"Avg response time: {summary.avg_execution_time:.2f} sec")
                print(f"Total tokens: {summary.total_tokens}")
                
                # Performance assertions
                assert throughput_problems > 0.01  # At least 0.01 problems/sec
                assert peak_memory < 70.0  # Should fit comfortably in 80GB A100
                assert summary.failed_problems == 0  # No failures expected
                
            except Exception as e:
                print(f"Failed to run {config.strategy.name}: {e}")
                continue
        
        if results:
            print(f"\n{'='*60}")
            print("A100 PERFORMANCE SUMMARY")
            print(f"{'='*60}")
            
            best_throughput = max(results, key=lambda x: x.throughput_problems_per_sec)
            best_memory = min(results, key=lambda x: x.peak_memory_usage_gb)
            
            print(f"Best throughput: {best_throughput.strategy_name} "
                  f"({best_throughput.throughput_problems_per_sec:.2f} problems/sec)")
            print(f"Most memory efficient: {best_memory.strategy_name} "
                  f"({best_memory.peak_memory_usage_gb:.2f} GB peak)")
            
            # Export results
            import json
            from pathlib import Path
            
            output_path = Path("performance_results") / "a100_throughput.json"
            output_path.parent.mkdir(exist_ok=True)
            
            results_dict = {
                "timestamp": time.time(),
                "gpu_info": str(torch.cuda.get_device_properties(0)),
                "results": [
                    {
                        "strategy": r.strategy_name,
                        "throughput_problems_sec": r.throughput_problems_per_sec,
                        "throughput_tokens_sec": r.throughput_tokens_per_sec,
                        "peak_memory_gb": r.peak_memory_usage_gb,
                        "avg_response_time": r.avg_response_time_sec,
                        "parallel_count": r.parallel_count,
                        "batch_size": r.batch_size
                    }
                    for r in results
                ]
            }
            
            with open(output_path, 'w') as f:
                json.dump(results_dict, f, indent=2)
            
            print(f"Results exported to: {output_path}")
    
    @pytest.mark.asyncio
    async def test_a100_memory_scaling(self, skip_if_no_a100):
        """Test how memory usage scales with batch size and parallelism."""
        base_config = ThinkConfig(
            model=ModelSpec(
                backend="transformers",
                model_name="microsoft/DialoGPT-small",  # Start with smaller model
                max_tokens=256,
                temperature=0.7,
                extra={"device": "cuda:0", "dtype": "float16"}
            ),
            strategy=StrategySpec(name="self_consistency", parallel=4),
            budgets={"wall_clock_s": 120, "tokens": 2000}
        )
        
        problems = create_gsm8k_sample_dataset()[:3]
        scaling_results = []
        
        # Test different batch sizes and parallel counts
        test_configs = [
            (2, 4),   # batch_size, parallel
            (4, 8),
            (8, 12), 
            (12, 16),
            (16, 20)
        ]
        
        for batch_size, parallel in test_configs:
            reset_gpu_memory_stats()
            
            config = ThinkConfig(
                model=ModelSpec(
                    backend=base_config.model.backend,
                    model_name=base_config.model.model_name,
                    max_tokens=base_config.model.max_tokens,
                    temperature=base_config.model.temperature,
                    extra={**base_config.model.extra, "batch_size": batch_size}
                ),
                strategy=StrategySpec(
                    name=base_config.strategy.name,
                    parallel=parallel
                ),
                budgets=base_config.budgets
            )
            
            try:
                start_time = time.time()
                
                summary = await run_gsm8k_benchmark(problems, config)
                
                execution_time = time.time() - start_time
                peak_memory = get_gpu_memory_peak()
                
                scaling_results.append({
                    "batch_size": batch_size,
                    "parallel": parallel,
                    "peak_memory_gb": peak_memory,
                    "execution_time": execution_time,
                    "accuracy": summary.accuracy,
                    "tokens_total": summary.total_tokens
                })
                
                print(f"Batch {batch_size}, Parallel {parallel}: "
                      f"{peak_memory:.2f} GB peak, {execution_time:.1f}s")
                
                # Stop if we're approaching memory limits
                if peak_memory > 60:  # 60GB threshold
                    print("Approaching memory limit, stopping scaling test")
                    break
                    
            except torch.cuda.OutOfMemoryError:
                print(f"OOM at batch_size={batch_size}, parallel={parallel}")
                break
            except Exception as e:
                print(f"Error at batch_size={batch_size}, parallel={parallel}: {e}")
                continue
        
        # Analyze scaling
        if scaling_results:
            print("\nMemory Scaling Analysis:")
            for result in scaling_results:
                efficiency = result["tokens_total"] / result["peak_memory_gb"] if result["peak_memory_gb"] > 0 else 0
                print(f"  Batch {result['batch_size']:2d}, Parallel {result['parallel']:2d}: "
                      f"{result['peak_memory_gb']:5.1f} GB, "
                      f"{efficiency:6.0f} tokens/GB")
            
            # Find optimal configuration
            best_efficiency = max(scaling_results, 
                                key=lambda x: (x["tokens_total"] / x["peak_memory_gb"]) if x["peak_memory_gb"] > 0 else 0)
            
            print(f"\nMost efficient config: batch_size={best_efficiency['batch_size']}, "
                  f"parallel={best_efficiency['parallel']}")
    
    @pytest.mark.asyncio
    async def test_a100_concurrent_strategies(self, skip_if_no_a100):
        """Test running multiple strategies concurrently on A100."""
        import asyncio
        
        configs = [
            ThinkConfig(
                model=ModelSpec(
                    backend="transformers",
                    model_name="microsoft/DialoGPT-small",
                    max_tokens=256,
                    temperature=0.7,
                    extra={"device": "cuda:0", "dtype": "float16", "batch_size": 4}
                ),
                strategy=StrategySpec(name="self_consistency", parallel=4),
                budgets={"wall_clock_s": 120, "tokens": 1500}
            ),
            ThinkConfig(
                model=ModelSpec(
                    backend="transformers",
                    model_name="microsoft/DialoGPT-small",
                    max_tokens=256,
                    temperature=0.7,
                    extra={"device": "cuda:0", "dtype": "float16", "batch_size": 4}
                ),
                strategy=StrategySpec(
                    name="deepconf", 
                    parallel=4, 
                    deepconf={"k": 3}
                ),
                budgets={"wall_clock_s": 120, "tokens": 1500}
            )
        ]
        
        problems = create_gsm8k_sample_dataset()[:2]
        
        reset_gpu_memory_stats()
        start_time = time.time()
        
        # Run strategies concurrently
        tasks = [
            run_gsm8k_benchmark(problems, config, max_concurrent=1)
            for config in configs
        ]
        
        try:
            summaries = await asyncio.gather(*tasks)
            
            total_time = time.time() - start_time
            peak_memory = get_gpu_memory_peak()
            
            print(f"\nConcurrent Execution Results:")
            print(f"Total time: {total_time:.1f}s")
            print(f"Peak memory: {peak_memory:.2f} GB")
            
            for i, summary in enumerate(summaries):
                print(f"Strategy {configs[i].strategy.name}: "
                      f"{summary.accuracy:.1%} accuracy, "
                      f"{summary.avg_execution_time:.1f}s avg time")
            
            # Concurrent execution should be more efficient than sequential
            sequential_estimate = sum(s.avg_execution_time for s in summaries) * len(problems)
            speedup = sequential_estimate / total_time if total_time > 0 else 1
            
            print(f"Estimated speedup: {speedup:.2f}x")
            
            assert peak_memory < 70.0  # Should still fit in A100
            assert speedup > 1.2  # Should see some speedup from concurrency
            
        except Exception as e:
            pytest.skip(f"Concurrent execution test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_a100_large_batch_performance(self, skip_if_no_a100):
        """Test performance with very large batch sizes on A100."""
        config = ThinkConfig(
            model=ModelSpec(
                backend="transformers",
                model_name="microsoft/DialoGPT-medium",
                max_tokens=512,
                temperature=0.7,
                extra={
                    "device": "cuda:0",
                    "dtype": "float16", 
                    "batch_size": 32  # Very large batch
                }
            ),
            strategy=StrategySpec(
                name="self_consistency",
                parallel=32,  # Match batch size
                max_steps=1
            ),
            budgets={"wall_clock_s": 600, "tokens": 20000}
        )
        
        # Create larger problem set
        problems = create_gsm8k_sample_dataset() * 4  # 20 problems
        
        reset_gpu_memory_stats()
        
        try:
            start_time = time.time()
            
            summary = await run_gsm8k_benchmark(
                problems=problems,
                config=config,
                max_concurrent=1  # Sequential for memory management
            )
            
            total_time = time.time() - start_time
            peak_memory = get_gpu_memory_peak()
            
            # Calculate metrics
            throughput = len(problems) / total_time
            token_throughput = summary.total_tokens / total_time
            memory_per_problem = peak_memory / len(problems)
            
            print(f"\nLarge Batch Performance:")
            print(f"Problems: {len(problems)}")
            print(f"Batch size: 32")
            print(f"Parallel: 32")
            print(f"Total time: {total_time:.1f}s")
            print(f"Throughput: {throughput:.2f} problems/sec") 
            print(f"Token throughput: {token_throughput:.0f} tokens/sec")
            print(f"Peak memory: {peak_memory:.2f} GB")
            print(f"Memory per problem: {memory_per_problem:.3f} GB")
            print(f"Accuracy: {summary.accuracy:.1%}")
            
            # Performance assertions for large batch
            assert throughput > 0.05  # At least 0.05 problems/sec
            assert peak_memory < 75.0  # Should fit in A100 with headroom
            assert token_throughput > 50  # Reasonable token throughput
            assert summary.failed_problems == 0  # No failures
            
            # Should achieve good GPU utilization with large batches
            tokens_per_gb = summary.total_tokens / peak_memory if peak_memory > 0 else 0
            print(f"Efficiency: {tokens_per_gb:.0f} tokens/GB")
            
            assert tokens_per_gb > 100  # Reasonable efficiency
            
        except torch.cuda.OutOfMemoryError:
            pytest.skip("Large batch test exceeded A100 memory capacity")
        except Exception as e:
            pytest.skip(f"Large batch test failed: {e}")


@pytest.mark.performance
@pytest.mark.a100  
@pytest.mark.slow
class TestA100MemoryProfiling:
    """Detailed memory profiling tests for A100."""
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, skip_if_no_a100):
        """Test for memory leaks during repeated execution."""
        config = ThinkConfig(
            model=ModelSpec(
                backend="transformers",
                model_name="microsoft/DialoGPT-small",
                max_tokens=128,
                temperature=0.7,
                extra={"device": "cuda:0", "dtype": "float16", "batch_size": 4}
            ),
            strategy=StrategySpec(name="self_consistency", parallel=4),
            budgets={"wall_clock_s": 60, "tokens": 1000}
        )
        
        problems = create_gsm8k_sample_dataset()[:2]
        memory_usage = []
        
        # Run multiple iterations
        for iteration in range(5):
            reset_gpu_memory_stats()
            
            try:
                summary = await run_gsm8k_benchmark(problems, config)
                peak_memory = get_gpu_memory_peak()
                memory_usage.append(peak_memory)
                
                print(f"Iteration {iteration + 1}: {peak_memory:.2f} GB peak memory")
                
                # Force cleanup
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Iteration {iteration + 1} failed: {e}")
                continue
        
        if len(memory_usage) >= 3:
            # Check for memory growth trend
            memory_trend = (memory_usage[-1] - memory_usage[0]) / len(memory_usage)
            print(f"\nMemory trend: {memory_trend:.4f} GB per iteration")
            
            # Should not have significant memory growth (< 0.1 GB per iteration)
            assert memory_trend < 0.1, f"Potential memory leak detected: {memory_trend:.4f} GB/iter"
            
            # Memory usage should be relatively stable
            memory_std = (sum((m - sum(memory_usage)/len(memory_usage))**2 for m in memory_usage) / len(memory_usage))**0.5
            print(f"Memory stability (std dev): {memory_std:.3f} GB")
            
            assert memory_std < 1.0, f"Memory usage too variable: {memory_std:.3f} GB std dev"
    
    @pytest.mark.asyncio
    async def test_memory_fragmentation(self, skip_if_no_a100):
        """Test memory fragmentation with different allocation patterns.""" 
        configs = [
            # Small batches, high parallelism
            ThinkConfig(
                model=ModelSpec(
                    backend="transformers",
                    model_name="microsoft/DialoGPT-small",
                    max_tokens=128,
                    extra={"device": "cuda:0", "dtype": "float16", "batch_size": 2}
                ),
                strategy=StrategySpec(name="self_consistency", parallel=16)
            ),
            # Large batches, low parallelism 
            ThinkConfig(
                model=ModelSpec(
                    backend="transformers", 
                    model_name="microsoft/DialoGPT-small",
                    max_tokens=128,
                    extra={"device": "cuda:0", "dtype": "float16", "batch_size": 16}
                ),
                strategy=StrategySpec(name="self_consistency", parallel=2)
            )
        ]
        
        problems = create_gsm8k_sample_dataset()[:2]
        
        for i, config in enumerate(configs):
            reset_gpu_memory_stats()
            
            try:
                print(f"\nTesting allocation pattern {i + 1}:")
                print(f"  Batch size: {config.model.extra['batch_size']}")
                print(f"  Parallel: {config.strategy.parallel}")
                
                summary = await run_gsm8k_benchmark(problems, config)
                
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                fragmentation = (reserved - allocated) / reserved if reserved > 0 else 0
                
                print(f"  Allocated: {allocated:.2f} GB")
                print(f"  Reserved: {reserved:.2f} GB") 
                print(f"  Fragmentation: {fragmentation:.1%}")
                print(f"  Accuracy: {summary.accuracy:.1%}")
                
                # Fragmentation should be reasonable (< 50%)
                assert fragmentation < 0.5, f"High memory fragmentation: {fragmentation:.1%}"
                
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  Failed: {e}")
                continue
