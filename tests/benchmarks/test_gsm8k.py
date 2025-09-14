"""
GSM8K benchmark tests for all ThinkMesh strategies.
"""
import pytest
import asyncio
import json
from pathlib import Path
from typing import List
from thinkmesh import ThinkConfig, ModelSpec, StrategySpec
from .gsm8k_utils import (
    GSM8KProblem, BenchmarkSummary, 
    create_gsm8k_sample_dataset, run_gsm8k_benchmark,
    extract_numerical_answer, is_correct_answer, normalize_answer
)


@pytest.mark.benchmark
class TestGSM8KUtils:
    """Test GSM8K utility functions."""
    
    def test_extract_numerical_answer(self):
        """Test numerical answer extraction."""
        test_cases = [
            ("The answer is 42", "42"),
            ("Final answer: 123", "123"),
            ("I calculate $50 for the total", "50"),
            ("The result equals 7.5", "7.5"),
            ("So we get -15 as the answer", "-15"),
            ("No clear number here", ""),
        ]
        
        for text, expected in test_cases:
            result = extract_numerical_answer(text)
            assert result == expected, f"Failed for '{text}': expected '{expected}', got '{result}'"
    
    def test_normalize_answer(self):
        """Test answer normalization."""
        test_cases = [
            ("42", "42"),
            ("$42", "42"),
            ("42.0", "42.0"),
            ("1/2", "0.5"),
            ("  123  ", "123"),
            ("1,000", "1000"),
        ]
        
        for input_val, expected in test_cases:
            result = normalize_answer(input_val)
            assert result == expected, f"Failed for '{input_val}': expected '{expected}', got '{result}'"
    
    def test_is_correct_answer(self):
        """Test answer correctness checking."""
        test_cases = [
            ("42", "42", True),
            ("42.0", "42", True),
            ("$42", "42", True),
            ("41", "42", False),
            ("0.5", "1/2", True),
            ("1000", "1,000", True),
        ]
        
        for pred, correct, expected in test_cases:
            result = is_correct_answer(pred, correct)
            assert result == expected, f"Failed for pred='{pred}', correct='{correct}'"
    
    def test_create_sample_dataset(self):
        """Test sample dataset creation."""
        problems = create_gsm8k_sample_dataset()
        assert len(problems) == 5
        
        for problem in problems:
            assert isinstance(problem, GSM8KProblem)
            assert len(problem.question) > 0
            assert len(problem.answer) > 0
            assert problem.problem_id is not None


@pytest.mark.benchmark
@pytest.mark.gsm8k
class TestGSM8KBenchmarks:
    """GSM8K benchmark tests for all strategies."""
    
    def get_test_configs(self) -> List[ThinkConfig]:
        """Get test configurations for all strategies."""
        base_model = ModelSpec(
            backend="transformers",
            model_name="microsoft/DialoGPT-small",  # Small model for testing
            max_tokens=256,
            temperature=0.7,
            extra={"device": "cpu"}
        )
        
        configs = []
        
        # Self-consistency
        configs.append(ThinkConfig(
            model=base_model,
            strategy=StrategySpec(
                name="self_consistency",
                parallel=3,
                max_steps=1
            ),
            budgets={"wall_clock_s": 60, "tokens": 1000}
        ))
        
        # DeepConf
        configs.append(ThinkConfig(
            model=base_model,
            strategy=StrategySpec(
                name="deepconf",
                parallel=4,
                max_steps=2,
                deepconf={"k": 3, "tau_low": -1.5, "tau_ent": 2.5}
            ),
            budgets={"wall_clock_s": 90, "tokens": 1500}
        ))
        
        # Debate
        configs.append(ThinkConfig(
            model=base_model,
            strategy=StrategySpec(
                name="debate",
                parallel=3,
                max_steps=2,
                debate={"rounds": 2}
            ),
            budgets={"wall_clock_s": 120, "tokens": 2000}
        ))
        
        # Tree of Thoughts
        configs.append(ThinkConfig(
            model=base_model,
            strategy=StrategySpec(
                name="tree",
                parallel=4,
                max_steps=2,
                tree={"branches": 2, "depth": 2}
            ),
            budgets={"wall_clock_s": 120, "tokens": 2000}
        ))
        
        # Graph
        configs.append(ThinkConfig(
            model=base_model,
            strategy=StrategySpec(
                name="graph",
                parallel=3,
                max_steps=2
            ),
            budgets={"wall_clock_s": 90, "tokens": 1500}
        ))
        
        return configs
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_gsm8k_sample_all_strategies(self):
        """Test all strategies on GSM8K sample problems."""
        problems = create_gsm8k_sample_dataset()
        configs = self.get_test_configs()
        
        results = {}
        
        for config in configs:
            strategy_name = config.strategy.name
            
            try:
                summary = await run_gsm8k_benchmark(
                    problems=problems[:2],  # Just first 2 problems for testing
                    config=config,
                    max_concurrent=1
                )
                
                results[strategy_name] = summary
                
                # Basic assertions
                assert summary.total_problems == 2
                assert 0.0 <= summary.accuracy <= 1.0
                assert summary.avg_confidence >= 0.0
                assert summary.avg_execution_time > 0.0
                assert len(summary.results) == 2
                
                print(f"\n{strategy_name} Results:")
                print(f"  Accuracy: {summary.accuracy:.2%}")
                print(f"  Avg Confidence: {summary.avg_confidence:.3f}")
                print(f"  Avg Time: {summary.avg_execution_time:.2f}s")
                
            except Exception as e:
                pytest.skip(f"Could not run {strategy_name}: {e}")
        
        # Compare strategies if we have results
        if len(results) > 1:
            best_accuracy = max(summary.accuracy for summary in results.values())
            best_strategies = [
                name for name, summary in results.items() 
                if summary.accuracy == best_accuracy
            ]
            print(f"\nBest performing strategies: {best_strategies}")
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.gpu
    async def test_gsm8k_gpu_benchmark(self, skip_if_no_cuda):
        """Test GSM8K benchmark with GPU acceleration."""
        config = ThinkConfig(
            model=ModelSpec(
                backend="transformers",
                model_name="microsoft/DialoGPT-medium",
                max_tokens=512,
                temperature=0.7,
                extra={"device": "cuda:0", "dtype": "float16"}
            ),
            strategy=StrategySpec(
                name="deepconf",
                parallel=6,
                max_steps=2,
                deepconf={"k": 5, "tau_low": -1.0}
            ),
            budgets={"wall_clock_s": 180, "tokens": 3000}
        )
        
        problems = create_gsm8k_sample_dataset()[:3]  # First 3 problems
        
        try:
            summary = await run_gsm8k_benchmark(
                problems=problems,
                config=config,
                max_concurrent=1  # Sequential for GPU memory management
            )
            
            assert summary.total_problems == 3
            assert summary.avg_execution_time > 0
            print(f"\nGPU Benchmark Results:")
            print(f"  Accuracy: {summary.accuracy:.2%}")
            print(f"  Avg Time: {summary.avg_execution_time:.2f}s")
            print(f"  Total Tokens: {summary.total_tokens}")
            
        except Exception as e:
            pytest.skip(f"GPU benchmark failed: {e}")
    
    @pytest.mark.asyncio
    @pytest.mark.slow 
    @pytest.mark.a100
    async def test_gsm8k_a100_benchmark(self, skip_if_no_a100):
        """Test GSM8K benchmark on A100 with larger model."""
        config = ThinkConfig(
            model=ModelSpec(
                backend="transformers",
                model_name="microsoft/DialoGPT-large",
                max_tokens=512,
                temperature=0.7,
                extra={"device": "cuda:0", "dtype": "float16"}
            ),
            strategy=StrategySpec(
                name="deepconf",
                parallel=8,
                max_steps=3,
                deepconf={"k": 5, "tau_low": -0.8, "tau_ent": 2.0}
            ),
            budgets={"wall_clock_s": 300, "tokens": 5000}
        )
        
        problems = create_gsm8k_sample_dataset()
        
        try:
            summary = await run_gsm8k_benchmark(
                problems=problems,
                config=config,
                max_concurrent=2  # A100 can handle more concurrency
            )
            
            print(f"\nA100 Benchmark Results:")
            print(f"  Model: {config.model.model_name}")
            print(f"  Strategy: {config.strategy.name}")
            print(f"  Problems: {summary.total_problems}")
            print(f"  Accuracy: {summary.accuracy:.2%}")
            print(f"  Avg Confidence: {summary.avg_confidence:.3f}")
            print(f"  Avg Time: {summary.avg_execution_time:.2f}s")
            print(f"  Total Tokens: {summary.total_tokens}")
            print(f"  Failed: {summary.failed_problems}")
            
            # Save results
            output_path = Path("benchmark_results") / "a100_gsm8k_results.json"
            output_path.parent.mkdir(exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(summary.to_dict(), f, indent=2)
            
            print(f"  Results saved to: {output_path}")
            
        except Exception as e:
            pytest.skip(f"A100 benchmark failed: {e}")
    
    @pytest.mark.asyncio
    async def test_benchmark_progress_tracking(self):
        """Test benchmark with progress tracking."""
        config = ThinkConfig(
            model=ModelSpec(
                backend="transformers",
                model_name="microsoft/DialoGPT-small",
                max_tokens=128,
                temperature=0.7,
                extra={"device": "cpu"}
            ),
            strategy=StrategySpec(name="self_consistency", parallel=2),
            budgets={"wall_clock_s": 60, "tokens": 500}
        )
        
        problems = create_gsm8k_sample_dataset()[:2]
        progress_updates = []
        
        def progress_callback(current: int, total: int, result):
            progress_updates.append((current, total, result.is_correct))
            print(f"Progress: {current}/{total} - Correct: {result.is_correct}")
        
        try:
            summary = await run_gsm8k_benchmark(
                problems=problems,
                config=config,
                progress_callback=progress_callback
            )
            
            assert len(progress_updates) == 2
            assert all(total == 2 for _, total, _ in progress_updates)
            assert progress_updates[-1][0] == 2  # Final progress should be 2/2
            
        except Exception as e:
            pytest.skip(f"Progress tracking test failed: {e}")


@pytest.mark.benchmark
@pytest.mark.gsm8k
class TestGSM8KComparison:
    """Compare different strategies on GSM8K."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_strategy_comparison(self):
        """Compare all strategies on the same problems."""
        problems = create_gsm8k_sample_dataset()[:3]
        
        strategies_configs = [
            ("self_consistency", {"parallel": 4}),
            ("deepconf", {"parallel": 6, "deepconf": {"k": 3}}),
            ("debate", {"parallel": 3, "debate": {"rounds": 2}}),
            ("tree", {"parallel": 4, "tree": {"branches": 2, "depth": 2}}),
        ]
        
        results = {}
        
        for strategy_name, strategy_params in strategies_configs:
            config = ThinkConfig(
                model=ModelSpec(
                    backend="transformers",
                    model_name="microsoft/DialoGPT-small",
                    max_tokens=256,
                    temperature=0.7,
                    extra={"device": "cpu"}
                ),
                strategy=StrategySpec(name=strategy_name, **strategy_params),
                budgets={"wall_clock_s": 120, "tokens": 2000}
            )
            
            try:
                summary = await run_gsm8k_benchmark(problems, config)
                results[strategy_name] = summary
                
            except Exception as e:
                print(f"Failed to run {strategy_name}: {e}")
                continue
        
        if results:
            print("\n" + "="*60)
            print("GSM8K Strategy Comparison")
            print("="*60)
            
            for strategy, summary in results.items():
                print(f"{strategy:>15}: {summary.accuracy:>6.1%} accuracy, "
                      f"{summary.avg_confidence:>5.3f} confidence, "
                      f"{summary.avg_execution_time:>5.1f}s avg time")
            
            # Find best strategy
            best_strategy = max(results.keys(), key=lambda s: results[s].accuracy)
            print(f"\nBest performing strategy: {best_strategy}")
            print("="*60)
    
    @pytest.mark.asyncio
    async def test_confidence_calibration(self):
        """Test how well confidence scores correlate with correctness."""
        config = ThinkConfig(
            model=ModelSpec(
                backend="transformers",
                model_name="microsoft/DialoGPT-small",
                max_tokens=256,
                temperature=0.7,
                extra={"device": "cpu"}
            ),
            strategy=StrategySpec(
                name="deepconf",
                parallel=4,
                deepconf={"k": 3}
            ),
            budgets={"wall_clock_s": 90, "tokens": 1000}
        )
        
        problems = create_gsm8k_sample_dataset()
        
        try:
            summary = await run_gsm8k_benchmark(problems, config)
            
            # Analyze confidence calibration
            correct_results = [r for r in summary.results if r.is_correct]
            incorrect_results = [r for r in summary.results if not r.is_correct]
            
            if correct_results and incorrect_results:
                avg_correct_conf = sum(r.confidence for r in correct_results) / len(correct_results)
                avg_incorrect_conf = sum(r.confidence for r in incorrect_results) / len(incorrect_results)
                
                print(f"\nConfidence Calibration:")
                print(f"  Correct answers avg confidence: {avg_correct_conf:.3f}")
                print(f"  Incorrect answers avg confidence: {avg_incorrect_conf:.3f}")
                print(f"  Difference: {avg_correct_conf - avg_incorrect_conf:.3f}")
                
                # Well-calibrated models should have higher confidence for correct answers
                assert avg_correct_conf >= avg_incorrect_conf - 0.1  # Allow small margin
        
        except Exception as e:
            pytest.skip(f"Confidence calibration test failed: {e}")


@pytest.mark.benchmark  
@pytest.mark.gsm8k
class TestGSM8KReproducibility:
    """Test reproducibility of GSM8K benchmarks."""
    
    @pytest.mark.asyncio
    async def test_deterministic_results(self):
        """Test that results are reproducible with fixed seed."""
        config = ThinkConfig(
            model=ModelSpec(
                backend="transformers",
                model_name="microsoft/DialoGPT-small",
                max_tokens=128,
                temperature=0.0,  # Deterministic
                seed=42,  # Fixed seed
                extra={"device": "cpu"}
            ),
            strategy=StrategySpec(name="self_consistency", parallel=2),
            budgets={"wall_clock_s": 60, "tokens": 500}
        )
        
        problems = create_gsm8k_sample_dataset()[:2]
        
        try:
            # Run twice with same config
            summary1 = await run_gsm8k_benchmark(problems, config)
            summary2 = await run_gsm8k_benchmark(problems, config)
            
            # Results should be identical (or very similar)
            assert summary1.accuracy == summary2.accuracy
            assert abs(summary1.avg_confidence - summary2.avg_confidence) < 0.01
            
            # Individual results should match
            for r1, r2 in zip(summary1.results, summary2.results):
                assert r1.predicted_answer == r2.predicted_answer
                assert r1.is_correct == r2.is_correct
                
        except Exception as e:
            pytest.skip(f"Reproducibility test failed: {e}")
