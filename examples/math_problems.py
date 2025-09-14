#!/usr/bin/env python3
"""
Examples of using ThinkMesh for mathematical reasoning problems.
"""
import asyncio
from thinkmesh import think, ThinkConfig, ModelSpec, StrategySpec


def basic_arithmetic_example():
    """Example: Basic arithmetic with self-consistency."""
    print("=== Basic Arithmetic Example ===")
    
    config = ThinkConfig(
        model=ModelSpec(
            backend="transformers",
            model_name="microsoft/DialoGPT-small",
            max_tokens=128,
            temperature=0.3,  # Lower temperature for mathematical accuracy
            extra={"device": "cpu"}
        ),
        strategy=StrategySpec(
            name="self_consistency",
            parallel=6,
            max_steps=1
        ),
        verifier={
            "type": "regex", 
            "pattern": r"(?:answer|result|equals?)\s*:?\s*(\d+)"
        }
    )
    
    problems = [
        "What is 127 * 43?",
        "Calculate 2^8", 
        "Find the square root of 144",
        "What is 15% of 240?"
    ]
    
    for problem in problems:
        print(f"\nProblem: {problem}")
        answer = think(problem, config)
        print(f"Answer: {answer.content}")
        print(f"Confidence: {answer.confidence:.3f}")


def complex_proof_example():
    """Example: Mathematical proof with DeepConf strategy."""
    print("\n=== Mathematical Proof Example ===")
    
    config = ThinkConfig(
        model=ModelSpec(
            backend="transformers",
            model_name="microsoft/DialoGPT-medium",
            max_tokens=512,
            temperature=0.7,
            extra={"device": "cpu"}
        ),
        strategy=StrategySpec(
            name="deepconf",
            parallel=8,
            max_steps=2,
            deepconf={
                "k": 5,
                "tau_low": -1.5,
                "tau_ent": 2.5,
                "realloc_top_p": 0.3
            }
        )
    )
    
    proof_problems = [
        "Prove that the sum of the first n positive integers is n(n+1)/2.",
        "Show that the square root of 2 is irrational.",
        "Prove that there are infinitely many prime numbers."
    ]
    
    for problem in proof_problems:
        print(f"\nProblem: {problem}")
        try:
            answer = think(problem, config)
            print(f"Proof: {answer.content[:200]}...")
            print(f"Confidence: {answer.confidence:.3f}")
        except Exception as e:
            print(f"Error: {e}")


def word_problem_debate_example():
    """Example: Word problems using debate strategy."""
    print("\n=== Word Problem Debate Example ===")
    
    config = ThinkConfig(
        model=ModelSpec(
            backend="transformers",
            model_name="microsoft/DialoGPT-medium", 
            max_tokens=384,
            temperature=0.8,
            extra={"device": "cpu"}
        ),
        strategy=StrategySpec(
            name="debate",
            parallel=4,
            max_steps=2,
            debate={"rounds": 2}
        )
    )
    
    word_problems = [
        """A farmer has chickens and cows. Together they have 30 heads and 74 legs. 
        How many chickens and how many cows does the farmer have?""",
        
        """A train travels from city A to city B at 60 mph and returns at 40 mph. 
        If the total trip takes 5 hours, what is the distance between the cities?""",
        
        """Alice has twice as many apples as Bob. Bob has 3 more apples than Charlie. 
        Together they have 33 apples. How many apples does each person have?"""
    ]
    
    for problem in word_problems:
        print(f"\nProblem: {problem}")
        try:
            answer = think(problem, config)
            print(f"Solution: {answer.content}")
            print(f"Confidence: {answer.confidence:.3f}")
        except Exception as e:
            print(f"Error: {e}")


def tree_search_optimization_example():
    """Example: Optimization problem with Tree of Thoughts."""
    print("\n=== Optimization with Tree of Thoughts ===")
    
    config = ThinkConfig(
        model=ModelSpec(
            backend="transformers",
            model_name="microsoft/DialoGPT-medium",
            max_tokens=512,
            temperature=0.7,
            extra={"device": "cpu"}
        ),
        strategy=StrategySpec(
            name="tree",
            parallel=8,
            max_steps=3,
            tree={"branches": 3, "depth": 2}
        )
    )
    
    optimization_problems = [
        """Find the maximum value of f(x) = -x² + 6x - 5 for x in the real numbers.""",
        
        """A company wants to maximize profit. They can produce products A and B.
        Product A gives $3 profit per unit and requires 2 hours of labor.
        Product B gives $5 profit per unit and requires 4 hours of labor.
        They have 40 hours of labor available. How many of each product should they make?""",
        
        """What is the shortest path connecting points (0,0), (3,4), and (7,1) 
        if you must visit all three points?"""
    ]
    
    for problem in optimization_problems:
        print(f"\nProblem: {problem}")
        try:
            answer = think(problem, config)
            print(f"Solution approach: {answer.content}")
            print(f"Confidence: {answer.confidence:.3f}")
        except Exception as e:
            print(f"Error: {e}")


async def gpu_accelerated_example():
    """Example: GPU-accelerated mathematical reasoning."""
    print("\n=== GPU Accelerated Example ===")
    
    import torch
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU example")
        return
    
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
            parallel=12,  # Higher parallelism with GPU
            max_steps=2,
            deepconf={
                "k": 5,
                "tau_low": -1.0,
                "tau_ent": 2.0,
                "realloc_top_p": 0.4
            }
        )
    )
    
    complex_problems = [
        "Solve the differential equation dy/dx = xy with initial condition y(0) = 1",
        "Find all solutions to x³ - 6x² + 11x - 6 = 0",
        "Calculate the definite integral of x²e^x from 0 to 1"
    ]
    
    for problem in complex_problems:
        print(f"\nProblem: {problem}")
        try:
            answer = think(problem, config)
            print(f"Solution: {answer.content}")
            print(f"Confidence: {answer.confidence:.3f}")
            print(f"Tokens used: {answer.meta.get('total_tokens', 'unknown')}")
            print(f"Time: {answer.meta.get('elapsed_s', 0):.2f}s")
        except Exception as e:
            print(f"Error: {e}")


def custom_verifier_example():
    """Example: Using custom verifiers for answer validation."""
    print("\n=== Custom Verifier Example ===")
    
    config = ThinkConfig(
        model=ModelSpec(
            backend="transformers",
            model_name="microsoft/DialoGPT-small",
            max_tokens=256,
            temperature=0.5,
            extra={"device": "cpu"}
        ),
        strategy=StrategySpec(
            name="self_consistency",
            parallel=6
        ),
        verifier={
            "type": "regex",
            "pattern": r"Final Answer:\s*([+-]?\d+(?:\.\d+)?)"
        }
    )
    
    problems = [
        "Calculate 15 * 23 and provide your answer in the format 'Final Answer: <number>'",
        "Find the area of a circle with radius 5. Use π ≈ 3.14159. Format: 'Final Answer: <number>'",
        "Solve for x: 3x + 7 = 22. Format: 'Final Answer: <number>'"
    ]
    
    for problem in problems:
        print(f"\nProblem: {problem}")
        answer = think(problem, config)
        print(f"Answer: {answer.content}")
        print(f"Confidence: {answer.confidence:.3f}")


def performance_comparison_example():
    """Example: Comparing different strategies on the same problems."""
    print("\n=== Strategy Performance Comparison ===")
    
    base_model = ModelSpec(
        backend="transformers",
        model_name="microsoft/DialoGPT-small",
        max_tokens=256,
        temperature=0.7,
        extra={"device": "cpu"}
    )
    
    strategies = [
        ("Self-Consistency", StrategySpec(name="self_consistency", parallel=4)),
        ("DeepConf", StrategySpec(name="deepconf", parallel=6, deepconf={"k": 3})),
        ("Debate", StrategySpec(name="debate", parallel=3, debate={"rounds": 1}))
    ]
    
    problems = [
        "What is 234 + 567?",
        "Find the prime factorization of 60",
        "If a rectangle has length 8 and width 5, what is its perimeter?"
    ]
    
    results = {}
    
    for strategy_name, strategy_spec in strategies:
        print(f"\n--- Testing {strategy_name} ---")
        config = ThinkConfig(model=base_model, strategy=strategy_spec)
        strategy_results = []
        
        for problem in problems:
            print(f"Problem: {problem}")
            try:
                answer = think(problem, config)
                result = {
                    "problem": problem,
                    "answer": answer.content,
                    "confidence": answer.confidence,
                    "time": answer.meta.get("elapsed_s", 0)
                }
                strategy_results.append(result)
                print(f"  Answer: {answer.content[:50]}...")
                print(f"  Confidence: {answer.confidence:.3f}")
            except Exception as e:
                print(f"  Error: {e}")
                strategy_results.append({
                    "problem": problem,
                    "error": str(e)
                })
        
        results[strategy_name] = strategy_results
    
    # Summary
    print("\n=== Performance Summary ===")
    for strategy_name, strategy_results in results.items():
        successful = [r for r in strategy_results if "error" not in r]
        if successful:
            avg_confidence = sum(r["confidence"] for r in successful) / len(successful)
            avg_time = sum(r["time"] for r in successful) / len(successful)
            print(f"{strategy_name}: {len(successful)}/{len(strategy_results)} successful, "
                  f"avg confidence: {avg_confidence:.3f}, avg time: {avg_time:.2f}s")
        else:
            print(f"{strategy_name}: All attempts failed")


def main():
    """Run all examples."""
    print("ThinkMesh Mathematical Reasoning Examples")
    print("=" * 50)
    
    # Run synchronous examples
    basic_arithmetic_example()
    complex_proof_example()
    word_problem_debate_example()
    tree_search_optimization_example()
    custom_verifier_example()
    performance_comparison_example()
    
    # Run async example
    print("\nRunning GPU example...")
    asyncio.run(gpu_accelerated_example())
    
    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    main()
