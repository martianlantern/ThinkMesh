# Quick Start Guide

This guide will get you up and running with ThinkMesh in 5 minutes.

## Installation

```bash
# Clone the repository
git clone https://github.com/martianlantern/thinkmesh.git
cd thinkmesh

# Install with transformers support for local models
pip install -e ".[dev,transformers]"

# For vLLM support (optional)
pip install -e ".[vllm]"

# For OpenAI/Anthropic APIs (optional)
pip install -e ".[openai,anthropic]"
```

## Basic Usage

### 1. Simple Math Problem

```python
from thinkmesh import think, ThinkConfig, ModelSpec, StrategySpec

# Configure ThinkMesh
config = ThinkConfig(
    model=ModelSpec(
        backend="transformers",
        model_name="microsoft/DialoGPT-medium",  # Small model for testing
        max_tokens=256,
        temperature=0.7,
        extra={"device": "cpu"}  # Use "cuda:0" if you have GPU
    ),
    strategy=StrategySpec(
        name="self_consistency",
        parallel=4,  # Generate 4 parallel solutions
        max_steps=1
    ),
    budgets={"wall_clock_s": 30, "tokens": 2000}
)

# Solve a problem
answer = think("What is 127 * 34? Show your calculation.", config)

print(f"Answer: {answer.content}")
print(f"Confidence: {answer.confidence:.3f}")
```

### 2. With GPU Acceleration

```python
from thinkmesh import think, ThinkConfig, ModelSpec, StrategySpec

# GPU-optimized configuration
config = ThinkConfig(
    model=ModelSpec(
        backend="transformers",
        model_name="microsoft/DialoGPT-medium",
        max_tokens=512,
        temperature=0.7,
        extra={
            "device": "cuda:0",
            "dtype": "float16",  # Use half precision for speed
            "batch_size": 8      # Process multiple prompts together
        }
    ),
    strategy=StrategySpec(
        name="deepconf",
        parallel=8,
        max_steps=2,
        deepconf={
            "k": 5,                # Look at last 5 tokens for confidence
            "tau_low": -1.25,      # Confidence threshold
            "tau_ent": 2.2,        # Entropy threshold
            "realloc_top_p": 0.4   # Reallocate to top 40% of candidates
        }
    ),
    budgets={"wall_clock_s": 60, "tokens": 4000}
)

answer = think("Prove that the square root of 2 is irrational.", config)
print(answer.content)
```

### 3. Using Different Strategies

```python
# Self-Consistency: Multiple independent attempts, majority vote
self_consistency_config = ThinkConfig(
    model=ModelSpec(backend="transformers", model_name="microsoft/DialoGPT-small"),
    strategy=StrategySpec(name="self_consistency", parallel=6)
)

# Debate: Multiple agents argue and refine answers
debate_config = ThinkConfig(
    model=ModelSpec(backend="transformers", model_name="microsoft/DialoGPT-small"),
    strategy=StrategySpec(
        name="debate", 
        parallel=4, 
        debate={"rounds": 2}
    )
)

# Tree of Thoughts: Systematic exploration
tree_config = ThinkConfig(
    model=ModelSpec(backend="transformers", model_name="microsoft/DialoGPT-small"),
    strategy=StrategySpec(
        name="tree",
        parallel=6,
        tree={"branches": 3, "depth": 2}
    )
)

problem = "A farmer has chickens and cows. Together they have 30 heads and 74 legs. How many chickens and cows are there?"

# Try each strategy
for name, config in [
    ("Self-Consistency", self_consistency_config),
    ("Debate", debate_config), 
    ("Tree of Thoughts", tree_config)
]:
    answer = think(problem, config)
    print(f"{name}: {answer.content[:100]}... (confidence: {answer.confidence:.3f})")
```

## Working with Results

```python
from thinkmesh import think

answer = think("What is 2^10?", config)

# Access the answer
print(f"Content: {answer.content}")
print(f"Confidence: {answer.confidence}")

# Access metadata
print(f"Execution time: {answer.meta.get('elapsed_s', 0):.2f}s")
print(f"Total tokens: {answer.meta.get('total_tokens', 0)}")

# The answer object contains:
# - content: The final answer text
# - confidence: Confidence score (0.0 to 1.0)
# - meta: Dictionary with execution metadata
```

## Using Verifiers

Verifiers help ensure answers follow specific formats:

```python
config = ThinkConfig(
    model=ModelSpec(backend="transformers", model_name="microsoft/DialoGPT-small"),
    strategy=StrategySpec(name="self_consistency", parallel=4),
    verifier={
        "type": "regex",
        "pattern": r"Final Answer:\s*(\d+)"  # Expect "Final Answer: <number>"
    }
)

answer = think(
    "Calculate 15 * 8. Provide your answer in the format 'Final Answer: <number>'", 
    config
)
print(answer.content)
```

## Command Line Interface

ThinkMesh also provides a CLI for quick testing:

```bash
# Basic usage
thinkmesh think "What is 17 * 19?" --backend transformers --model microsoft/DialoGPT-small

# With specific strategy
thinkmesh think "Solve x^2 + 5x + 6 = 0" \
    --backend transformers \
    --model microsoft/DialoGPT-medium \
    --strategy deepconf \
    --parallel 8 \
    --device cuda:0

# Save results to file
thinkmesh think "Explain photosynthesis" \
    --backend transformers \
    --model microsoft/DialoGPT-medium \
    --strategy debate \
    --output results.json
```

## Running Tests

Verify your installation works correctly:

```bash
# Run basic tests
pytest tests/unit/ -v -m "unit and not gpu"

# Run integration tests  
pytest tests/integration/ -v -m "integration and not gpu and not slow"

# Run quick benchmark
python scripts/run_benchmarks.py --model small_cpu --strategies self_consistency_small --quick
```

## Next Steps

1. **Learn about strategies**: Read [Strategies Guide](strategies.md) to understand when to use each approach
2. **Optimize performance**: See [Performance Guide](performance.md) for GPU optimization tips
3. **Configure properly**: Check [Configuration Guide](configuration.md) for all options
4. **Run benchmarks**: Use [Benchmarking Guide](benchmarking.md) to evaluate performance
5. **Explore examples**: Look at [Examples](examples/) for real-world use cases

## Common Issues

### Out of Memory
- Reduce `parallel` count or `batch_size`
- Use smaller model or `dtype="float16"`
- Set `max_tokens` to smaller value

### Slow Performance
- Enable GPU with `device="cuda:0"`
- Increase `batch_size` for better throughput
- Use `dtype="float16"` for faster inference
- Consider vLLM backend for production

### Import Errors
- Make sure you installed the right extras: `pip install -e ".[transformers]"`
- For GPU support, install PyTorch with CUDA support

### Model Loading Issues
- Check model name is correct and accessible
- Ensure sufficient disk space for model download
- Verify internet connection for first-time download
