# ThinkMesh

ThinkMesh is a Python library for running diverse reasoning paths in parallel with language models

## Installation

```bash
git clone https://github.com/martianlantern/thinkmesh.git
cd thinkmesh
pip install -e ".[dev,transformers]"
```

## Basic Usage

```python
from thinkmesh import think, ThinkConfig, ModelSpec, StrategySpec

config = ThinkConfig(
    model=ModelSpec(
        backend="transformers",
        model_name="Qwen2.5-7B-Instruct",
        max_tokens=512,
        temperature=0.7,
        extra={
            "device": "cuda:0", 
            "dtype": "float16",
            "batch_size": 16
        }
    ),
    strategy=StrategySpec(
        name="deepconf",
        parallel=12,
        max_steps=2,
        deepconf={
            "k": 5,
            "tau_low": -1.0,
            "realloc_top_p": 0.4
        }
    ),
    budgets={"wall_clock_s": 60, "tokens": 8000}
)

answer = think("What is 2 + 2?", config)
print(f"Answer: {answer.content}")
print(f"Confidence: {answer.confidence:.3f}")
```

## Strategies

ThinkMesh supports five reasoning strategies:

DeepConf: Two-stage reasoning with confidence-based filtering and compute reallocation. Best for complex mathematical proofs and multi-step reasoning problems.

Self-Consistency: Generates multiple independent solutions and selects the most common answer via majority voting. Fast and effective for math problems and factual questions.

Debate: Multiple agents argue different positions through several rounds of discussion. Good for controversial topics and validation of different perspectives.

Tree of Thoughts: Systematic exploration of reasoning space using tree search with branching and depth control. Ideal for planning tasks and creative problem solving.

Graph: Reasoning paths that can reference and build upon each other. Suitable for problems requiring integration of multiple interconnected concepts.

## Configuration Examples

```python
# Self-consistency for math problems
self_config = ThinkConfig(
    model=ModelSpec(backend="transformers", model_name="Qwen2.5-7B-Instruct"),
    strategy=StrategySpec(name="self_consistency", parallel=8, max_steps=1)
)

# Debate for complex topics
debate_config = ThinkConfig(
    model=ModelSpec(backend="transformers", model_name="Qwen2.5-7B-Instruct"),
    strategy=StrategySpec(name="debate", parallel=4, debate={"rounds": 3})
)

# Tree of thoughts for planning
tree_config = ThinkConfig(
    model=ModelSpec(backend="transformers", model_name="Qwen2.5-7B-Instruct"),
    strategy=StrategySpec(name="tree", parallel=6, tree={"branches": 3, "depth": 2})
)
```

## Testing

Run the full test suite:

```bash
python scripts/run_full_test_suite.py
```

Run specific test categories:

```bash
pytest tests/unit/ -v
pytest tests/integration/ -v  
pytest tests/benchmarks/ -v
```

## Benchmarking

Run GSM8K mathematical reasoning benchmarks:

```bash
python scripts/run_benchmarks.py --model medium_gpu --strategies deepconf_small self_consistency_small --num-problems 10

python scripts/run_benchmarks.py --model large_gpu --strategies deepconf_large tree --num-problems 50
```

Generate performance reports:

```bash
python scripts/generate_report.py benchmark_results/
```

## Performance Monitoring

```python
from thinkmesh.production import PerformanceMonitor, validate_config

# Validate configuration
validation_result = validate_config(config)
if validation_result["warnings"]:
    print(f"Warnings: {validation_result['warnings']}")

# Monitor performance
monitor = PerformanceMonitor()
monitor.start_monitoring()

answer = think(problem, config)

summary = monitor.get_performance_summary(minutes=30)
print(f"Throughput: {summary['avg_tokens_per_second']:.0f} tokens/sec")
```

## Command Line Interface

```bash
# Basic usage
thinkmesh think "What is the derivative of x^3?" --backend transformers --model Qwen2.5-7B-Instruct

# With strategy options
thinkmesh think "Solve this equation" --strategy deepconf --parallel 8 --device cuda:0
```

## Backends

ThinkMesh supports multiple backends:

Transformers: Local HuggingFace models with GPU acceleration
vLLM: High-throughput inference server
OpenAI/Anthropic: External model via API (This is not well tested yet) :' (
TGI: Text Generation Inference server

## Examples

Mathematical reasoning examples:

```bash
python examples/math_problems.py
```

GSM8K benchmarking workflow:

```bash
python examples/gsm8k_benchmark.py
```

## Contributing

Development setup:

```bash
git clone https://github.com/martianlantern/thinkmesh.git
cd thinkmesh
pip install -e ".[dev,transformers]"
```

Run tests before submitting:

```bash
python scripts/run_full_test_suite.py --quick
```

## References

```bibex
@misc{deepconf2025,
  title         = {DeepConf: Deep Think with Confidence},
  year          = {2025},
  howpublished  = {\url{https://jiaweizzhao.github.io/deepconf/}}
}

@misc{wang2022selfconsistency,
  title         = {Self-Consistency Improves Chain-of-Thought Reasoning in Language Models},
  author        = {Wang, Xuezhi and Wei, Jason and others},
  year          = {2022},
  eprint        = {2203.11171},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

@misc{yao2023tree,
  title         = {Tree of Thoughts: Deliberate Problem Solving with Large Language Models},
  author        = {Yao, Shunyu and others},
  year          = {2023},
  eprint        = {2305.10601},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}
```

## Citation

```bibtex
@software{thinkmesh2025,
  title        = {ThinkMesh: Parallel Reasoning for Language Models},
  author       = {ThinkMesh Contributors},
  year         = {2025},
  url          = {https://github.com/martianlantern/thinkmesh}
}
```