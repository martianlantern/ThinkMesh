# ThinkMesh Documentation

Welcome to ThinkMesh, a powerful Python library for parallel reasoning with language models. This documentation will help you understand and use all features of ThinkMesh effectively.

## Table of Contents

1. [Quick Start](quick_start.md)
2. [Configuration Guide](configuration.md)
3. [Strategies](strategies.md)
4. [Adapters](adapters.md)
5. [Benchmarking](benchmarking.md)
6. [Performance Optimization](performance.md)
7. [API Reference](api_reference.md)
8. [Examples](examples/)

## Overview

ThinkMesh enables you to run diverse reasoning paths in parallel, score them with internal confidence signals, reallocate compute to promising branches, and fuse outcomes with verifiers and reducers. It supports both local models (via HuggingFace Transformers, vLLM, TGI) and hosted APIs (OpenAI, Anthropic).

### Key Features

- **Parallel Reasoning**: Run multiple reasoning paths simultaneously
- **Confidence-Based Reallocation**: DeepConf-style confidence gating and budget reallocation
- **Multiple Strategies**: Self-consistency, DeepConf, Debate, Tree of Thoughts, Graph reasoning
- **Flexible Backends**: Local models, vLLM, TGI, OpenAI, Anthropic
- **Production Ready**: Async execution, caching, metrics, comprehensive testing
- **A100 Optimized**: Performance optimizations for high-end GPU deployment

## Getting Started

```bash
# Install with transformers support
pip install -e ".[dev,transformers]"

# Basic usage
from thinkmesh import think, ThinkConfig, ModelSpec, StrategySpec

config = ThinkConfig(
    model=ModelSpec(
        backend="transformers",
        model_name="Qwen2.5-7B-Instruct",
        max_tokens=256,
        temperature=0.7,
        extra={"device": "cuda:0"}
    ),
    strategy=StrategySpec(
        name="deepconf",
        parallel=8,
        max_steps=2
    )
)

answer = think("Solve this math problem: What is 15 * 23?", config)
print(f"Answer: {answer.content}")
print(f"Confidence: {answer.confidence}")
```

## Architecture

ThinkMesh is built with a modular architecture:

```
ThinkMesh
├── Core Engine (orchestrator.py)
├── Strategies
│   ├── DeepConf (confidence-based reasoning)
│   ├── Self-Consistency (majority voting)
│   ├── Debate (adversarial reasoning)
│   ├── Tree of Thoughts (tree exploration)
│   └── Graph (graph-based reasoning)
├── Adapters
│   ├── Transformers (local HF models)
│   ├── vLLM (high-throughput inference)
│   ├── TGI (text generation inference)
│   ├── OpenAI API
│   └── Anthropic API
├── Reducers
│   ├── Majority Vote
│   └── Judge-based
└── Utilities
    ├── Confidence Meters
    ├── Caching
    ├── Metrics & Telemetry
    └── Verification
```

## Next Steps

1. Follow the [Quick Start Guide](quick_start.md) for your first ThinkMesh program
2. Read the [Configuration Guide](configuration.md) to understand all options
3. Explore [Strategies](strategies.md) to pick the right reasoning approach
4. Check out [Examples](examples/) for real-world use cases
5. See [Performance Optimization](performance.md) for A100 deployment tips
