# Reasoning Strategies

ThinkMesh supports multiple reasoning strategies, each optimized for different types of problems. This guide explains when and how to use each strategy.

## Strategy Overview

| Strategy | Best For | Complexity | Parallel Count | Steps |
|----------|----------|------------|----------------|--------|
| Self-Consistency | Simple problems, quick results | Low | 4-16 | 1 |
| DeepConf | Complex reasoning, quality control | High | 6-12 | 2-3 |
| Debate | Controversial topics, validation | Medium | 3-6 | 2-4 |
| Tree of Thoughts | Systematic exploration | High | 6-12 | 2-4 |
| Graph | Connected reasoning paths | Medium | 4-8 | 2-3 |

## Self-Consistency

**Concept**: Generate multiple independent solutions and select the most common answer.

**Best for**:
- Math problems with clear answers
- Factual questions
- Quick prototyping
- When you need fast results

**Configuration**:
```python
strategy = StrategySpec(
    name="self_consistency",
    parallel=8,        # Number of independent attempts
    max_steps=1        # Single step generation
)
```

**Example**:
```python
from thinkmesh import think, ThinkConfig, ModelSpec, StrategySpec

config = ThinkConfig(
    model=ModelSpec(
        backend="transformers",
        model_name="microsoft/DialoGPT-medium",
        max_tokens=256,
        temperature=0.7,
        extra={"device": "cuda:0"}
    ),
    strategy=StrategySpec(
        name="self_consistency",
        parallel=12,  # Generate 12 solutions
        max_steps=1
    )
)

answer = think("What is 234 * 567?", config)
```

**Pros**:
- Simple and fast
- Works well for problems with clear correct answers
- Low computational overhead per attempt

**Cons**:
- May miss complex reasoning steps
- Doesn't improve answers through iteration
- Can be fooled by systematic model biases

## DeepConf (Confidence-Based Reasoning)

**Concept**: Two-stage process where initial attempts are filtered by confidence, and promising candidates are expanded with more computation.

**Best for**:
- Complex mathematical proofs
- Multi-step reasoning problems
- When answer quality is critical
- Problems requiring verification

**Configuration**:
```python
strategy = StrategySpec(
    name="deepconf",
    parallel=8,
    max_steps=2,
    deepconf={
        "k": 5,                    # Tokens to examine for confidence
        "tau_low": -1.25,          # Low confidence threshold
        "tau_ent": 2.2,            # High entropy threshold  
        "realloc_top_p": 0.4       # Reallocate to top 40%
    }
)
```

**Parameters**:
- `k`: Number of final tokens to examine for confidence scoring
- `tau_low`: Threshold below which candidates are filtered out
- `tau_ent`: Entropy threshold above which candidates are filtered out
- `realloc_top_p`: Fraction of top candidates to continue with

**Example**:
```python
config = ThinkConfig(
    model=ModelSpec(
        backend="transformers", 
        model_name="Qwen2.5-7B-Instruct",
        max_tokens=512,
        temperature=0.7,
        extra={"device": "cuda:0", "dtype": "float16"}
    ),
    strategy=StrategySpec(
        name="deepconf",
        parallel=10,
        max_steps=2,
        deepconf={
            "k": 5,
            "tau_low": -1.0,       # More permissive threshold
            "tau_ent": 2.5,        # Higher entropy tolerance
            "realloc_top_p": 0.3   # Focus on top 30%
        }
    )
)

answer = think(
    "Prove that there are infinitely many prime numbers.", 
    config
)
```

**How it works**:
1. **Stage 1**: Generate initial reasoning with reduced token budget
2. **Filtering**: Score candidates using token probabilities and entropy
3. **Stage 2**: Expand promising candidates with full token budget
4. **Selection**: Choose best final answer based on confidence

**Pros**:
- Allocates compute efficiently to promising paths
- Filters out low-quality reasoning early
- Produces high-confidence answers
- Good for complex, multi-step problems

**Cons**:
- Requires models that support logprobs
- More complex configuration
- Higher computational cost
- May filter out unconventional but correct approaches

## Debate

**Concept**: Multiple agents argue different positions, then refine their arguments through multiple rounds.

**Best for**:
- Controversial or subjective topics
- Problems with multiple valid approaches
- When you need to consider different perspectives
- Argument validation and refinement

**Configuration**:
```python
strategy = StrategySpec(
    name="debate",
    parallel=4,              # Number of debaters
    max_steps=3,            # Number of rounds
    debate={
        "rounds": 2         # Rounds of rebuttals
    }
)
```

**Example**:
```python
config = ThinkConfig(
    model=ModelSpec(
        backend="transformers",
        model_name="microsoft/DialoGPT-large",
        max_tokens=384,
        temperature=0.8,  # Higher temperature for diversity
        extra={"device": "cuda:0"}
    ),
    strategy=StrategySpec(
        name="debate",
        parallel=4,
        max_steps=3,
        debate={"rounds": 2}
    )
)

answer = think(
    "Should artificial intelligence development be regulated? Discuss the pros and cons.",
    config
)
```

**How it works**:
1. **Initial Arguments**: Each debater proposes a position
2. **Rebuttal Rounds**: Debaters respond to opponents' arguments
3. **Refinement**: Arguments evolve through interaction
4. **Final Selection**: Best argument selected via voting or judging

**Pros**:
- Explores multiple perspectives
- Arguments improve through iteration
- Good for complex, nuanced topics
- Reduces single-perspective bias

**Cons**:
- Can be slow with many rounds
- May not converge to single answer
- Requires careful prompt engineering
- Higher token usage

## Tree of Thoughts

**Concept**: Systematic exploration of reasoning space using tree search with branching and depth control.

**Best for**:
- Problems requiring systematic exploration
- Multi-step planning tasks
- Creative problem solving
- When you need to explore solution space thoroughly

**Configuration**:
```python
strategy = StrategySpec(
    name="tree",
    parallel=8,
    max_steps=3,
    tree={
        "branches": 4,      # Branches per node
        "depth": 2          # Maximum tree depth
    }
)
```

**Example**:
```python
config = ThinkConfig(
    model=ModelSpec(
        backend="transformers",
        model_name="microsoft/DialoGPT-large",
        max_tokens=384,
        temperature=0.7,
        extra={"device": "cuda:0", "dtype": "float16"}
    ),
    strategy=StrategySpec(
        name="tree",
        parallel=12,
        max_steps=3,
        tree={
            "branches": 4,  # 4 different approaches per level
            "depth": 2      # 2 levels of expansion
        }
    )
)

answer = think(
    "Plan a 7-day itinerary for visiting Tokyo, considering budget constraints and cultural experiences.",
    config
)
```

**How it works**:
1. **Root**: Start with the original problem
2. **Branching**: Generate multiple reasoning paths
3. **Expansion**: Develop each promising branch further
4. **Depth Control**: Limit exploration to manageable depth
5. **Evaluation**: Score paths and select best solution

**Pros**:
- Systematic exploration
- Good coverage of solution space
- Finds creative solutions
- Balances breadth and depth

**Cons**:
- Exponential growth with depth/branches
- Can be computationally expensive
- May over-explore simple problems
- Requires tuning of depth/branches

## Graph-Based Reasoning

**Concept**: Reasoning paths that can reference and build upon each other, forming a connected graph.

**Best for**:
- Problems with interconnected concepts
- When reasoning steps depend on each other
- Complex analysis requiring multiple viewpoints
- Integration of different reasoning approaches

**Configuration**:
```python
strategy = StrategySpec(
    name="graph",
    parallel=6,
    max_steps=2
)
```

**Example**:
```python
config = ThinkConfig(
    model=ModelSpec(
        backend="transformers",
        model_name="microsoft/DialoGPT-medium",
        max_tokens=512,
        temperature=0.7,
        extra={"device": "cuda:0"}
    ),
    strategy=StrategySpec(
        name="graph",
        parallel=6,
        max_steps=2
    )
)

answer = think(
    "Analyze the economic, environmental, and social impacts of renewable energy adoption.",
    config
)
```

**How it works**:
1. **Path Generation**: Create diverse reasoning paths
2. **Cross-referencing**: Paths can reference insights from other paths
3. **Integration**: Combine insights from multiple perspectives
4. **Synthesis**: Generate unified final answer

**Pros**:
- Captures interconnected reasoning
- Good for complex, multi-faceted problems
- Enables knowledge integration
- Flexible reasoning structure

**Cons**:
- Less structured than tree search
- Harder to debug reasoning process
- May produce unfocused results
- Requires careful prompt design

## Strategy Selection Guide

### By Problem Type

**Mathematical/Logical Problems**:
- Simple: Self-Consistency
- Complex: DeepConf
- Multi-step: Tree of Thoughts

**Creative/Open-ended**:
- Single perspective: Self-Consistency
- Multiple perspectives: Debate
- Systematic exploration: Tree of Thoughts

**Analysis/Research**:
- Factual: Self-Consistency
- Multi-faceted: Graph
- Argumentative: Debate

**Planning/Strategy**:
- Simple: Self-Consistency  
- Complex: Tree of Thoughts
- Multi-stakeholder: Debate

### By Resource Constraints

**Low Compute Budget**: Self-Consistency (parallel=4-8)
**Medium Compute Budget**: DeepConf or Graph (parallel=6-8)  
**High Compute Budget**: Tree of Thoughts or Debate (parallel=8-16)

### By Quality Requirements

**Speed over Quality**: Self-Consistency
**Balanced**: DeepConf or Graph
**Quality over Speed**: Tree of Thoughts or Debate

## Advanced Configuration Tips

### Parallel Count Guidelines
- **CPU**: 2-4 parallel processes
- **Single GPU**: 4-8 parallel processes  
- **Multiple GPUs**: 8-16 parallel processes
- **A100 80GB**: 12-32 parallel processes

### Temperature Settings
- **Deterministic**: 0.0-0.3
- **Balanced**: 0.5-0.8
- **Creative**: 0.8-1.2

### Token Budget Allocation
- **Simple problems**: 128-256 tokens
- **Medium complexity**: 256-512 tokens
- **Complex reasoning**: 512-1024 tokens
- **Long-form generation**: 1024+ tokens

### Multi-Strategy Approaches

You can combine strategies for even better results:

```python
# First pass: Tree exploration
tree_config = ThinkConfig(
    strategy=StrategySpec(name="tree", parallel=8, tree={"branches": 3, "depth": 2})
)
candidates = think_multi(problem, tree_config)

# Second pass: DeepConf refinement on best candidates  
deepconf_config = ThinkConfig(
    strategy=StrategySpec(name="deepconf", parallel=6)
)
final_answer = think(candidates.best_k(3), deepconf_config)
```

This allows you to get the systematic exploration of Tree of Thoughts with the quality filtering of DeepConf.
