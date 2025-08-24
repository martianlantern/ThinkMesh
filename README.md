# ThinkMesh

ThinkMesh is a python library for running  diverse reasoning paths in parallel, scoring them with internal confidence signals, reallocates compute to promising branches, and fuses outcomes with verifiers and reducers. It works with offline Hugging Face Transformers and vLLM/TGI, and with hosted APIs.

> Note: This is still in it's early development phase and breaking changes can sometimes occur

## Highlights

- Parallel reasoning with DeepConf‑style confidence gating and budget reallocation
- Offline‑first with Transformers; optional vLLM/TGI for server‑side batching
- Hosted adapters for OpenAI and Anthropic
- Async execution with dynamic micro‑batches
- Reducers (majority/judge) and pluggable verifiers (regex/numeric/custom)
- Caching, metrics, and JSON traces

## Install

```bash
git clone https://github.com/martianlantern/thinkmesh.git
cd thinkmesh
pip install -e ".[dev,transformers]"
```

## Quickstart: Offline DeepConf

```python
from thinkmesh import think, ThinkConfig, ModelSpec, StrategySpec

cfg = ThinkConfig(
  model=ModelSpec(backend="transformers", model_name="Qwen2.5-7B-Instruct",
                  max_tokens=256, temperature=0.7, seed=42, extra={"device":"cuda:0"}),
  strategy=StrategySpec(name="deepconf", parallel=8, max_steps=2,
                        deepconf={"k":5,"tau_low":-1.25,"tau_ent":2.2,"realloc_top_p":0.4}),
  reducer={"name":"majority"},
  budgets={"wall_clock_s":20,"tokens":4000},
)
ans = think("Show that the product of any three consecutive integers is divisible by 3.", cfg)
print(ans.content, ans.confidence)
```

## Quickstart: OpenAI Self‑Consistency

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
from thinkmesh import think, ThinkConfig, ModelSpec, StrategySpec
cfg = ThinkConfig(
  model=ModelSpec(backend="openai", model_name="gpt-4o-mini", max_tokens=256, temperature=0.6),
  strategy=StrategySpec(name="self_consistency", parallel=6, max_steps=1),
  reducer={"name":"majority"},
  budgets={"wall_clock_s":15,"tokens":3000},
)
print(think("List three creative uses for a paperclip.", cfg).content)
```

## CLI

```bash
thinkmesh think -m Qwen2.5-7B-Instruct --backend transformers --strategy deepconf "What is 37*43?"
```

## Examples

### Debate Strategy (hosted)

```python
from thinkmesh import think, ThinkConfig, ModelSpec, StrategySpec
cfg = ThinkConfig(
  model=ModelSpec(backend="openai", model_name="gpt-4o-mini", max_tokens=256, temperature=0.7),
  strategy=StrategySpec(name="debate", parallel=4, max_steps=2, debate={"rounds":2}),
  reducer={"name":"judge"},
  budgets={"wall_clock_s":25,"tokens":5000},
)
print(think("Argue whether every even integer > 2 is the sum of two primes.", cfg).content)
```

### vLLM Local Server

```python
from thinkmesh import think, ThinkConfig, ModelSpec, StrategySpec
cfg = ThinkConfig(
  model=ModelSpec(backend="vllm", model_name="Qwen2.5-7B-Instruct",
                  max_tokens=256, temperature=0.7, extra={"base_url":"http://localhost:8000/v1","api_key":"sk-"}),
  strategy=StrategySpec(name="deepconf", parallel=8, max_steps=2, deepconf={"k":5}),
  reducer={"name":"majority"},
  budgets={"wall_clock_s":20,"tokens":4000},
)
print(think("Give a constructive proof for the Pigeonhole Principle on a simple case.", cfg).content)
```

### Custom Verifier

```python
from thinkmesh import think, ThinkConfig, ModelSpec, StrategySpec
cfg = ThinkConfig(
  model=ModelSpec(backend="transformers", model_name="Qwen2.5-7B-Instruct", max_tokens=128),
  strategy=StrategySpec(name="self_consistency", parallel=5, max_steps=1),
  reducer={"name":"majority"},
  verifier={"type":"regex","pattern":r"Final Answer\s*:\s*.+$"},
  budgets={"wall_clock_s":10,"tokens":1500},
)
print(think("Answer with 'Final Answer: <value>' for 19*21.", cfg).content)
```

### Tree Of Thought (offline)

```python
from thinkmesh import think, ThinkConfig, ModelSpec, StrategySpec
cfg = ThinkConfig(
  model=ModelSpec(backend="transformers", model_name="Qwen2.5-7B-Instruct", max_tokens=192),
  strategy=StrategySpec(name="tree", parallel=6, max_steps=2, tree={"branches":3,"depth":2}),
  reducer={"name":"majority"},
  budgets={"wall_clock_s":20,"tokens":3500},
)
print(think("Sketch a plan to prove that sqrt(2) is irrational.", cfg).content)
```

## Traces, Metrics, Caching

Traces are emitted as JSON graphs inside the returned structure. Prometheus metrics and OpenTelemetry spans can be enabled via config extras. A local disk cache deduplicates repeated generations by hashing adapter, model, prompt, and params.

## Extending

- Implement a new backend by providing a `Thinker.generate` method that returns token text and optional token logprobs
- Add a new strategy by wiring a function in `thinkmesh/strategies` and registering by name
- Add reducers/verifiers under `thinkmesh/reduce`

## License

MIT

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

If you use this library in your work, please cite:

```bibtex
@software{thinkmesh2025,
  title        = {ThinkMesh: Parallel Thinking for LLMs},
  author       = {martianlantern},
  year         = {2025},
  note         = {Version 0.1.1},
}
```