from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any

class ModelSpec(BaseModel):
    backend: Literal["transformers","vllm","tgi","openai","anthropic"]
    model_name: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: Optional[float] = None
    seed: Optional[int] = None
    extra: Dict[str, Any] = Field(default_factory=dict)

class StrategySpec(BaseModel):
    name: Literal["deepconf","self_consistency","debate","tree","graph"]
    parallel: int = 8
    max_steps: int = 2
    deepconf: Dict[str, Any] = Field(default_factory=dict)
    debate: Dict[str, Any] = Field(default_factory=dict)
    tree: Dict[str, Any] = Field(default_factory=dict)
    graph: Dict[str, Any] = Field(default_factory=dict)

class ThinkConfig(BaseModel):
    model: ModelSpec
    strategy: StrategySpec
    reducer: Dict[str, Any] = Field(default_factory=lambda: {"name":"majority"})
    verifier: Optional[Dict[str, Any]] = None
    budgets: Dict[str, Any] = Field(default_factory=lambda: {"wall_clock_s":30,"tokens":8000})
    cache: Dict[str, Any] = Field(default_factory=lambda: {"enabled":True,"ttl_s":86400})
    telemetry: Dict[str, Any] = Field(default_factory=lambda: {"otel":False,"metrics":True,"trace_dump":True})
