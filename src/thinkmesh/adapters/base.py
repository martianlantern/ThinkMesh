from typing import Protocol, List, Dict, Any, Optional
from dataclasses import dataclass
from ..config import ModelSpec

@dataclass
class GenResult:
    text: str
    tokens: Optional[list]
    token_logprobs: Optional[list]
    finish_reason: str
    meta: Dict[str, Any]

class Thinker(Protocol):
    async def generate(self, prompts: List[str], *, params: Dict[str, Any]) -> List[GenResult]: ...
    def supports_logprobs(self) -> bool: ...
    def max_batch_size(self) -> int: ...

async def load_thinker(model: ModelSpec) -> Thinker:
    if model.backend == "transformers":
        from .transformers_local import TransformersLocal
        return await TransformersLocal.create(model)
    if model.backend == "openai":
        from .openai_api import OpenAIAdapter
        return OpenAIAdapter(model)
    if model.backend == "anthropic":
        from .anthropic_api import AnthropicAdapter
        return AnthropicAdapter(model)
    if model.backend == "vllm":
        from .vllm import VLLMAdapter
        return VLLMAdapter(model)
    if model.backend == "tgi":
        from .tgi import TGIAdapter
        return TGIAdapter(model)
    raise ValueError("unknown backend")
