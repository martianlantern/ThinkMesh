from typing import List, Dict, Any
from .base import GenResult
from ..config import ModelSpec

class AnthropicAdapter:
    def __init__(self, model: ModelSpec):
        from anthropic import AsyncAnthropic
        self.client = AsyncAnthropic()
        self.model = model

    def supports_logprobs(self) -> bool:
        return False

    def max_batch_size(self) -> int:
        return int(self.model.extra.get("batch_size", 4))

    async def generate(self, prompts: List[str], *, params: Dict[str, Any]) -> List[GenResult]:
        out = []
        for p in prompts:
            r = await self.client.messages.create(model=self.model.model_name, max_tokens=params.get("max_tokens", self.model.max_tokens), temperature=self.model.temperature, top_p=self.model.top_p, messages=[{"role":"user","content":p}])
            txt = "".join([c.text for c in r.content if getattr(c,"type","text")=="text"])
            out.append(GenResult(text=txt, tokens=None, token_logprobs=None, finish_reason="stop", meta={}))
        return out
