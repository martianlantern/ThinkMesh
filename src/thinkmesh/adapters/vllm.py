from typing import List, Dict, Any
from .base import GenResult
from ..config import ModelSpec

class VLLMAdapter:
    def __init__(self, model: ModelSpec):
        self.model = model

    def supports_logprobs(self) -> bool:
        return False

    def max_batch_size(self) -> int:
        return int(self.model.extra.get("batch_size", 8))

    async def generate(self, prompts: List[str], *, params: Dict[str, Any]) -> List[GenResult]:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(base_url=self.model.extra.get("base_url","http://localhost:8000/v1"), api_key=self.model.extra.get("api_key","sk-"))
        out = []
        for p in prompts:
            r = await client.completions.create(model=self.model.model_name, prompt=p, max_tokens=params.get("max_tokens", self.model.max_tokens), temperature=self.model.temperature, top_p=self.model.top_p)
            txt = r.choices[0].text
            out.append(GenResult(text=txt, tokens=None, token_logprobs=None, finish_reason="stop", meta={}))
        return out
