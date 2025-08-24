from typing import List, Dict, Any
from .base import GenResult
from ..config import ModelSpec

class TGIAdapter:
    def __init__(self, model: ModelSpec):
        self.model = model

    def supports_logprobs(self) -> bool:
        return False

    def max_batch_size(self) -> int:
        return int(self.model.extra.get("batch_size", 8))

    async def generate(self, prompts: List[str], *, params: Dict[str, Any]) -> List[GenResult]:
        from text_generation import AsyncClient
        client = AsyncClient(self.model.extra.get("endpoint","http://localhost:8080"))
        res = []
        for p in prompts:
            r = await client.generate(p, max_new_tokens=params.get("max_tokens", self.model.max_tokens), temperature=self.model.temperature, top_p=self.model.top_p)
            res.append(GenResult(text=r.generated_text, tokens=None, token_logprobs=None, finish_reason="stop", meta={}))
        return res
