from typing import List, Dict, Any
from .base import GenResult
from ..config import ModelSpec

class OpenAIAdapter:
    def __init__(self, model: ModelSpec):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI()
        self.model = model

    def supports_logprobs(self) -> bool:
        return False

    def max_batch_size(self) -> int:
        return int(self.model.extra.get("batch_size", 4))

    async def generate(self, prompts: List[str], *, params: Dict[str, Any]) -> List[GenResult]:
        out = []
        for p in prompts:
            r = await self.client.responses.create(model=self.model.model_name, input=p, max_output_tokens=params.get("max_tokens", self.model.max_tokens), temperature=self.model.temperature, top_p=self.model.top_p)
            try:
                txt = r.output_text
            except Exception:
                try:
                    txt = r.output[0].content[0].text
                except Exception:
                    txt = str(r.to_dict_recursive())
            out.append(GenResult(text=txt, tokens=None, token_logprobs=None, finish_reason="stop", meta={}))
        return out
