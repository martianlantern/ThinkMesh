from typing import List, Dict, Any
from .base import GenResult
from ..config import ModelSpec

class TransformersLocal:
    def __init__(self, model: ModelSpec, pipe):
        self.model = model
        self.pipe = pipe
        self.batch_size = int(model.extra.get("batch_size", 4))

    @staticmethod
    async def create(model: ModelSpec):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = model.extra.get("device","cpu")
        dtype = model.extra.get("dtype","auto")
        tok = AutoTokenizer.from_pretrained(model.model_name, use_fast=True)
        if isinstance(dtype, str):
            if dtype == "auto":
                torch_dtype = torch.float16
            else:
                torch_dtype = getattr(torch, dtype)
        else:
            torch_dtype = dtype
        mdl = AutoModelForCausalLM.from_pretrained(model.model_name, torch_dtype=torch_dtype, device_map="auto" if device!="cpu" else None)
        return TransformersLocal(model, (mdl, tok, device))

    def supports_logprobs(self) -> bool:
        return True

    def max_batch_size(self) -> int:
        return self.batch_size

    async def generate(self, prompts: List[str], *, params: Dict[str, Any]) -> List[GenResult]:
        import torch
        mdl, tok, device = self.pipe
        inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(mdl.device)
        gen_kwargs = dict(max_new_tokens=params.get("max_tokens", self.model.max_tokens), do_sample=(self.model.temperature or 0) > 0, temperature=self.model.temperature, top_p=self.model.top_p, return_dict_in_generate=True, output_scores=True)
        with torch.no_grad():
            out = mdl.generate(**inputs, **gen_kwargs)
        scores = out.scores
        seqs = out.sequences
        res = []
        attn = inputs["attention_mask"]
        for i in range(seqs.size(0)):
            input_len = int(attn[i].sum().item())
            gen_ids = seqs[i][input_len:]
            toks = tok.convert_ids_to_tokens(gen_ids.tolist())
            lps = []
            for t in range(len(gen_ids)):
                step_logits = scores[t][i]
                logprobs = step_logits.log_softmax(dim=-1)
                lp = float(logprobs[gen_ids[t]].item())
                lps.append(lp)
            text = tok.decode(gen_ids, skip_special_tokens=True)
            res.append(GenResult(text=text, tokens=toks, token_logprobs=lps, finish_reason="length", meta={"tokens": len(gen_ids)}))
        return res
