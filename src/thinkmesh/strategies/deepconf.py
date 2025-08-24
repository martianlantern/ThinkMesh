import math
from typing import Any, Dict, List
from ..sched.runner import Runner
from ..adapters.base import Thinker
from ..config import ThinkConfig
from ..confidence.meters import avg_logprob_last_k, entropy_last_k, self_rated_confidence

def make_variants(task: str, k: int) -> List[str]:
    out = []
    for i in range(k):
        out.append(f"{task}\nVariant #{i+1}: Continue reasoning step by step. Provide a concise final answer.")
    return out

async def deepconf_run(runner: Runner, thinker: Thinker, task: str | Dict[str, Any], cfg: ThinkConfig):
    k = int(cfg.strategy.deepconf.get("k", 5))
    tau_low = float(cfg.strategy.deepconf.get("tau_low", -1.25))
    tau_ent = float(cfg.strategy.deepconf.get("tau_ent", 2.2))
    realloc_top_p = float(cfg.strategy.deepconf.get("realloc_top_p", 0.4))
    parallel = int(cfg.strategy.parallel)
    step1_tokens = max(32, min(128, int(cfg.model.max_tokens * 0.25)))
    step2_tokens = cfg.model.max_tokens
    variants = make_variants(task if isinstance(task,str) else str(task), parallel)
    step1 = await runner.generate_batched(thinker, variants, {"max_tokens": step1_tokens})
    scored = []
    for r in step1:
        if thinker.supports_logprobs() and r.token_logprobs:
            conf = avg_logprob_last_k(r.token_logprobs, k=min(k,len(r.token_logprobs)))
            ent = entropy_last_k(r.token_logprobs, k=min(k,len(r.token_logprobs)))
        else:
            conf = self_rated_confidence(r.text)
            ent = 0.0
        keep = True
        if thinker.supports_logprobs() and r.token_logprobs:
            if conf < tau_low or ent > tau_ent:
                keep = False
        if keep:
            scored.append((r, conf))
    if not scored:
        scored = [(max(step1, key=lambda x: len(x.text)), 0.0)]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_n = max(1, int(math.ceil(len(scored) * realloc_top_p)))
    seeds = []
    for i in range(top_n):
        seeds.append(scored[i][0].text + "\nContinue and finalize the solution. Provide the final answer clearly marked.")
    step2 = await runner.generate_batched(thinker, seeds, {"max_tokens": step2_tokens})
    cands = []
    for i, r in enumerate(step2):
        if thinker.supports_logprobs() and r.token_logprobs:
            conf2 = avg_logprob_last_k(r.token_logprobs, k=min(k,len(r.token_logprobs)))
        else:
            conf2 = self_rated_confidence(r.text)
        cands.append({"text": r.text, "scores": {"conf": float(conf2)}, "steps": [{"text": seeds[i], "scores": {"conf": float(scored[min(i,len(scored)-1)][1])}}]})
    trace = {"nodes": [{"id": i, "conf": c["scores"]["conf"]} for i,c in enumerate(cands)], "edges": []}
    return {"candidates": cands, "trace": trace}
