from ..sched.runner import Runner
from ..adapters.base import Thinker
from ..config import ThinkConfig
from ..confidence.meters import self_rated_confidence

async def self_consistency_run(runner: Runner, thinker: Thinker, task, cfg: ThinkConfig):
    k = int(cfg.strategy.parallel)
    prompts = [str(task) for _ in range(k)]
    res = await runner.generate_batched(thinker, prompts, {"max_tokens": cfg.model.max_tokens})
    cands = []
    for r in res:
        conf = r.token_logprobs and sum(r.token_logprobs[-5:])/max(1,len(r.token_logprobs[-5:])) or self_rated_confidence(r.text)
        cands.append({"text": r.text, "scores": {"conf": float(conf)}, "steps": []})
    trace = {"nodes": [{"id": i, "conf": c["scores"]["conf"]} for i,c in enumerate(cands)], "edges": []}
    return {"candidates": cands, "trace": trace}
