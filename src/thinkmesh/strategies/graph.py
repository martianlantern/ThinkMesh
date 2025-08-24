from ..sched.runner import Runner
from ..adapters.base import Thinker
from ..config import ThinkConfig
from ..confidence.meters import self_rated_confidence

async def graph_run(runner: Runner, thinker: Thinker, task, cfg: ThinkConfig):
    k = int(cfg.strategy.parallel)
    prompts = [f"{task}\nPath {i+1}: explore a unique chain of thought and propose an answer." for i in range(k)]
    mids = await runner.generate_batched(thinker, prompts, {"max_tokens": max(64, cfg.model.max_tokens//3)})
    cont = [m.text + "\nCross-check assumptions and provide the final answer." for m in mids]
    finals = await runner.generate_batched(thinker, cont, {"max_tokens": cfg.model.max_tokens})
    cands = [{"text": r.text, "scores": {"conf": float(self_rated_confidence(r.text))}, "steps": []} for r in finals]
    trace = {"nodes": [{"id": i, "conf": c["scores"]["conf"]} for i,c in enumerate(cands)], "edges": []}
    return {"candidates": cands, "trace": trace}
