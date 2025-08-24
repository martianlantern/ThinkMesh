from ..sched.runner import Runner
from ..adapters.base import Thinker
from ..config import ThinkConfig
from ..confidence.meters import self_rated_confidence

async def tree_run(runner: Runner, thinker: Thinker, task, cfg: ThinkConfig):
    branches = int(cfg.strategy.tree.get("branches", max(2, cfg.strategy.parallel//2)))
    depth = int(cfg.strategy.tree.get("depth", min(2, cfg.strategy.max_steps)))
    frontier = [str(task)]
    for d in range(depth):
        prompts = []
        for f in frontier:
            for b in range(branches):
                prompts.append(f"{f}\nBranch {b+1}: explore a distinct line of reasoning and move toward a final answer.")
        res = await runner.generate_batched(thinker, prompts, {"max_tokens": max(64, cfg.model.max_tokens//3)})
        texts = [r.text for r in res]
        frontier = [t + "\nConclude with a final answer." for t in texts[:branches]]
    finals = await runner.generate_batched(thinker, frontier, {"max_tokens": cfg.model.max_tokens})
    cands = [{"text": r.text, "scores": {"conf": float(self_rated_confidence(r.text))}, "steps": []} for r in finals]
    trace = {"nodes": [{"id": i, "conf": c["scores"]["conf"]} for i,c in enumerate(cands)], "edges": []}
    return {"candidates": cands, "trace": trace}
