from ..sched.runner import Runner
from ..adapters.base import Thinker
from ..config import ThinkConfig
from ..confidence.meters import self_rated_confidence

def seed_prompts(task: str, k: int):
    return [f"You are Debater {i+1}. Read the problem and propose a solution. Then await rebuttal.\n{task}" for i in range(k)]

async def debate_run(runner: Runner, thinker: Thinker, task, cfg: ThinkConfig):
    k = int(cfg.strategy.parallel)
    rounds = int(cfg.strategy.debate.get("rounds", 2))
    prompts = seed_prompts(str(task), k)
    res = await runner.generate_batched(thinker, prompts, {"max_tokens": max(64, cfg.model.max_tokens//4)})
    texts = [r.text for r in res]
    for _ in range(rounds-1):
        rebuttals = []
        for i, t in enumerate(texts):
            others = "\n".join([f"Debater {j+1}: {texts[j]}" for j in range(len(texts)) if j!=i])
            rebuttals.append(f"Your earlier argument:\n{t}\nOpponents:\n{others}\nWrite a concise rebuttal and improved final answer.")
        res = await runner.generate_batched(thinker, rebuttals, {"max_tokens": max(64, cfg.model.max_tokens//3)})
        texts = [r.text for r in res]
    cands = [{"text": x, "scores": {"conf": float(self_rated_confidence(x))}, "steps": []} for x in texts]
    trace = {"nodes": [{"id": i, "conf": c["scores"]["conf"]} for i,c in enumerate(cands)], "edges": []}
    return {"candidates": cands, "trace": trace}
