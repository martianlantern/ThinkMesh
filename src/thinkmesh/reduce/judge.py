from typing import Dict, Any, List
from ..types import Answer

judge_prompt = "You are a strict judge comparing two candidate answers for the SAME question. Return JSON with keys winner (\"A\"|\"B\"|\"tie\"), scoreA (0..1), scoreB (0..1), rationale."

async def reduce_judge(runner, thinker, task, cfg, candidates: List[Dict[str, Any]], rconf: Dict[str, Any]) -> Dict[str, Any]:
    if len(candidates) == 1:
        c = candidates[0]
        ans = Answer(content=c["text"], confidence=float(c["scores"].get("conf", 0.0)), meta={})
        return {"answer": ans, "candidates": candidates}
    pairs = []
    for i in range(0, len(candidates)-1, 2):
        a = candidates[i]["text"]
        b = candidates[i+1]["text"]
        q = f"{judge_prompt}\nQuestion:\n{task}\nA:\n{a}\nB:\n{b}\nJSON:"
        pairs.append(q)
    res = await runner.judge(thinker, pairs, {"max_tokens": 128})
    wins = []
    for i, r in enumerate(res):
        txt = r.text
        w = "A" if "winner" in txt and '\"A\"' in txt else "B" if '\"B\"' in txt else "tie"
        if w == "A":
            wins.append(candidates[2*i])
        elif w == "B":
            wins.append(candidates[2*i+1])
    if not wins:
        wins = [max(candidates, key=lambda c: c["scores"].get("conf", 0.0))]
    best = max(wins, key=lambda c: c["scores"].get("conf", 0.0))
    ans = Answer(content=best["text"], confidence=float(best["scores"].get("conf", 0.0)), meta={})
    return {"answer": ans, "candidates": candidates}
