import time
from typing import Any, Dict, List
from ..adapters.base import Thinker, GenResult
from ..sched.rate_limit import TokenBucket
from ..telemetry.metrics import Metrics

class Runner:
    def __init__(self, cache, logger, budgets: Dict[str, Any]):
        self.cache = cache
        self.logger = logger
        self.wall_clock_s = budgets.get("wall_clock_s", 30)
        self.token_budget = budgets.get("tokens", 8000)
        self.metrics = Metrics()
        self.bucket = TokenBucket(rate=1e9, capacity=1e9)

    async def generate_batched(self, thinker: Thinker, prompts: List[str], params: Dict[str, Any]) -> List[GenResult]:
        batches = []
        max_bs = max(1, thinker.max_batch_size())
        for i in range(0, len(prompts), max_bs):
            batches.append(prompts[i:i+max_bs])
        out: List[GenResult] = []
        start = time.time()
        for b in batches:
            r = await thinker.generate(b, params=params)
            out.extend(r)
            if time.time() - start > self.wall_clock_s:
                break
        return out

    async def judge(self, thinker: Thinker, prompt_pairs: List[str], params: Dict[str, Any]) -> List[GenResult]:
        return await self.generate_batched(thinker, prompt_pairs, params)
