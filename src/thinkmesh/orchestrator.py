import time
from typing import Any, Dict
from .config import ThinkConfig
from .sched.runner import Runner
from .adapters.base import load_thinker
from .strategies.base import load_strategy
from .reduce.majority import reduce_majority
from .reduce.judge import reduce_judge
from .reduce.verifier import build_verifier
from .telemetry.logging import get_logger
from .cache.store import Cache

class Orchestrator:
    def __init__(self, cfg: ThinkConfig):
        self.cfg = cfg
        self.logger = get_logger()
        self.cache = Cache(enabled=cfg.cache.get("enabled", True), ttl_s=cfg.cache.get("ttl_s", 86400))
        self.runner = Runner(self.cache, self.logger, budgets=cfg.budgets)

    async def run(self, task: str | Dict[str, Any]):
        thinker = await load_thinker(self.cfg.model)
        strategy = load_strategy(self.cfg.strategy.name)
        verifier = build_verifier(self.cfg.verifier) if self.cfg.verifier else None
        start = time.time()
        strat_res = await strategy(self.runner, thinker, task, self.cfg)
        reducer_name = self.cfg.reducer.get("name","majority")
        if reducer_name == "judge":
            final = await reduce_judge(self.runner, thinker, task, self.cfg, strat_res["candidates"], self.cfg.reducer)
        else:
            final = reduce_majority(strat_res["candidates"])
        elapsed = time.time() - start
        final["answer"].meta["elapsed_s"] = elapsed
        return final["answer"]
