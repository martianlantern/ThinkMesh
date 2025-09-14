import asyncio
from typing import Any, Dict
from .config import ThinkConfig
from .orchestrator import Orchestrator
from .types import Answer, Trace

def think(task: str | Dict[str, Any], config: ThinkConfig) -> Answer:
    return asyncio.run(Orchestrator(config).run(task))
