import asyncio
from dataclasses import dataclass
from typing import Any, Dict
from .config import ThinkConfig
from .orchestrator import Orchestrator

@dataclass
class Answer:
    content: str
    confidence: float
    meta: Dict[str, Any]

@dataclass
class Trace:
    graph_json: Dict[str, Any]
    logs_path: str | None

def think(task: str | Dict[str, Any], config: ThinkConfig) -> Answer:
    return asyncio.run(Orchestrator(config).run(task))
