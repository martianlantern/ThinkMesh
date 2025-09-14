"""
Type definitions for ThinkMesh.
"""
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Answer:
    content: str
    confidence: float
    meta: Dict[str, Any]


@dataclass
class Trace:
    graph_json: Dict[str, Any]
    logs_path: str | None
