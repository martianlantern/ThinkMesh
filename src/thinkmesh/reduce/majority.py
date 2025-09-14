import re
from collections import Counter
from typing import Dict, Any, List
from ..types import Answer

def normalize(x: str) -> str:
    s = x.strip().lower()
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if m:
        return m.group(0)
    return s

def reduce_majority(cands: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [normalize(c["text"]) for c in cands]
    counts = Counter(keys)
    top_key, _ = counts.most_common(1)[0]
    filtered = [c for c in cands if normalize(c["text"]) == top_key]
    best = max(filtered, key=lambda c: c["scores"].get("conf", 0.0))
    ans = Answer(content=best["text"], confidence=float(best["scores"].get("conf", 0.0)), meta={"votes": dict(counts)})
    return {"answer": ans, "candidates": cands}
