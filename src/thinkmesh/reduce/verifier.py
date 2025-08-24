import re
from typing import Callable, Dict, Any, Optional

def build_verifier(cfg: Optional[Dict[str, Any]]) -> Callable[[str], tuple[bool,float]]:
    if not cfg:
        return lambda x: (True, 1.0)
    if cfg.get("type") == "regex":
        pattern = re.compile(cfg.get("pattern",".*"), re.S)
        def f(x: str):
            ok = bool(pattern.search(x))
            return (ok, 1.0 if ok else 0.0)
        return f
    if cfg.get("type") == "numeric":
        def f(x: str):
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", x)
            return (bool(m), 1.0 if m else 0.0)
        return f
    return lambda x: (True, 1.0)
