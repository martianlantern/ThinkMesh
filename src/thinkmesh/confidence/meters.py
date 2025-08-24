import math
import re
from typing import List

def avg_logprob_last_k(lps: List[float], k: int = 5) -> float:
    if not lps:
        return 0.0
    k = max(1, min(k, len(lps)))
    return float(sum(lps[-k:]) / k)

def entropy_last_k(lps: List[float], k: int = 5) -> float:
    if not lps:
        return 0.0
    k = max(1, min(k, len(lps)))
    xs = [math.exp(x) for x in lps[-k:]]
    s = sum(xs)
    ps = [x/s for x in xs]
    return float(-sum(p*math.log(p+1e-12) for p in ps))

def self_rated_confidence(text: str) -> float:
    m = re.search(r"<confidence>\s*([0-1](?:\.\d+)?)", text)
    if not m:
        return 0.5
    try:
        return max(0.0, min(1.0, float(m.group(1))))
    except:
        return 0.5
