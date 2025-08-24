from typing import List, Tuple

def select_top_p(scored: List[Tuple[object, float]], p: float) -> List[Tuple[object, float]]:
    if not scored:
        return []
    scored = sorted(scored, key=lambda x: x[1], reverse=True)
    n = max(1, int(len(scored) * p))
    return scored[:n]
