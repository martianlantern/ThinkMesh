from .deepconf import deepconf_run
from .self_consistency import self_consistency_run
from .debate import debate_run
from .tree import tree_run
from .graph import graph_run

def load_strategy(name: str):
    if name == "deepconf":
        return deepconf_run
    if name == "self_consistency":
        return self_consistency_run
    if name == "debate":
        return debate_run
    if name == "tree":
        return tree_run
    if name == "graph":
        return graph_run
    raise ValueError("unknown strategy")
