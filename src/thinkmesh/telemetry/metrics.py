from prometheus_client import Counter, Histogram

class Metrics:
    def __init__(self):
        self.tokens_generated = Counter("thinkmesh_tokens_generated","")
        self.branches_pruned = Counter("thinkmesh_branches_pruned","")
        self.reallocations = Counter("thinkmesh_reallocations","")
        self.latency_ms = Histogram("thinkmesh_latency_ms","")
        self.batch_size = Histogram("thinkmesh_batch_size","")
