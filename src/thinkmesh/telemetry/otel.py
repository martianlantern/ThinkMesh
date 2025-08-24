class OTEL:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def span(self, name: str):
        class N:
            def __enter__(self_non):
                return None
            def __exit__(self_non, exc_type, exc, tb):
                return False
        return N()
