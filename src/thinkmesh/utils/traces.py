import orjson

def dump_trace(trace: dict) -> str:
    return orjson.dumps(trace).decode()
