import structlog

def get_logger():
    structlog.configure(processors=[structlog.processors.TimeStamper(fmt="iso"), structlog.processors.JSONRenderer()])
    return structlog.get_logger()
