"""
Centralized logging setup. Every module gets a logger via
logging.getLogger(__name__) as usual; what this adds is a request-id
correlation field (set by api/main.py's middleware into the contextvar
below) so a single client-visible error can be traced back through every
module it touched during that request, across retrieval, the pipeline, and
storage -- without that, the API layer could only ever report "Pipeline
error: <str(e)>" with no way to find the surrounding context after the
fact.
"""

import logging
import sys
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if root.handlers:
        return  # already configured (e.g. uvicorn --reload re-imports this module)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s [req=%(request_id)s] %(name)s: %(message)s"
    ))
    handler.addFilter(_RequestIdFilter())
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet down noisy third-party loggers at INFO; still surfaces at DEBUG.
    for noisy in ("httpx", "httpcore", "urllib3", "neo4j"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
