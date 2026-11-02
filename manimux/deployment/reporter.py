"""Optional process-local session status; never participates in hardware cleanup."""

import logging
import os
import time

from .processes import atomic_json


def report(phase, **details):
    path = os.environ.get("MANIMUX_DEPLOYMENT_STATUS")
    if not path:
        return
    try:
        atomic_json(
            path, {"pid": os.getpid(), "phase": phase, "updated_at": time.time(), **details}
        )
    except OSError:
        logging.getLogger(__name__).exception("Deployment status write failed; runtime continues")
