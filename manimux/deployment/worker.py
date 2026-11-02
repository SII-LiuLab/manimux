"""Internal PTY bootstrap. No service starts before ownership is persisted."""

import json
import os
import sys
import time
from pathlib import Path


def main():
    launch = json.loads(Path(sys.argv[1]).read_text())
    deadline = time.monotonic() + 30
    while not Path(launch["gate"]).exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Launcher did not commit process ownership")
        time.sleep(0.05)
    os.chdir(launch["cwd"])
    os.execvpe(launch["command"][0], launch["command"], os.environ)


if __name__ == "__main__":
    main()
