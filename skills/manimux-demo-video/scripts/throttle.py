"""Enforce 60% wall-time duty cycle when user CPU quotas are unavailable."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

unit=sys.argv[1]
root=Path('/sys/fs/cgroup/user.slice')/f'user-{os.getuid()}.slice'/f'user@{os.getuid()}.service'/'app.slice'/unit
process=subprocess.Popen(sys.argv[2:]) if len(sys.argv)>2 else None
freeze=root/'cgroup.freeze'

def interrupted(signum, frame):
    raise KeyboardInterrupt

signal.signal(signal.SIGTERM,interrupted)
try:
    deadline=time.monotonic()+10
    while not freeze.exists():
        if process is not None and process.poll() is not None:
            sys.exit(process.returncode)
        if time.monotonic()>deadline:
            raise RuntimeError('Resource scope did not appear')
        time.sleep(.05)
    while freeze.exists() and (process is None or process.poll() is None):
        time.sleep(.06)
        try:
            freeze.write_text('1')
            time.sleep(.04)
            freeze.write_text('0')
        except FileNotFoundError:
            break
except BaseException:
    if freeze.exists():freeze.write_text('0')
    if process is not None and process.poll() is None:
        subprocess.run(['systemctl','--user','stop',unit],check=False)
    raise
finally:
    if freeze.exists():freeze.write_text('0')
if process is not None:
    sys.exit(process.wait())
