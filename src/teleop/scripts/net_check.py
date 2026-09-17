#!/usr/bin/env python3
"""XR link health check — frame rate, dropouts, button range. Run once before going live.

See docs/scripts.md for the measured interpretation thresholds.

Usage:
    python3 scripts/net_check.py [seconds]      # 20s by default
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drivers.xr_source import XRSource  # noqa: E402

DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0

src = XRSource()
src.start()
if not src.wait_for_data(30.0):
    print('收不到数据')
    src.stop()
    raise SystemExit(1)

mx = {('left', 'grip'): 0.0, ('left', 'trigger'): 0.0,
      ('right', 'grip'): 0.0, ('right', 'trigger'): 0.0}
n_over = {'left': 0, 'right': 0}   # samples where grip exceeds CLUTCH_THRESHOLD
samples = 0
gaps = []
t0 = time.monotonic()
last_age = None
while time.monotonic() - t0 < DUR:
    f = src.latest(max_age_s=1.0)
    if f is not None:
        samples += 1
        for hand in ('left', 'right'):
            for b in ('grip', 'trigger'):
                v = f.button(hand, b, 0.0)
                if v > mx[(hand, b)]:
                    mx[(hand, b)] = v
            if f.button(hand, 'grip', 0.0) >= 0.5:
                n_over[hand] += 1
    a = src.age_s()
    if a > 0.1:
        gaps.append(a)
    time.sleep(0.002)

st = src.stats()
src.stop()
print('\n=== %.0fs 窗口 ===' % DUR)
print('SDK 收帧 %d  解析错误 %d  平均 %.1f Hz'
      % (st['frames'], st['parse_errors'], st['frames'] / DUR))
print('采样 %d 次，其中 age>100ms 的采样 %d 次' % (samples, len(gaps)))
if gaps:
    print('  最大 age %.3fs' % max(gaps))
print('\n按键量程（本窗口最大值）:')
for hand in ('left', 'right'):
    print('  %-5s grip=%.3f  trigger=%.3f   grip>=0.5 的采样 %d'
          % (hand, mx[(hand, 'grip')], mx[(hand, 'trigger')], n_over[hand]))
