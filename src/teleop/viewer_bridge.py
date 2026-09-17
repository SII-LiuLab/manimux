#!/usr/bin/env python3
"""Optional, best-effort mirror of live joint state into universal_viewer.

teleop is the source of truth here -- there is no policy yet, so this only
ever publishes RobotSnapshot ("observe mode": state only, no PolicyPlan, no
pause/home control channel). See docs/viewer.md for why this is its own
module instead of living inside drivers/arm_driver.py, and for the
observe-mode precedent this follows.

universal_policy_viewer is only imported when a caller actually asks for
--viewer -- ViewerBridge(enabled=False) (the default) never imports it, so
running teleop with no --viewer flag needs no new dependency at all.
"""
import os
import sys
import time

# Not a pip dependency (see docs/viewer.md) -- point this at a sibling
# checkout, same convention as MARVIN_SDK in drivers/arm_driver.py.
_VIEWER_ROOT = os.environ.get(
    'UNIVERSAL_VIEWER_SRC',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 'universal_viewer', 'src'))


class ViewerBridge:
    """Publishes RobotSnapshot at a throttled rate; a no-op when disabled."""

    def __init__(self, enabled, endpoint='tcp://127.0.0.1:5568', publish_hz=20.0):
        self.enabled = enabled
        self._publisher = None
        self._snapshot_cls = None
        self._period = 1.0 / publish_hz
        self._last_publish = 0.0
        self._warned = False
        if not enabled:
            return
        if _VIEWER_ROOT not in sys.path:
            sys.path.insert(0, _VIEWER_ROOT)
        from universal_policy_viewer.bridge import ViewerPublisher
        from universal_policy_viewer.protocol import RobotSnapshot
        self._snapshot_cls = RobotSnapshot
        self._publisher = ViewerPublisher(endpoint)

    def due(self):
        """True if the next publish_state() call would actually publish.

        Lets a caller skip building/fetching `state` (e.g. a fresh
        conn.subscribe()) on ticks that would just be throttled away --
        matters in tight loops like goto_joints.py's hold loop, which
        would otherwise pay for an extra SDK read every tick just to feed
        a 20Hz viewer.
        """
        return self.enabled and time.monotonic() - self._last_publish >= self._period

    def publish_state(self, state, step=0, max_steps=0):
        """state: a RobotConnection.subscribe() dict (both arms, A then B).

        Always mirrors both arms regardless of which are actively driven --
        subscribe() reports both unconditionally, so e.g. goto_joints.py
        --arm A still shows B sitting at its real (idle) pose instead of
        picking one arbitrary side to omit.
        """
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self._last_publish < self._period:
            return
        self._last_publish = now
        try:
            joints = (list(state['outputs'][0]['fb_joint_pos'])
                      + list(state['outputs'][1]['fb_joint_pos']))
            self._publisher.publish(self._snapshot_cls(
                joint_positions=joints, cameras={}, step=step,
                max_steps=max_steps, robot='tianji'))
        except Exception as e:   # best-effort: never take down the control loop
            if not self._warned:
                print('⚠️  查看器发布失败（后续不再重复提示）：%s' % e)
                self._warned = True

    def close(self):
        if self._publisher:
            self._publisher.close()
