"""Offline tests: fake transport only, never import/connect the hardware SDK."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import sine_probe as probe


class Clock:
    def __init__(self):
        self.ns = 1_000_000_000_000

    def __call__(self):
        return self.ns

    def sleep(self, seconds):
        self.ns += max(1, round(seconds * 1e9))


class Connection:
    def __init__(self, clock, frozen=False):
        self.clock = clock
        self.frozen = frozen
        self.q = np.zeros(7)

    def subscribe(self):
        serial = 1 if self.frozen else self.clock() // 1_000_000
        return {'states': [{'cur_state': 1, 'err_code': 0}],
                'inputs': [{'joint_cmd_pos': self.q.copy(), 'in_frame_serial': serial}],
                'outputs': [{'fb_joint_cmd': self.q.copy(), 'fb_joint_pos': self.q.copy(),
                             'frame_serial': serial}]}

    def send(self, conn, targets):
        self.q = np.array(targets['A'])


class SineProbeTests(unittest.TestCase):
    def test_waveform_bounds_and_smooth_endpoints(self):
        a = probe.parser().parse_args([])
        probe.validate(a)
        end = 2*a.ramp_s + a.cycles/a.freq_hz
        t = np.arange(-.01, end+.01, .0001)
        q = np.array([probe.offset(x, a) for x in t])
        v = np.gradient(q, .0001)
        acc = np.gradient(v, .0001)
        vb, ab = probe.bounds(a)
        self.assertLessEqual(abs(q).max(), a.amp_deg+1e-10)
        self.assertLessEqual(abs(v).max(), vb)
        self.assertLessEqual(abs(acc).max(), ab)
        self.assertEqual(probe.offset(0, a), 0)
        self.assertEqual(probe.offset(end, a), 0)
        self.assertLess(abs(probe.offset(1e-4, a)/1e-4), 1e-5)

    def test_phase_with_different_sample_rates_and_position_offset(self):
        rng = np.random.default_rng(1)
        t = np.arange(0, 20, .01)
        u = np.arange(0, 20, .001) + rng.uniform(-.0001, .0001, 20000)
        ref = probe.fit_sine(t, 13 + np.sin(2*np.pi*.5*t), .5)
        response = probe.fit_sine(u, 12.8 + .85*np.sin(2*np.pi*.5*(u-.075)), .5)
        self.assertAlmostEqual(probe.phase_lag(ref, response, .5), 75, places=7)
        self.assertAlmostEqual(response['amplitude_deg'], .85, places=7)

    def test_independent_sampling_and_only_selected_joint_moves(self):
        a = probe.parser().parse_args(['--cycles', '3', '--freq-hz', '1', '--ramp-s', '1'])
        c = Clock()
        conn = Connection(c)
        frames, sent, meta = [], [], {}
        with contextlib.redirect_stdout(io.StringIO()):
            probe.stream(conn, 0, np.zeros(7), a, frames, sent, meta, conn.send, c, c.sleep)
        self.assertGreater(len(frames), 3*len(sent))
        q = np.array(sent)[:, 2:]
        np.testing.assert_array_equal(q[:, :6], 0)
        self.assertGreater(np.ptp(q[:, 6]), 1.9)
        self.assertEqual(q[-1, 6], 0)
        self.assertGreaterEqual(np.diff(np.array(sent)[:, 0]).min(), 4_000_000)

    def test_frozen_feedback_aborts(self):
        a = probe.parser().parse_args([])
        c = Clock()
        conn = Connection(c, frozen=True)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'did not advance'):
            probe.stream(conn, 0, np.zeros(7), a, [], [], {}, conn.send, c, c.sleep)

    def test_error_state_holding_axis_and_nonfinite_feedback(self):
        a = probe.parser().parse_args([])
        conn = Connection(Clock())
        row = probe.snapshot(conn, 0)
        row[4] = 100
        with self.assertRaisesRegex(RuntimeError, 'Robot state'):
            probe.guard(row, np.zeros(7), np.zeros(7), a)
        row[4], row[20] = 1, .6
        with self.assertRaisesRegex(RuntimeError, 'envelope'):
            probe.guard(row, np.zeros(7), np.zeros(7), a)
        conn.q[0] = np.nan
        with self.assertRaisesRegex(RuntimeError, 'Invalid SDK'):
            probe.snapshot(conn, 0)

    def test_stalled_send_does_not_burst_to_catch_up(self):
        a = probe.parser().parse_args([])
        c = Clock()
        conn = Connection(c)
        sent = []

        def slow_send(conn, targets):
            conn.send(conn, targets)
            c.sleep(.08)

        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'loop stalled'):
            probe.stream(conn, 0, np.zeros(7), a, [], sent, {}, slow_send, c, c.sleep)
        self.assertEqual(len(sent), 1)

    def test_prepare_failure_stops_disables_closes_and_saves(self):
        conn = Connection(Clock())
        conn.version, conn.close = 123, Mock()
        drv = Mock(idx=0)
        drv.prepare.side_effect = RuntimeError('prepare failed after enabling')
        fake_driver = SimpleNamespace(ArmDriver=Mock(return_value=drv),
                                      RobotConnection=Mock(return_value=conn), send_joint_commands=Mock())
        fake_axis = SimpleNamespace(read_axis=lambda arm, joint: dict(
            lim_lo=-180, lim_hi=180, vel_max=180, acc_max=1000))
        with tempfile.TemporaryDirectory() as tmp:
            a = probe.parser().parse_args(['--execute', '--output', str(Path(tmp)/'run')])
            with patch.dict(sys.modules, {'drivers.arm_driver': fake_driver,
                                         'bench.servo_lag_probe': fake_axis}), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'prepare failed'):
                    probe.execute(a)
            drv.stop.assert_called_once()
            drv.disable.assert_called_once()
            conn.close.assert_called_once()
            self.assertEqual(json.loads((a.output/'meta.json').read_text())['status'], 'aborted')
            self.assertTrue((a.output/'frames.csv').exists())

    def test_rejects_invalid_parameters_offline(self):
        for flags in [['--amp-deg', 'nan'], ['--freq-hz', '0'], ['--hz', '1000'],
                      ['--read-hz', '50'], ['--cycles', '1']]:
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                probe.validate(probe.parser().parse_args(flags))

    def test_recorded_csv_analysis_end_to_end(self):
        import csv
        origin = 1_000_000_000_000
        ts = np.arange(0, 17, .004)
        tf = np.arange(0, 17, .001)
        sent = [[origin+round(t*1e9), origin+round(t*1e9)+1000,
                 *([0]*6), np.sin(np.pi*t)] for t in ts]
        frames = [[origin+round(t*1e9), origin+round(t*1e9), i, i, 1, 0,
                   *([0]*6), np.sin(np.pi*(t-.005)),
                   *([0]*6), np.sin(np.pi*(t-.010)),
                   *([0]*6), .9*np.sin(np.pi*(t-.075))] for i, t in enumerate(tf)]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out/'meta.json').write_text(json.dumps(dict(
                start_ns=origin, ramp_s=2, cycles=6, freq_hz=.5, joint=7)))
            for name, header, data in [('sent.csv', probe.SENT_HEADER, sent),
                                       ('frames.csv', probe.FRAME_HEADER, frames)]:
                with (out/name).open('w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(header)
                    w.writerows(data)
            with contextlib.redirect_stdout(io.StringIO()):
                probe.analyze(out)
            summary = json.loads((out/'summary.json').read_text())
            self.assertAlmostEqual(summary['lag_ms']['sent_to_fb_joint_pos'], 75, places=5)
            self.assertAlmostEqual(summary['lag_ms']['joint_cmd_pos_to_fb_joint_pos'], 70, places=5)
            self.assertAlmostEqual(summary['send_hz'], 250, places=5)
            self.assertAlmostEqual(summary['feedback_observed_hz'], 1000, places=5)
            self.assertTrue((out/'tracking.png').is_file())
            self.assertTrue((out/'tracking.pdf').is_file())


if __name__ == '__main__':
    unittest.main()
