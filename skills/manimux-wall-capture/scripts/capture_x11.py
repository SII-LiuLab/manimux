#!/usr/bin/env python3
"""Bounded X11 region capture only; never starts ManiMux or controls a robot."""
import argparse
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--display', default=os.environ.get('DISPLAY'))
    parser.add_argument('--region', required=True, help='Observed WxH+X+Y in screen pixels')
    parser.add_argument('--duration', type=int, default=60)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if os.name != 'posix' or os.environ.get('XDG_SESSION_TYPE', '').lower() == 'wayland':
        parser.error('Use the native window/PipeWire recorder on this display system')
    if not args.display or not shutil.which('ffmpeg'):
        parser.error('An actual X11 DISPLAY and installed ffmpeg are required')
    match = re.fullmatch(r'(\d+)x(\d+)\+(\d+)\+(\d+)', args.region)
    if not match:
        parser.error('Region must be an observed WxH+X+Y rectangle')
    width, height, x, y = map(int, match.groups())
    if width < 1280 or height < 720 or width > 2560 or height > 1440:
        parser.error('Use a native 720p, 1080p or 1440p-sized region')
    if width % 2 or height % 2 or abs(width / height - 16 / 9) > .01:
        parser.error('Region must have even dimensions and 16:9 aspect ratio')
    if not 1 <= args.duration <= 600:
        parser.error('Duration must be between 1 and 600 seconds')
    if args.output.exists() or args.output.suffix.lower() != '.mp4':
        parser.error('Choose a new .mp4 output path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-n',
               '-thread_queue_size', '32', '-f', 'x11grab', '-framerate', '30',
               '-video_size', f'{width}x{height}', '-i', f'{args.display}+{x},{y}',
               '-t', str(args.duration), '-an', '-filter_threads', '1',
               '-c:v', 'libx264', '-threads', '1', '-preset', 'veryfast', '-crf', '20',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(args.output)]
    os.nice(10)
    print('Recording GUI region only. The operator controls all robot motion.', flush=True)
    process = subprocess.Popen(command, start_new_session=True)
    interrupted = False
    try:
        code = process.wait()
    except KeyboardInterrupt:
        interrupted = True
        process.send_signal(signal.SIGINT)
        try:
            code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise RuntimeError('Recorder did not finalize; inspect the incomplete file')
    if (code != 0 and not interrupted) or not args.output.exists() or args.output.stat().st_size == 0:
        raise RuntimeError('Capture failed; inspect FFmpeg output and do not submit an incomplete file')
    print(f'Capture stopped: {args.output.resolve()}')
    print('Review motion and validate with submission.py before sharing.')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
