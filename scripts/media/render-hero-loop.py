"""Make a lightweight RoboGUI background; run inside the video resource scope."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'skills/manimux-demo-video/assets/inputs/robogui-live.mp4'
OUT = ROOT / 'website/assets'


def ffmpeg(*args):
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin',
                    '-y', '-threads', '1', *map(str, args)], check=True)


if __name__ == '__main__':
    # Wrap the final 0.8 seconds into the first 0.8 seconds. The exported start
    # follows that dissolve, so looping does not introduce a hard camera cut.
    ffmpeg('-i', SOURCE, '-filter_complex',
           '[0:v]split=2[a][b];'
           '[a]trim=start=0.8:end=24,setpts=PTS-STARTPTS[body];'
           '[b]trim=start=0:end=0.8,setpts=PTS-STARTPTS[head];'
           '[body][head]xfade=transition=fade:duration=0.8:offset=22.4,format=yuv420p[v]',
           '-map', '[v]', '-an', '-t', '23.2', '-filter_complex_threads', '1',
           '-c:v', 'libx264', '-preset', 'fast', '-crf', '24', '-r', '30',
           '-threads', '1', '-movflags', '+faststart', OUT / 'hero-robogui.mp4')
    ffmpeg('-i', OUT / 'hero-robogui.mp4', '-frames:v', '1', '-threads', '1',
           '-q:v', '3', OUT / 'hero-poster.jpg')
    ffmpeg('-i', OUT / 'hero-robogui.mp4', '-an', '-f', 'null', '-')
    print('RoboGUI loop rendered and decode checked.', flush=True)
