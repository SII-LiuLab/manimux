"""Reorder the accepted film, using one single-threaded FFmpeg job at a time.

Run under the existing video throttle/systemd memory scope. Originals are retained.
"""
import json
import subprocess
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'recordings/agent-demo-v5-1440p/manimux-product-demo-en-1440p30.mp4'
OUT = ROOT / 'recordings/agent-demo-v6-hook'
FONT = ROOT / 'skills/manimux-demo-video/assets/fonts/Manrope.ttf'
OUT.mkdir(parents=True, exist_ok=True)


def ffmpeg(*args):
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin',
                    '-y', '-threads', '1', *map(str, args)], check=True)


def encode(start, length, name, filters=None):
    args = ['-ss', start, '-i', SOURCE, '-t', length, '-an']
    if filters:
        args += ['-vf', filters]
    ffmpeg(*args, '-c:v', 'libx264', '-preset', 'fast', '-crf', '21',
           '-pix_fmt', 'yuv420p', '-r', '30', '-threads', '1',
           '-filter_threads', '1', '-movflags', '+faststart', OUT / name)
    print(f'Encoded {name}', flush=True)


def encode_hook():
    # Instantiate the variable font at explicit weights. FFmpeg's direct font
    # loading otherwise picks Manrope's ultra-light default (200).
    overlay = Image.new('RGBA', (2560, 1440), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, 432, 2560, 950), fill=(0, 0, 0, 155))
    for message, size, weight, y, color in [
        ('ManiMux', 156, 600, 500, 'white'),
        ('Any embodiment × Any policy × Any inference', 44, 500, 770, '#B9D789'),
    ]:
        font = ImageFont.truetype(str(FONT), size)
        font.set_variation_by_axes([weight])
        width = draw.textlength(message, font=font)
        draw.text(((2560-width)/2, y), message, font=font, fill=color)
    overlay.save(OUT / 'title-overlay.png')
    # Fade the band and title together; preserve the original footer disclosure.
    ffmpeg('-ss', 74, '-i', SOURCE, '-threads', '1', '-loop', '1',
           '-i', OUT / 'title-overlay.png', '-t', 11, '-an',
           '-filter_complex', '[1:v]format=rgba,fade=t=in:st=7:d=0.6:alpha=1[brand];[0:v][brand]overlay=shortest=1:format=auto',
           '-filter_complex_threads', '1', '-c:v', 'libx264', '-preset', 'fast',
           '-crf', '21', '-pix_fmt', 'yuv420p', '-r', '30', '-threads', '1',
           '-movflags', '+faststart', OUT / '01-hook.mp4')
    print('Encoded 01-hook.mp4', flush=True)


if __name__ == '__main__':
    encode_hook()
    # Prepare the homepage loop early so layout work can proceed during the recut.
    ffmpeg('-i', OUT / '01-hook.mp4', '-vf', 'scale=1280:720', '-an',
           '-c:v', 'libx264', '-preset', 'fast', '-crf', '24', '-threads', '1',
           '-filter_threads', '1', '-movflags', '+faststart', OUT / 'hero-hook.mp4')
    for timestamp, name in [(0, 'hero-poster.jpg'), (8.5, 'hook-title.jpg')]:
        ffmpeg('-ss', timestamp, '-i', OUT / 'hero-hook.mp4', '-frames:v', '1',
               '-threads', '1', '-q:v', '3', OUT / name)
    print('Homepage assets ready', flush=True)
    if '--hook-only' not in sys.argv:
        encode(0, 74, '02-story.mp4')
        encode(85, 7, '03-close.mp4')
    (OUT / 'concat.txt').write_text("file '01-hook.mp4'\nfile '02-story.mp4'\nfile '03-close.mp4'\n")
    target = OUT / 'manimux-product-demo-hook-1440p30.mp4'
    ffmpeg('-f', 'concat', '-safe', '0', '-i', OUT / 'concat.txt', '-c', 'copy',
           '-movflags', '+faststart', target)
    ffmpeg('-i', target, '-an', '-f', 'null', '-')
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error',
        '-show_entries', 'format=duration:stream=width,height,r_frame_rate,nb_frames',
        '-of', 'json', str(target)]))
    stream = probe['streams'][0]
    assert (stream['width'], stream['height'], stream['r_frame_rate'], stream['nb_frames']) == (2560, 1440, '30/1', '2760'), probe
    assert abs(float(probe['format']['duration']) - 92) < .05, probe
    (OUT / 'validation.json').write_text(json.dumps(probe, indent=2) + '\n')
    for timestamp in [0, 8.5, 11.1, 65.5, 91]:
        ffmpeg('-ss', timestamp, '-i', target, '-frames:v', '1', '-vf', 'scale=1280:720',
               '-filter_threads', '1', '-threads', '1', '-q:v', '3', OUT / f'review-{timestamp}.jpg')
    print(f'Validated {target}', flush=True)
