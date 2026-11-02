"""Render the English 30fps wall cut from existing CLI evidence and GUI footage."""
import os
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'threads;1'
import argparse
import bisect
import json
import math
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

cv2.setNumThreads(1)

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / 'assets'
ROOT = Path(os.environ.get('MANIMUX_VIDEO_OUTPUT', str(Path.cwd() / 'recordings/product-demo')))
ROOT.mkdir(parents=True, exist_ok=True)
INPUT = ASSETS / 'inputs'
W, H, FPS = 1920, 1080, 30
OUTPUT_SIZE = (1280, 720)
TOTAL = 92
BG = '#121212'
FG = '#f5f5f5'
MUTED = '#aaaaaa'
MINT = '#76b900'
LINE = '#343434'
FONT = ASSETS / 'fonts/Manrope.ttf'
MONO = ASSETS / 'fonts/JetBrainsMono-Regular.ttf'
COMMANDS = json.loads((INPUT / 'commands.json').read_text())

def ease(x):
    x = max(0., min(1., x))
    return 1 - (1 - x) ** 3

@lru_cache(maxsize=80)
def font(size, weight=500, mono=False):
    result = ImageFont.truetype(str(MONO if mono else FONT), size)
    if not mono:
        result.set_variation_by_axes([weight])
    return result

@lru_cache(maxsize=256)
def text_image(value, size, color=FG, weight=500, mono=False):
    face = font(size, weight, mono)
    box = face.getbbox(value, anchor='lt')
    width = max(1, math.ceil(face.getlength(value)) + 5)
    im = Image.new('RGBA', (width, max(1, box[3] - box[1] + 10)))
    ImageDraw.Draw(im).text((0, 0), value, font=face, fill=color, anchor='lt')
    return im

def paste(im, layer, xy, alpha=1.):
    if alpha <= 0:
        return
    if alpha < 1:
        layer = layer.copy()
        layer.putalpha(layer.getchannel('A').point(lambda p: int(p * alpha)))
    im.paste(layer, (int(xy[0]), int(xy[1])), layer)

def text(im, xy, value, size=32, color=FG, weight=500, mono=False, t=100, delay=0, rise=20):
    a = ease((t - delay) / .65)
    paste(im, text_image(value, size, color, weight, mono), (xy[0], xy[1] + rise * (1-a)), a)

def tracked(im, xy, value, size=21, tracking=3, color=MUTED, t=100, delay=0):
    x = xy[0]
    for ch in value:
        text(im, (x, xy[1]), ch, size, color, 650, t=t, delay=delay, rise=10)
        x += font(size, 650).getlength(ch) + tracking

def panel(im, box, fill='#1e1e1e', outline=LINE, radius=12, t=100, delay=0):
    a = ease((t-delay)/.65)
    w, h = box[2]-box[0], box[3]-box[1]
    layer = Image.new('RGBA', (w+2, h+2))
    ImageDraw.Draw(layer).rounded_rectangle((0, 0, w, h), radius, fill=fill, outline=outline, width=1)
    paste(im, layer, (box[0], box[1]+22*(1-a)), a)

# A very restrained stationary glow keeps type legible while the content moves.
yy, xx = np.mgrid[0:H, 0:W]
glow = np.exp(-(((xx-1620)/680)**2 + ((yy-260)/650)**2))
pixels = np.zeros((H, W, 3), dtype=np.uint8)
for c, (base_value, gain) in enumerate([(18, 1), (18, 3), (18, 0)]):
    pixels[:, :, c] = base_value + glow * gain
BACKGROUND = Image.fromarray(pixels)
del yy, xx, glow, pixels

def base(chapter, footer):
    im = BACKGROUND.copy()
    d = ImageDraw.Draw(im)
    d.line((96, 112, 1824, 112), fill=LINE)
    text(im, (96, 53), 'manimux', 34, FG, 750)
    tracked(im, (1400, 62), 'PRODUCT PREVIEW / V4', 17, 2)
    text(im, (96, 984), footer, 21, MUTED)
    text(im, (1690, 982), f'{chapter:02d} / 07', 23, MUTED, mono=True)
    return im

def title(t):
    im=base(1, 'Illustrative integrations · Some hardware is experimental; combinations require compatible adapters')
    text(im,(96,166),'Any embodiment × Any policy × Any inference',60,FG,650,t=t,delay=.1)
    text(im,(100,259),'Build your stack. Keep one workflow.',36,MUTED,t=t,delay=.3)
    groups=[('EMBODIMENT',['YAM','Tianji','ARX X5','PiPER','Custom']),
            ('POLICY',['Pi05','DP','ABC-DiT','GR00T N1.7','XR-1','LingBot-VLA2','OpenWAM','SAPolicy']),
            ('INFERENCE',['Serial','Async','RTC','ACT','AAC','PAINT','BID','AutoHorizon'])]
    for col,(label,items) in enumerate(groups):
        x=96+col*590
        panel(im,(x,350,x+520,888),fill='#1e1e1e',t=t,delay=.4+col*.12)
        # Clip falling tiles to the container body so headlines remain legible.
        layer=Image.new('RGBA',(520,442))
        for i,item in enumerate(items):
            delay=.8+col*.18+i*.28
            age=t-delay
            if age<0:continue
            target_y=338-(i//2)*96
            px=22+(i%2)*248
            duration=1.1
            if age<duration:
                u=age/duration
                py=-160+(target_y+160)*u*u
                tilt=(-1 if (i+col)%2 else 1)*5*(1-u)
            else:
                elapsed=age-duration
                py=target_y-10*math.exp(-5*elapsed)*abs(math.sin(12*elapsed))
                tilt=0
            tile=Image.new('RGBA',(228,80))
            td=ImageDraw.Draw(tile)
            landing=max(0,1-abs(age-duration)/.25)
            td.rounded_rectangle((0,0,227,79),10,fill='#262b1e',outline=MINT if landing>.25 else '#475339',width=2)
            td.rectangle((15,20,19,59),fill=MINT)
            text(tile,(32,26),item,25,FG,600)
            if item in ('ARX X5','PiPER'):
                text(tile,(32,59),'experimental',12,MUTED)
            if tilt:
                tile=tile.rotate(tilt,resample=Image.Resampling.BICUBIC,expand=True)
            layer.alpha_composite(tile,(int(px),int(py)))
        paste(im,layer,(x,434))
        tracked(im,(x+28,380),label,22,2,MINT,t,.6+col*.12)
        count=sum(t>(.8+col*.18+i*.28+1.1) for i in range(len(items)))
        text(im,(x+446,380),f'{count:02d}',22,MUTED,mono=True,t=t,delay=.6+col*.12)
        if col<2:text(im,(x+539,596),'×',49,MINT,t=t,delay=1.2+col*.2)
    text(im,(100,923),'One extensible harness. Many possible stacks.',30,FG,550,t=t,delay=5)
    return im

def wire(im, points, t, delay, duration=.8):
    if t<delay:return
    d=ImageDraw.Draw(im)
    progress=ease((t-delay)/duration)
    lengths=[math.dist(a,b) for a,b in zip(points,points[1:])]
    total=sum(lengths);remaining=total*progress
    for (a,b),length in zip(zip(points,points[1:]),lengths):
        if remaining<=0:break
        f=min(1,remaining/length)
        d.line((a,(a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f)),fill='#4e6921',width=3)
        remaining-=length
    if progress==1:
        remaining=((t-delay-duration)*.38%1)*total
        for (a,b),length in zip(zip(points,points[1:]),lengths):
            if remaining<=length:
                f=remaining/length;x=a[0]+(b[0]-a[0])*f;y=a[1]+(b[1]-a[1])*f
                d.ellipse((x-5,y-5,x+5,y+5),fill=MINT);break
            remaining-=length

def architecture(t):
    im=base(2, 'Extensible architecture · Supported combinations depend on adapters and backend capabilities')
    text(im,(96,168),'Any embodiment × Any policy × Any inference',60,FG,650,t=t,delay=.1)
    text(im,(100,254),'Independent choices. One deployment and experiment workflow.',32,MUTED,t=t,delay=.3)
    labels=[('EMBODIMENT','Robot + sensor interfaces','YAM · Tianji · custom adapters'),
            ('POLICY','Model clients + serving','XPolicyLab · StarVLA · custom'),
            ('INFERENCE STRATEGY','Timing + chunk handoff','Serial · Async · RTC · PAINT')]
    for i,(label,desc,examples) in enumerate(labels):
        x=96+i*590
        panel(im,(x,335,x+520,510),t=t,delay=.7+i*.2)
        tracked(im,(x+28,363),label,18,2,MINT,t,.9+i*.2)
        text(im,(x+28,408),desc,29,FG,600,t=t,delay=1.05+i*.2)
        text(im,(x+28,464),examples,23,MUTED,t=t,delay=1.2+i*.2)
        wire(im,[(x+260,510),(x+260,574)],t,2.1+i*.15)
    panel(im,(96,574,1796,744),fill='#202719',outline='#50652f',t=t,delay=2.8)
    text(im,(126,601),'MANIMUX',31,MINT,750,t=t,delay=3.05)
    tracked(im,(1400,611),'RUNTIME HARNESS',16,2,MUTED,t,3.15)
    for i,label in enumerate(['Adapt observations / actions','Schedule + build the timeline','Execute robot commands']):
        x=126+i*565
        text(im,(x,670),label,28,FG,550,t=t,delay=3.25+i*.18)
        if i<2:text(im,(x+500,668),'→',30,MINT,t=t,delay=3.55+i*.18)
    wire(im,[(495,744),(495,798)],t,4.2)
    wire(im,[(1385,744),(1385,798)],t,4.4)
    panel(im,(96,798,886,925),t=t,delay=4.8)
    panel(im,(1006,798,1796,925),t=t,delay=5.05)
    text(im,(126,823),'RoboGUI',31,FG,650,t=t,delay=5.1)
    text(im,(126,874),'Visualize state · operator control',25,MUTED,t=t,delay=5.25)
    text(im,(1036,823),'Experiment records',31,FG,650,t=t,delay=5.35)
    text(im,(1036,874),'Record · replay · human evaluation',25,MUTED,t=t,delay=5.5)
    return im

def type_schedule(value, duration):
    # Deterministic variation; spaces and token boundaries carry small pauses.
    gaps = [1 + .22*math.sin(i*2.7) + (.9 if ch == ' ' else .25 if ch in './-' else 0) for i, ch in enumerate(value)]
    total = sum(gaps)
    times=[]
    acc=0
    for gap in gaps:
        acc += gap
        times.append(acc/total*duration)
    return times

def typed(im, xy, value, elapsed, duration, size=29, mono=True, color=MINT, cursor=True):
    count = bisect.bisect_right(type_schedule(value, duration), max(-.001, elapsed))
    visible = value[:count]
    lines = visible.split('\n')
    for j, line in enumerate(lines):
        text(im, (xy[0], xy[1]+j*(size+15)), line, size, color, 500, mono)
    if cursor and elapsed >= 0 and (count<len(value) or int((elapsed-duration)*2.5)%2==0):
        line = lines[-1]
        x = xy[0]+font(size, 500, mono).getlength(line)+3
        y = xy[1]+(len(lines)-1)*(size+15)
        ImageDraw.Draw(im).rounded_rectangle((x, y-1, x+2, y+size+3), 1, fill=color)
    return count == len(value)

def request(t):
    im = base(3, 'Target interaction · This preview uses --demo, without model weights or hardware')
    text(im, (96, 184), 'One request. One connected workflow.', 61, FG, 650, t=t, delay=.1)
    panel(im, (96, 320, 1824, 642), t=t, delay=.25)
    tracked(im, (140, 361), 'THE INTENDED REQUEST', 19, 3, MINT, t, .45)
    if t>.7:
        typed(im, (140, 430), 'Deploy my model. Open RoboGUI.', t-.7, 1.9, 60, False, FG)
    text(im, (140, 558), 'Resolve the recipe. Start the stack. Hand over to the operator.', 30, MUTED, t=t, delay=2.85)
    labels=['Preflight checks', 'Managed services', 'A visible workspace']
    d=ImageDraw.Draw(im)
    for i, label in enumerate(labels):
        x=100+i*580
        text(im, (x, 768), f'0{i+1}', 25, MINT, mono=True, t=t, delay=3.1+i*.2)
        text(im, (x, 822), label, 38, FG, 600, t=t, delay=3.1+i*.2)
        if t>3.6+i*.2 and i<2:
            d.line((x+410, 842, x+410+105*ease((t-3.6-i*.2)/.6), 842), fill='#4c660f', width=2)
    return im

def command_display(index):
    args=COMMANDS[index]['argv']
    split=args.index('--deployment-dir')
    return ' '.join(args[:split])+' \\\n  '+' '.join(args[split:])

def terminal(index, t):
    im = base(4, 'Actual output, retypeset · Typing and output timing are editorial')
    titles=['Check before launch.', 'Launch an isolated preview.', 'Confirm the service state.']
    text(im, (96, 176), titles[index], 66, FG, 650, t=t, delay=.25)
    tracked(im, (1480, 203), 'ACTUAL CLI OUTPUT', 17, 2, MINT, t, .2)
    panel(im, (96, 296, 1824, 936), fill='#151515', radius=12)
    d=ImageDraw.Draw(im)
    for i,c in enumerate(['#dc7b7c','#d6b777','#73bf9e']):
        d.ellipse((126+i*28, 323, 137+i*28, 334), fill=c)
    text(im, (244, 318), 'manimux / local station', 21, MUTED, mono=True)
    for i,label in enumerate(['check', 'launch', 'status']):
        x=1410+i*125
        text(im, (x, 320), label, 19, MINT if i==index else '#666666', mono=True)
        if i==index:
            d.line((x, 350, x+68, 350), fill=MINT, width=2)
    value=command_display(index)
    duration=[3.9,3.7,2.8][index]
    ready=.55+duration+.4
    text(im, (126, 388), '$', 28, MINT, mono=True)
    typed(im, (161, 388), value, t-.55, duration, 28, cursor=t<ready)
    output=COMMANDS[index]['stdout'].splitlines()
    y=518
    for n,line in enumerate(output):
        # The displayed output text is preserved exactly from commands.json.
        text(im, (128, y), line, 21, FG if n==0 else MUTED, mono=True, t=t, delay=ready+n*.10, rise=7)
        y+=29
    if t>ready+len(output)*.1+.6 and int(t*2.5)%2==0:
        d.rectangle((128, y+12, 139, y+34), fill=MINT)
    if index>0:
        text(im, (128, 839), f'Exit code: {COMMANDS[index]["returncode"]}  ·  Captured wall time: {COMMANDS[index]["duration_s"]:.2f}s', 21, MINT, mono=True, t=t, delay=ready+len(output)*.1+.2)
    return im

cap = cv2.VideoCapture(str(INPUT/'robogui-live.mp4'), cv2.CAP_FFMPEG,
                       [cv2.CAP_PROP_N_THREADS, 1])
if not cap.isOpened():
    raise RuntimeError('Actual GUI recording is missing')

def source_frame(t):
    idx=round(t*cap.get(cv2.CAP_PROP_FPS))
    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame=cap.read()
    if not ok:
        raise RuntimeError('GUI source is too short')
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def gui(t):
    im=source_frame(t).resize((W,H))
    clean=im.copy() if t>12 else None
    panel(im, (536, 24, 1498, 173), fill=BG, outline=BG, t=t, delay=.2)
    text(im, (573, 46), 'RoboGUI is open.', 43, MINT, 650, t=t, delay=.4)
    text(im, (573, 113), 'Actual capture · Synthetic preview · No hardware', 27, FG, t=t, delay=.55)
    panel(im, (530, 952, 1514, 1048), fill=BG, outline=BG, t=t, delay=.65)
    caption='Inspect robot geometry and action chunks.' if t<10 else 'Cameras and run controls await real deployment.'
    text(im, (566, 982), caption, 31, FG, t=t if t<10 else t-10+.65, delay=.8 if t<10 else .3, rise=10)
    if clean is not None:
        im=Image.blend(im,clean,smooth((t-12)/2))
    return im

def smooth(x):
    x=max(0.,min(1.,x))
    return x*x*(3-2*x)

def showcase(t):
    """One continuous camera pullback, 36 repeated clips, then a white title."""
    if t < 14:
        return gui(t)
    age=t-14
    # One decoder and one thumbnail resize per frame, shared by all 36 cells.
    # After the pullback, replay the moving part of the recorded orbit.
    source_t=t if t<18 else 2+(t-18)
    clip=source_frame(source_t)
    im=BACKGROUND.copy()
    top,gap=28,10
    th=(H-110-5*gap)//6
    tw=round(th*16/9)
    left=(W-(6*tw+5*gap))//2
    thumb=clip.resize((tw,th),Image.Resampling.BILINEAR)
    p=smooth(age/3.2)
    if age>=1.5:
        for row in range(6):
            for col in range(6):
                if row==0 and col==0:continue
                delay=1.5+(row+col)*.18
                a=smooth((age-delay)/.8)
                if a<=0:continue
                x=left+col*(tw+gap)
                y=top+row*(th+gap)+int(24*(1-a))
                tile=thumb if a==1 else Image.blend(Image.new('RGB',thumb.size,BG),thumb,a)
                im.paste(tile,(x,y))
    # Anchor tile stays on top so the full-screen view really travels to top left.
    width=round(W+(tw-W)*p)
    height=round(H+(th-H)*p)
    anchor=clip.resize((width,height),Image.Resampling.BILINEAR)
    im.paste(anchor,(round(left*p),round(top*p)))
    text(im,(32,1046),'MONTAGE STUDY · Repeated synthetic RoboGUI footage · No hardware execution',20,'#c9c9c9',t=age,delay=3.4,rise=0)
    # Blur a small intermediate image to keep the white wash inexpensive.
    wash=smooth((age-8)/2.2)
    if wash>0:
        small=im.resize((480,270),Image.Resampling.BILINEAR)
        blurred=small.filter(ImageFilter.GaussianBlur(1+wash*5)).resize((W,H),Image.Resampling.BILINEAR)
        im=Image.blend(im,blurred,wash)
        im=Image.blend(im,Image.new('RGB',(W,H),'#ffffff'),wash*.94)
    if age>9.6:
        def centered(value,y,size,color,weight=500,delay=0):
            tile=text_image(value,size,color,weight)
            text(im,((W-tile.width)/2,y),value,size,color,weight,t=age,delay=delay,rise=20)
        centered('ManiMux',352,148,'#28513a',750,9.6)
        centered('Any embodiment × Any policy × Any inference',558,42,'#343434',550,10.1)
        a=ease((age-10.3)/.8)
        ImageDraw.Draw(im).rectangle((W//2-int(90*a),655,W//2+int(90*a),660),fill='#76b900')
        # Read the tiny title file here so editorial corrections made during the
        # earlier scenes can reach the closing card without restarting the export.
        title_lines=(ASSETS/'closing-title.txt').read_text().strip().splitlines()
        for i,line in enumerate(title_lines):
            centered(line,710+i*56,36,'#5d5d5d',550,10.5+i*.15)
        centered('Concept preview · Repeated synthetic footage',1006,20,'#777777',500,10.7)
    return im

def workflow(t):
    im=base(6, 'Planned hardware and evaluation sequence · Not performed in this video')
    tracked(im, (100, 182), 'NEXT: THE COMPLETE LOOP', 20, 3, MINT, t, .1)
    text(im, (96, 262), 'The operator takes it from here.', 76, FG, 650, t=t, delay=.25)
    text(im, (100, 387), 'Run the task. Review the result. Save the evaluation.', 35, MUTED, t=t, delay=.45)
    for i,(label, sub) in enumerate([('Prepare', 'Prepare the run'), ('Warmup', 'Warm up the policy'), ('Start', 'Begin execution')]):
        x=96+i*590
        panel(im, (x, 488, x+520, 690), t=t, delay=.6+i*.22)
        text(im, (x+30, 520), f'0{i+1}', 22, MINT, mono=True, t=t, delay=.8+i*.22)
        text(im, (x+30, 562), label, 49, FG, 650, t=t, delay=.85+i*.22)
        text(im, (x+30, 641), sub, 26, MUTED, t=t, delay=1+i*.22)
    panel(im, (96, 744, 1796, 922), fill='#1a2013', outline='#435626', t=t, delay=2.1)
    tracked(im, (128, 774), 'AFTER THE TASK', 17, 2, MINT, t, 2.35)
    for i, label in enumerate(['Human review', 'Score', 'Save evaluation']):
        x=[128, 765, 1245][i]
        text(im, (x, 826), label, 39, FG, 600, t=t, delay=2.5+i*.4)
        if i<2:
            text(im, ([625,1094][i], 828), '→', 38, MINT, t=t, delay=2.8+i*.4)
    return im

def ending(t):
    im=base(7, 'Not tested here · Model inference, robot connectivity, physical task success or saved evaluation')
    text(im, (96, 250), 'manimux', 147, MINT, 700, t=t, delay=.1, rise=32)
    text(im, (100, 464), 'Deployment you can see.', 68, FG, 650, t=t, delay=.4)
    tracked(im, (103, 642), 'DEMONSTRATED IN THIS CUT', 19, 3, MUTED, t, .75)
    text(im, (100, 700), 'Preflight  →  Demo launch  →  RoboGUI', 43, FG, 550, t=t, delay=.95)
    text(im, (103, 841), 'SYNTHETIC PREVIEW / NO HARDWARE CONNECTED', 25, MINT, 600, t=t, delay=1.3)
    return im

SCENES=[(0,10,title),(10,22,architecture),(22,29,request),(29,39,lambda t:terminal(0,t)),
        (39,47,lambda t:terminal(1,t)),(47,54,lambda t:terminal(2,t)),
        (54,63,workflow),(63,92,showcase)]

def scene_at(t):
    return next((i,start,end,fn) for i,(start,end,fn) in enumerate(SCENES) if start<=t<end)

def main():
    global OUTPUT_SIZE
    parser=argparse.ArgumentParser()
    parser.add_argument('--stills',action='store_true')
    parser.add_argument('--start',type=float,default=0)
    parser.add_argument('--end',type=float,default=TOTAL)
    parser.add_argument('--output',default='manimux-product-demo-en-v4-30fps.mp4')
    parser.add_argument('--resolution',choices=['720p','1080p','1440p'],default='720p')
    args=parser.parse_args()
    OUTPUT_SIZE={'720p':(1280,720),'1080p':(1920,1080),'1440p':(2560,1440)}[args.resolution]
    if args.stills:
        for t in [3,8,19,37,58,70,78,79,81,84,86,90]:
            _,start,_,fn=scene_at(t)
            fn(t-start).resize(OUTPUT_SIZE).save(ROOT/f'review-{t:02d}.jpg', quality=92)
        print('STILLS_COMPLETE')
        return
    encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
                              '-s',f'{OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-threads','1',
                              '-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart','-y',
                              str(ROOT/args.output)],stdin=subprocess.PIPE)
    current=-1
    last=None
    outgoing=None
    for frame in range(round(args.start*FPS),round(args.end*FPS)):
        t=frame/FPS
        index,start,end,fn=scene_at(t)
        if index!=current:
            outgoing=last
            current=index
        im=fn(t-start)
        # Cross-dissolve from the preceding shot, rather than cutting through black.
        transition=.4 if index in (4,5) else .65
        if t-start<transition:
            a=ease((t-start)/transition)
            im=Image.blend(outgoing if outgoing is not None else BACKGROUND, im, a)
        # Finish on the white title card; no closing cut back to charcoal.
        encoder.stdin.write(im.resize(OUTPUT_SIZE,Image.Resampling.BILINEAR).tobytes())
        last=im
        if frame%150==0:
            print(f'Rendered {int(t)}/{TOTAL}s',flush=True)
    encoder.stdin.close()
    if encoder.wait()!=0:
        raise RuntimeError('Encoding failed')
    cap.release()
    print('RENDER_COMPLETE',flush=True)

if __name__=='__main__':
    main()
