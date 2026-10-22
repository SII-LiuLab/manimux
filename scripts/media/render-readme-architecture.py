#!/usr/bin/env python3
"""Render a lightweight, illustrative README architecture loop (Pillow and NumPy).

Run from the repository root:
    python scripts/media/render-readme-architecture.py --still
    python scripts/media/render-readme-architecture.py

Uses the bundled OFL Manrope font. No browser, network, GPU or hardware access.
The globe and flow particles are a schematic, not measured runtime telemetry.
"""
from __future__ import annotations

import argparse
import math
import io
import json
import struct

import numpy as np
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
W, H, SCALE = 1440, 600, 2
INK = '#20372B'
MUTED = '#657B6B'
GREEN = '#5C996B'
EDGE = '#D7E4DA'
FONT = ROOT / 'website/assets/Manrope.ttf'
FONTS = {}


def font(size, weight=500):
    key = (size, weight)
    if key not in FONTS:
        f = ImageFont.truetype(str(FONT), size * SCALE)
        f.set_variation_by_axes([weight])
        FONTS[key] = f
    return FONTS[key]


class EarthMesh:
    """Small orthographic CPU rasterizer for the bundled textured glTF mesh."""

    def __init__(self):
        data = (ROOT/'assets/architecture/earth-clouds.glb').read_bytes()
        size = struct.unpack_from('<I', data, 12)[0]
        meta = json.loads(data[20:20+size]); binary = data[28+size:]

        def accessor(index):
            item = meta['accessors'][index]
            view = meta['bufferViews'][item['bufferView']]
            count = {'VEC3': 3, 'VEC2': 2, 'SCALAR': 1}[item['type']]
            dtype = {5126: '<f4', 5123: '<u2'}[item['componentType']]
            offset = view.get('byteOffset', 0) + item.get('byteOffset', 0)
            return np.frombuffer(binary, dtype=dtype, count=item['count']*count,
                                 offset=offset).reshape(-1, count).copy()

        primitive = meta['meshes'][0]['primitives'][0]
        self.vertices = accessor(primitive['attributes']['POSITION'])
        self.vertices /= np.linalg.norm(self.vertices, axis=1).max()
        self.normals = accessor(primitive['attributes']['NORMAL'])
        self.uv = accessor(primitive['attributes']['TEXCOORD_0'])
        self.faces = accessor(primitive['indices']).reshape(-1, 3)
        view = meta['bufferViews'][meta['images'][0]['bufferView']]
        start = view.get('byteOffset', 0)
        texture = Image.open(io.BytesIO(binary[start:start+view['byteLength']])).convert('RGB')
        self.texture = np.asarray(texture, dtype=np.float32)

    def render(self, phase, radius=93*SCALE):
        size = radius*2+16; center = (size-1)/2
        yaw = phase*2*math.pi + .4; tilt = -.27
        cy, sy, ct, st = math.cos(yaw), math.sin(yaw), math.cos(tilt), math.sin(tilt)
        rotation = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]]) @ np.array(
            [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        vertices = self.vertices @ rotation.T
        normals = self.normals @ rotation.T
        points = vertices[:, :2]*[radius, -radius]+center
        yy, xx = np.mgrid[:size, :size]
        rr = np.sqrt((xx-center)**2+(yy-center)**2)/radius
        # Thin atmospheric rim outside the mesh. RGB is composited onto a white page.
        glow = (np.exp(-((rr-1.005)/.024)**2)*.20*(rr>.985))
        color = np.empty((size, size, 4), dtype=np.uint8)
        color[:, :, :3] = [115, 173, 205]; color[:, :, 3] = (glow*255).astype(np.uint8)
        depth = np.full((size, size), -np.inf)
        light = np.array([-.45, .55, 1.0]); light /= np.linalg.norm(light)
        th, tw = self.texture.shape[:2]
        for face in self.faces:
            if normals[face, 2].mean() <= -.05:
                continue
            tri = points[face]
            x0,y0 = np.maximum(np.floor(tri.min(axis=0)).astype(int), 0)
            x1,y1 = np.minimum(np.ceil(tri.max(axis=0)).astype(int), size-1)
            if x1<x0 or y1<y0: continue
            x,y = np.meshgrid(np.arange(x0,x1+1)+.5, np.arange(y0,y1+1)+.5)
            a,b,c=tri
            den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
            if abs(den)<1e-7: continue
            w0=((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/den
            w1=((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/den
            w2=1-w0-w1
            weights=np.stack([w0,w1,w2],axis=-1)
            z=weights@vertices[face,2]
            region=depth[y0:y1+1,x0:x1+1]
            mask=(w0>=-1e-6)&(w1>=-1e-6)&(w2>=-1e-6)&(z>region)
            if not mask.any(): continue
            w=weights[mask]; uv=w@self.uv[face]
            tx=np.clip(uv[:,0]*(tw-1),0,tw-1); ty=np.clip(uv[:,1]*(th-1),0,th-1)
            ix,iy=tx.astype(int),ty.astype(int); fx,fy=(tx-ix)[:,None],(ty-iy)[:,None]
            ix1=np.minimum(ix+1,tw-1);iy1=np.minimum(iy+1,th-1)
            tex=(self.texture[iy,ix]*(1-fx)+self.texture[iy,ix1]*fx)*(1-fy)
            tex+=(self.texture[iy1,ix]*(1-fx)+self.texture[iy1,ix1]*fx)*fy
            normal=w@normals[face];normal/=np.linalg.norm(normal,axis=1)[:,None]
            shade=.40+.60*np.clip(normal@light,0,1)
            rim=(1-np.maximum(normal[:,2],0))**3
            rgb=tex*shade[:,None]+rim[:,None]*np.array([15,35,55])
            pixels=color[y0:y1+1,x0:x1+1]
            pixels[mask,:3]=np.clip(rgb,0,255).astype(np.uint8); pixels[mask,3]=255
            region[mask]=z[mask]
        return Image.fromarray(color)


EARTH = None


def render(phase):
    im = Image.new('RGB', (W * SCALE, H * SCALE), '#FFFFFF')
    d = ImageDraw.Draw(im)

    def coords(points):
        return [(round(x * SCALE), round(y * SCALE)) for x, y in points]

    def line(points, color=EDGE, width=2):
        d.line(coords(points), fill=color, width=round(width*SCALE), joint='curve')

    def ellipse(box, fill, outline=None, width=1):
        d.ellipse(tuple(round(v*SCALE) for v in box), fill=fill,
                  outline=outline, width=width*SCALE)

    def text(x, y, value, size=23, weight=500, color=INK, anchor='mm'):
        d.text((round(x*SCALE), round(y*SCALE)), value, font=font(size, weight),
               fill=color, anchor=anchor)

    def box(bounds, fill='#FFFFFF', outline=EDGE, radius=20, width=2):
        d.rounded_rectangle(tuple(round(v*SCALE) for v in bounds), radius=radius*SCALE,
                            fill=fill, outline=outline, width=width*SCALE)

    def path(points, phase_offset=0, color=GREEN, arrows=True, active=True):
        line(points, '#C6D8CA', 2)
        if arrows:
            x, y = points[-1]; px, py = points[-2]
            ang = math.atan2(y-py, x-px)
            poly=[(x,y),(x-9*math.cos(ang-.5),y-9*math.sin(ang-.5)),
                  (x-9*math.cos(ang+.5),y-9*math.sin(ang+.5))]
            d.polygon(coords(poly),fill=color)
        if active:
            lengths=[math.hypot(b[0]-a[0],b[1]-a[1]) for a,b in zip(points,points[1:])]
            distance=((phase*3 + phase_offset)%1)*sum(lengths)
            for (a,b),length in zip(zip(points,points[1:]),lengths):
                if distance<=length:
                    f=distance/length; x=a[0]+f*(b[0]-a[0]); y=a[1]+f*(b[1]-a[1])
                    ellipse((x-7,y-7,x+7,y+7),'#E1EFE4')
                    ellipse((x-3.5,y-3.5,x+3.5,y+3.5),color)
                    break
                distance-=length

    box((1,1,W-2,H-2),'#F9FBF9','#E8EEE9',24,1)
    text(46,48,'A CONTINUOUS FEEDBACK LOOP',19,650,MUTED,'lm')
    text(46,81,'Policies meet the physical world.',21,450,MUTED,'lm')

    # Control interface, runtime, policy service and physical robot station.
    box((45,302,305,476))
    box((460,302,770,476),'#ECF5EE','#75A781',20,2)
    box((460,70,770,195))
    box((990,302,1395,476))

    text(175,340,'RoboGUI',29,700)
    text(175,386,'Run · Record',23,500,MUTED)
    text(175,422,'Review · Evaluate',23,500,MUTED)
    text(615,340,'ManiMux',33,750,'#28513A')
    text(615,388,'Adapt · Schedule',24,550)
    text(615,425,'Execute · Record',24,550)
    text(615,111,'Policy serving',28,700)
    text(615,156,'Local servers · Cloud APIs',21,500,MUTED)
    text(1192,340,'Robot station',29,700)
    text(1192,389,'Arms · Grippers · Sensors',23,500,MUTED)
    text(1192,425,'Camera views · Robot state',21,500,MUTED)

    path([(305,388),(460,388)],.15,arrows=False)
    text(382,362,'Control',19,550,MUTED)
    text(382,415,'Review',19,550,MUTED)
    path([(530,302),(530,195)],.0)
    path([(700,195),(700,302)],.5)
    # Label surfaces ensure the connectors do not run through text.
    box((437,235,624,263),'#F9FBF9',None,5,1)
    box((617,235,790,263),'#F9FBF9',None,5,1)
    text(530,249,'Observations',21,550,MUTED)
    text(704,249,'Action chunks',21,550,MUTED)
    path([(770,355),(990,355)],.33)
    path([(990,425),(770,425)],.8)
    text(880,329,'Commands',22,550,MUTED)
    text(880,452,'Observations',22,550,MUTED)

    # Render the NASA textured mesh, with local lighting and atmospheric rim.
    global EARTH
    if EARTH is None:
        EARTH = EarthMesh()
    cx, cy = 1192, 140
    ellipse((cx-101,cy+91,cx+101,cy+103),'#EFF4F0')
    globe = EARTH.render(phase)
    im.paste(globe, (round(cx*SCALE-globe.width/2), round(cy*SCALE-globe.height/2)), globe)
    text(cx,264,'Interactive world',24,650,'#3C7150')
    path([(1045,302),(1045,140),(1090,140)],.2)
    path([(1294,140),(1340,140),(1340,302)],.65)
    text(1045,279,'Act',18,550,MUTED)
    text(1340,279,'Sense',18,550,MUTED)

    # A restrained footer identifies illustration rather than live telemetry.
    ellipse((46,535,54,543),GREEN)
    text(67,539,'Observe. Infer. Act. Repeat.',21,550,MUTED,'lm')
    text(1392,539,'Earth: NASA / VTAD · Illustrative flow',17,450,MUTED,'rm')
    return im.resize((W,H),Image.Resampling.LANCZOS)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--still',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'assets/architecture')
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    poster=render(0)
    poster.save(args.output/'manimux-architecture.png',optimize=True)
    if args.still: return
    # RGB frames are immediately quantized: only small palette frames accumulate.
    # A common palette prevents flicker; 80 ms/frame gives a 9.6 second loop.
    sample=Image.new('RGB',(W,H*4))
    for j in range(4):
        sample.paste(render(j/4),(0,H*j))
    palette=sample.quantize(colors=256,method=Image.Quantize.MEDIANCUT)
    del sample
    frames=[]
    for i in range(120):
        frames.append(render(i/120).quantize(palette=palette,dither=Image.Dither.NONE))
        if i%20==0: print(f'Rendered {i+1}/120',flush=True)
    frames[0].save(args.output/'manimux-architecture.gif',save_all=True,
                   append_images=frames[1:],duration=80,loop=0,optimize=True,disposal=1)
    print(args.output/'manimux-architecture.gif',flush=True)


if __name__=='__main__':
    main()
