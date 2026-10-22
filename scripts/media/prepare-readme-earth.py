#!/usr/bin/env python3
"""Compact NASA's Earth-with-clouds GLB for the README. No network access.

Source and usage terms: assets/architecture/README.md.
Usage: python scripts/media/prepare-readme-earth.py /path/to/EarthClouds_1_12756.glb
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import struct

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    source = args.source.read_bytes()
    n = struct.unpack_from('<I', source, 12)[0]
    model = json.loads(source[20:20+n])
    binary = source[28+n:]
    views, chunks, offset = [], [], 0

    def add(raw, target=None):
        nonlocal offset
        view = {'buffer': 0, 'byteOffset': offset, 'byteLength': len(raw)}
        if target:
            view['target'] = target
        views.append(view)
        raw += b'\x00' * (-len(raw) % 4)
        chunks.append(raw)
        offset += len(raw)

    for v in model['bufferViews'][:4]:
        start = v.get('byteOffset', 0)
        add(binary[start:start+v['byteLength']], v.get('target'))
    v = model['bufferViews'][5]
    start = v.get('byteOffset', 0)
    texture = Image.open(io.BytesIO(binary[start:start+v['byteLength']])).convert('RGB')
    texture.thumbnail((2048, 1536))
    out = io.BytesIO(); texture.save(out, format='JPEG', quality=90)
    add(out.getvalue())
    model['bufferViews'] = views
    model['images'] = [{'bufferView': 4, 'mimeType': 'image/jpeg', 'name': 'Earth clouds — NASA VTAD'}]
    model['textures'] = [{'source': 0, 'sampler': 0}]
    model['materials'] = [{'name': 'Earth', 'pbrMetallicRoughness': {
        'baseColorTexture': {'index': 0}, 'metallicFactor': 0, 'roughnessFactor': .9}}]
    model['buffers'] = [{'byteLength': offset}]
    model['asset']['copyright'] = 'Source: NASA Visualization Technology Applications and Development (VTAD). See README for NASA media usage terms.'
    blob = b''.join(chunks)
    meta = json.dumps(model, ensure_ascii=True, separators=(',', ':')).encode()
    meta += b' ' * (-len(meta) % 4)
    glb = struct.pack('<III', 0x46546C67, 2, 12+8+len(meta)+8+len(blob))
    glb += struct.pack('<II', len(meta), 0x4E4F534A) + meta
    glb += struct.pack('<II', len(blob), 0x004E4942) + blob
    dest = ROOT/'assets/architecture/earth-clouds.glb'; dest.write_bytes(glb)
    provenance = {
        'source_page': 'https://science.nasa.gov/resource/earth-with-clouds-3d-model/',
        'download': 'https://assets.science.nasa.gov/content/dam/science/psd/solar/2023/09/e/EarthClouds_1_12756.glb',
        'source_sha256': hashlib.sha256(source).hexdigest(),
        'derived_sha256': hashlib.sha256(glb).hexdigest(),
        'changes': 'Original mesh and UVs preserved. Base-color texture reduced to 2048x1536 JPEG; normal map removed; neutral non-metallic material.'
    }
    (dest.parent/'earth-source.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(dest, len(glb))


if __name__ == '__main__':
    main()
