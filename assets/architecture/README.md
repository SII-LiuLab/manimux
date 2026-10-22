# README architecture animation

A schematic of RoboGUI, the ManiMux runtime, policy serving and a robot station
interacting with a changing physical world. The rotating sphere and moving dots
are illustrative, not live telemetry or a measured timing diagram.

- `manimux-architecture.gif`: 1440 × 540, 120 frames, 9.6-second seamless loop.
- `manimux-architecture.png`: static alternative linked from the project README.
- Generator: [`render-readme-architecture.py`](../../scripts/media/render-readme-architecture.py).

Render with Python, Pillow and NumPy from the repository root:

```bash
python scripts/media/render-readme-architecture.py --still
python scripts/media/render-readme-architecture.py
```

The renderer rasterizes the textured 3D mesh with NumPy and Pillow on the CPU, retains only quantized frames, and never
accesses browsers or robot hardware. On a busy workstation, use a single CPU
and the existing resource-limited media launcher. The font is the bundled
[Manrope](../../website/assets/Manrope.ttf), under its
[OFL license](../../website/assets/Manrope-OFL.txt).

## Earth model provenance

The globe uses the mesh and UV coordinates from NASA's
[Earth with Clouds 3D Model](https://science.nasa.gov/resource/earth-with-clouds-3d-model/),
credited to NASA Visualization Technology Applications and Development (VTAD).
The compact `earth-clouds.glb` retains the original geometry, reduces the color
texture to 2048 × 1536 JPEG, removes the normal map and uses a non-metallic material.
The diagram adds local illumination and a thin atmospheric rim. Source and
output SHA-256 hashes are recorded in `earth-source.json`.

NASA media usage terms apply to the model and derived imagery; they are not
relicensed under the repository's MIT license. See
[NASA's media usage guidelines](https://www.nasa.gov/nasa-brand-center/images-and-media/).
Use here does not imply NASA endorsement.

To reproduce the compact model, download the source URL in `earth-source.json`,
then run:

```bash
python scripts/media/prepare-readme-earth.py /path/to/EarthClouds_1_12756.glb
```
