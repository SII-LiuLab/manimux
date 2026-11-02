# ManiMux website

English, buildless HTML/CSS/JavaScript with local fonts and pre-rendered video.
No application dependencies, robot connection, analytics, or live GPU rendering.

## Build and preview

From the repository root:

```bash
node --check website/app.js
python3 website/build.py
python3 -m http.server 18087 --bind 127.0.0.1 --directory website/dist
```

Open http://127.0.0.1:18087/. Serve the generated `website/dist/` directory,
which contains both the homepage and guide. The build validates local links,
HTML IDs, guide anchors, and repository references, and removes stale build output.

## Publication

`.github/workflows/docs.yml` builds and deploys GitHub Pages on relevant pushes to
`main`, or on manual dispatch. The public routes are:

- Homepage: https://sii-liulab.github.io/manimux/
- Guide: https://sii-liulab.github.io/manimux/guideline.html

The repository-root `guideline.html` remains the canonical documentation source.
Do not maintain a second copy inside `website/`. The build copies the guide and its
media, and converts links to repository files into GitHub links. Old homepage
anchors such as `#support` and `#install` redirect to the guide. Hashes used by the
new homepage retain their homepage meaning; `#manimux` redirects to guide overview.

The earlier private Sites preview is a separate publication. GitHub Pages is the
public deployment maintained by this workflow; no Sites credentials or identity
are needed to build it.

## Media and interaction

`assets/media.json` records source paths, hashes, trims, and provenance boundaries.

- The hero is a single 23.2-second, 720p30 synthetic RoboGUI orbit with a 0.8-second
  seam dissolve. It does not show physical execution or real model inference.
- The full film is 92 seconds at 2560 × 1440 / 30 fps. It includes repeated
  synthetic wall tiles and illustrative evaluation, and retains its earlier slogan.
- The archived real-robot excerpt is 12 seconds at original speed. Its source
  lacks a verified model/checkpoint/inference manifest.
- Human score/save interactions are illustrative and stay in page memory until reload.
- Integration selectors describe individual components, not compatible combinations.

Only one video plays at a time. Hero playback pauses off-screen, on page hide,
and while the full-film dialog is open. The dialog unloads on close. Reduced
motion and Data Saver prevent initial autoplay. Deployment text types at 20 ms
per character. Fonts are Manrope and JetBrains Mono with included OFL notices.

New contributed footage must show the physical robot, model identity, and actual
inference activity together, with traceable model-generated trajectories. Retain
capture manifests, validate submitted media, and update provenance when replacing
illustrative footage. Do not interpret animation or a connection badge as execution.

## Recutting the opening

Source: `scripts/media/recut-product-hook.py` (Python with Pillow, FFmpeg). Run under
`skills/manimux-demo-video/scripts/throttle.py` with the same one-core,
60% duty cycle, 1536 MiB / zero-swap systemd scope described by the video skill.
The script encodes sequential segments, concatenates them, checks full decoding,
and verifies 2560 × 1440, 30 fps, 2760 frames / 92 seconds. `--hook-only` reuses the
existing story and closing segments when refining only the opening title.

## Continuous hero background

`scripts/media/render-hero-loop.py` produces the background and poster from the
bundled 24-second RoboGUI clip. Run under the same video throttle/systemd limits.
Only the wrap seam is dissolved; the orbit retains its original speed. The full
film is unchanged by this hero update and still contains its earlier wording.
