# Accepted product film

| Seconds | Scene |
| --- | --- |
| 0–10 | Staggered falling blocks in Embodiment, Policy, Inference containers |
| 10–22 | Choices connect to adaptation, scheduling/timeline, execution; then GUI/records |
| 22–29 | Intended fresh-chat deployment request |
| 29–54 | Captured doctor, up, status output with character typing |
| 54–63 | Planned Prepare, Warmup, Start; human review, score, save |
| 63–77 | Existing synthetic RoboGUI orbit, captions fading before pullback |
| 77–85 | Continuous pullback into upper-left cell, staggered 6×6 wall |
| 85–92 | Small-image blur, white wash, ManiMux title and slogan |

Use Manrope at varied weights and JetBrains Mono for code. Current palette:
charcoal #121212, panels #1E1E1E, text #F5F5F5, accent #76B900. The closing
background is white with a faint blurred wall. The video wordmark is white on
charcoal and deep green #28513A on the white closing card; the accent remains
#76B900. Use a small number of brand colors rather than tinting every label.

The closing subtitle uses the exact paper title, broken into two lines:

An open-source harness for real-world
manipulation inference and experiment management

The large ManiMux wordmark above supplies the paper's name. The subtitle omits
the redundant MANIMUX prefix and uses sentence case. Its two lines live in
assets/closing-title.txt, read when the closing scene renders.

Blocks fall with acceleration, slight tilt, soft bounce, then settle into stable
rows. Clip entries to the container interior, not across the headline. Commands
have deterministic uneven per-character delays; output follows line by line.
Scene transitions use short eased dissolves. Preserve clear pauses for reading.

The GUI source was captured from an actual synthetic RoboGUI preview. The wall
currently repeats it, synchronized, to study composition and motion. Do not call
these distinct real executions. With real clips, retain per-clip provenance and
use pre-sized proxies; do not create many live browsers or high-resolution decoders.

Historical CLI stdout in assets/inputs/commands.json includes the original local
paths and timing. Do not silently alter it and continue calling it actual output.
Recapture or explicitly label illustrative commands when changing the interface.

The earlier cuts developed the fresh-chat request, deployment lifecycle, typed
commands, architecture opener, falling integration tiles, human evaluation loop,
and video wall. The saved renderer combines them into one coherent cut. Generated
videos and legacy local capture scripts remain under recordings/agent-demo-v*/.

## Website hook-first recut (2026-10-18)

The original film above remains the source master. The homepage and its full-film
player now use a reordered cut from `scripts/media/recut-product-hook.py`:

| Output seconds | Source seconds | Scene |
| --- | --- | --- |
| 0–11 | 74–85 | RoboGUI → pullback → 36-cell wall; ManiMux title fades in at 7 s |
| 11–85 | 0–74 | Architecture → deployment → human review/score/save → RoboGUI |
| 85–92 | 85–92 | Original white closing card and paper title |

The full recut remains 2560 × 1440, 30 fps, 2760 frames, 92 seconds. Its 11-second
720p proxy plays immediately behind the homepage headline, which fades as the
video's ManiMux title appears. The wall remains repeated synthetic footage,
explicitly disclosed. Do not relabel it as real robot execution.
Use the same throttle/systemd limits with the repository `.venv/bin/python`
(Pillow is required). The script supports `--hook-only` for changing the opening
without re-encoding its existing story/closing segments. Originals are preserved.

## Continuous website hero (2026-10-18)

The homepage now uses `scripts/media/render-hero-loop.py`: a single 720p30
synthetic RoboGUI orbit with a 0.8-second seam dissolve (23.2 seconds total).
Keep its headline persistent, separate from the full film’s wall/title sequence.
Use supported-integration wording; do not imply universal policy/inference
compatibility. The hook-first film remains available in the full-film player.
