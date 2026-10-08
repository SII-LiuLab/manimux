#!/usr/bin/env bash
# Record the policy robogui with ffmpeg/x11grab.
#
# The robogui's own Record button can only capture the 3D scene -- viser renders
# in the browser and `get_render` returns the scene, not the window.  Screen
# capture is what gets the GUI column, the task prompt, and the camera panels
# into the clip, so that is what this script does.
#
#   scripts/media/record-robogui.sh                 # 2K 60 fps, click the window to record
#   scripts/media/record-robogui.sh --screen        # whole screen instead of one window
#   scripts/media/record-robogui.sh --duration 15   # stop automatically after 15 s
#   scripts/media/record-robogui.sh --gif           # also write a README-ready GIF
#   scripts/media/record-robogui.sh --delay 5       # 5 s to raise the window first
#
# Stop a running capture with a single Ctrl-C; ffmpeg finalizes the file.

set -euo pipefail

FPS=${FPS:-60}
SIZE=${SIZE:-2560x1440}
OUT_DIR=${OUT_DIR:-recordings}
CRF=${CRF:-18}
PRESET=${PRESET:-veryfast}
GIF_WIDTH=${GIF_WIDTH:-960}
GIF_FPS=${GIF_FPS:-15}
# Seconds between selecting the window and the first frame, so the window can
# be raised and the mouse moved out of shot.
DELAY=${DELAY:-3}

mode=window
duration=""
make_gif=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --screen) mode=screen; shift ;;
    --window) mode=window; shift ;;
    --region) mode=region; region=$2; shift 2 ;;
    --duration) duration=$2; shift 2 ;;
    --fps) FPS=$2; shift 2 ;;
    --size) SIZE=$2; shift 2 ;;
    --gif) make_gif=1; shift ;;
    --delay) DELAY=$2; shift 2 ;;
    -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

command -v ffmpeg >/dev/null || { echo "ffmpeg is not installed" >&2; exit 1; }
: "${DISPLAY:=:0}"
export DISPLAY

case "$mode" in
  window)
    command -v xwininfo >/dev/null || { echo "xwininfo is not installed" >&2; exit 1; }
    echo "Click the browser window showing the robogui..."
    info=$(xwininfo)
    x=$(awk '/Absolute upper-left X/ {print $NF}' <<<"$info")
    y=$(awk '/Absolute upper-left Y/ {print $NF}' <<<"$info")
    w=$(awk '/Width:/ {print $NF}' <<<"$info")
    h=$(awk '/Height:/ {print $NF}' <<<"$info")
    # Naming the window is the only guard against recording the terminal that
    # launched this script, which is the easy mistake to make here.
    title=$(sed -n 's/^xwininfo: Window id: [^ ]* //p' <<<"$info")
    echo "Selected window: ${title:-<unnamed>}"
    ;;
  screen)
    command -v xdpyinfo >/dev/null || { echo "xdpyinfo is not installed" >&2; exit 1; }
    geometry=$(xdpyinfo | awk '/dimensions:/ {print $2; exit}')
    w=${geometry%x*}
    h=${geometry#*x}
    x=0
    y=0
    ;;
  region)
    # WxH+X+Y
    w=${region%%x*}; rest=${region#*x}
    h=${rest%%+*}; rest=${rest#*+}
    x=${rest%%+*}; y=${rest#*+}
    ;;
esac

# x264 needs even dimensions; an odd window width otherwise fails at encode time.
w=$((w / 2 * 2))
h=$((h / 2 * 2))

mkdir -p "$OUT_DIR"
stamp=$(date +%Y%m%d-%H%M%S)
mp4="$OUT_DIR/robogui-$stamp.mp4"

args=(-hide_banner -loglevel warning -stats
      -f x11grab -framerate "$FPS" -video_size "${w}x${h}" -i "${DISPLAY}+${x},${y}")
[[ -n $duration ]] && args+=(-t "$duration")
# Downscale to the target size only when the capture is larger, so a 1080p
# screen is never blown up to 2K.
args+=(-vf "scale='min(${SIZE%x*},iw)':-2:flags=lanczos"
       -c:v libx264 -preset "$PRESET" -crf "$CRF" -pix_fmt yuv420p
       -movflags +faststart "$mp4")

if [[ ${DELAY:-0} -gt 0 ]]; then
  echo "Raise the window you want; recording starts in ${DELAY}s..."
  for ((i = DELAY; i > 0; i--)); do printf '\r  %d ' "$i"; sleep 1; done
  printf '\r      \r'
fi

echo "Recording ${w}x${h} at ${FPS} fps -> $mp4  (Ctrl-C to stop)"
ffmpeg "${args[@]}"
echo "wrote $mp4 ($(du -h "$mp4" | cut -f1))"

if [[ $make_gif -eq 1 ]]; then
  gif="${mp4%.mp4}.gif"
  palette=$(mktemp --suffix=.png)
  trap 'rm -f "$palette"' EXIT
  filters="fps=$GIF_FPS,scale=$GIF_WIDTH:-1:flags=lanczos"
  ffmpeg -hide_banner -loglevel warning -i "$mp4" -vf "$filters,palettegen=stats_mode=diff" -y "$palette"
  ffmpeg -hide_banner -loglevel warning -i "$mp4" -i "$palette" \
    -lavfi "$filters,paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" -y "$gif"
  echo "wrote $gif ($(du -h "$gif" | cut -f1))"
fi
