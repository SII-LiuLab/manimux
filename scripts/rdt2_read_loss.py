#!/usr/bin/env python3
"""Report whether an RDT2 smoke run actually shows a descent signal.

Reads the TensorBoard event files accelerate writes (rdt/train.py:429 logs
{"loss", "lr"} every optimizer step) and falls back to scraping the tee'd
train.log if the event files are unreadable.

Verdict is deliberately conservative: a short run on 15 episodes is noisy, so
we compare head/tail windows AND a least-squares slope, and only call it a
descent when both agree.
"""
import re
import sys
from pathlib import Path


def from_events(log_dir):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return None
    events = sorted(Path(log_dir).rglob("events.out.tfevents.*"))
    if not events:
        return None
    series = []
    for path in events:
        acc = EventAccumulator(str(path.parent), size_guidance={"scalars": 0})
        acc.Reload()
        if "loss" not in acc.Tags().get("scalars", []):
            continue
        series += [(e.step, e.value) for e in acc.Scalars("loss")]
    if not series:
        return None
    series.sort()
    return [v for _, v in series]


def from_log(log_dir):
    log = Path(log_dir).parent / "train.log"
    if not log.exists():
        return None
    text = log.read_text(errors="replace")
    # tqdm postfix, e.g. "loss=0.481, lr=0.0001"
    vals = [float(m) for m in re.findall(r"loss=([0-9.eE+-]+)", text)]
    return vals or None


def slope(ys):
    n = len(ys)
    mx = (n - 1) / 2
    my = sum(ys) / n
    num = sum((i - mx) * (y - my) for i, y in enumerate(ys))
    den = sum((i - mx) ** 2 for i in range(n))
    return num / den if den else 0.0


def main():
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "logs"
    losses = from_events(log_dir) or from_log(log_dir)
    if not losses:
        print(f"no loss scalars found under {log_dir} (and no train.log fallback)")
        return 1

    n = len(losses)
    print(f"steps logged      : {n}")
    if n < 20:
        print("too few steps to judge a trend; run at least ~50")
        print(f"values: {[round(v, 4) for v in losses]}")
        return 1

    w = max(5, n // 10)
    head, tail = losses[:w], losses[-w:]
    hm, tm = sum(head) / w, sum(tail) / w
    sd = (sum((v - hm) ** 2 for v in head) / w) ** 0.5
    k = slope(losses)

    print(f"first {w:>4} mean   : {hm:.4f}  (sd {sd:.4f})")
    print(f"last  {w:>4} mean   : {tm:.4f}")
    print(f"drop              : {hm - tm:+.4f}  ({100 * (hm - tm) / hm:+.1f}%)")
    print(f"lsq slope/step    : {k:+.3e}  (total {k * n:+.4f} over the run)")
    print(f"min / max         : {min(losses):.4f} / {max(losses):.4f}")

    descending = k < 0 and tm < hm
    # Require the drop to clear head-window noise, otherwise it is not a signal.
    convincing = descending and (hm - tm) > sd
    if convincing:
        print("\nVERDICT: descending, and the drop clears head-window noise.")
    elif descending:
        print(f"\nVERDICT: trending down, but the drop ({hm - tm:.4f}) is within "
              f"head-window noise (sd {sd:.4f}). Run longer or raise the batch size.")
    else:
        print("\nVERDICT: NO descent. Check lr, normalizer, and that "
              "--pretrained_model_name_or_path actually loaded.")
    return 0 if convincing else 2


if __name__ == "__main__":
    sys.exit(main())
