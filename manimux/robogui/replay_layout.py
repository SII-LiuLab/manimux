"""Evidence-first layout for the tracking extension; ordinary replay is unchanged."""


def tracking_layout(show_scene: bool) -> str:
    # Viser exposes controls through its scroll panel. Scope all overrides to
    # this extension marker; keep native inputs, image handles and their events.
    return """
<style>
div:is([style*="width: 24em"], [style*="width: 20em"]):has(.mm-replay-layout) {
  width: 100vw !important;
  background: #0c121c;
}
body:has(.mm-replay-layout[data-scene="on"]) div[style*="width: 24em"] {
  width: min(760px, 58vw) !important;
}
div:has(> div > .mm-replay-layout) {
  container: replay / inline-size;
  display: grid; grid-template-columns: repeat(12, minmax(0, 1fr));
  gap: 12px; padding: 12px 20px 24px !important; align-items: start;
}
div:has(> div > .mm-replay-layout) > div {
  min-width: 0; max-width: none !important; box-sizing: border-box;
}
div:has(> div > .mm-replay-layout) > div:first-child {
  grid-column: 1 / -1; padding: 0 !important;
}
div:has(> div > .mm-replay-layout) > div:first-child h2 {
  font-size: 20px; margin: 0;
}
div:has(> div > .mm-replay-layout) > div:first-child p {
  margin: 3px 0 0; font-size: 12px; color: #9dadc2;
}
div:has(> div > .mm-replay-layout) > div:nth-child(2) {
  grid-column: span 2;
}
div:has(> div > .mm-replay-layout) > div:nth-child(3) {
  grid-column: span 2;
}
div:has(> div > .mm-replay-layout) > div:nth-child(4) {
  grid-column: span 5;
}
div:has(> div > .mm-replay-layout) > div:nth-child(5) {
  grid-column: span 3; font-size: 12px;
}
div:has(> div > .mm-replay-layout)
  > div:nth-child(n+2):nth-child(-n+5) {
  position: sticky; top: 0; z-index: 8; background: #0c121c;
  min-height: 44px; padding-top: 8px !important;
}
div:has(> div > .mm-replay-layout) > div:has(> .mm-replay-layout),
div:has(> div > .mm-replay-layout)
  > div:has(> div > .mm-replay-layout) {
  display: none;
}
div:has(> div > .mm-replay-layout) > div:nth-child(7) {
  grid-column: span 3;
}
div:has(> div > .mm-replay-layout) > div:nth-child(8) {
  grid-column: span 6;
}
div:has(> div > .mm-replay-layout) > div:nth-child(9) {
  grid-column: span 3;
}
div:has(> div > .mm-replay-layout) > .mantine-Paper-root {
  grid-column: span 6; margin: 8px 0 0 !important; background: #111b29;
  border: 1px solid #29384c; border-radius: 10px;
}
div:has(> div > .mm-replay-layout)
  > .mantine-Paper-root:has(.mm-replay-wide) { grid-column: 1 / -1; }
.mantine-Paper-root:has(.mm-replay-arm) img {
  height: clamp(160px, 22vh, 240px) !important; width: 100%;
  max-width: none !important; object-fit: contain; background: #080e17;
}
.mm-replay-chart svg { width: 100%; height: 160px; }
.mm-replay-chunks { overflow-x: auto; }
.mm-replay-chunks svg { width: 100%; min-width: 760px; height: 115px; }
.mm-replay-values {
  container: replay / inline-size;
  display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px 20px;
}
.mm-replay-values > div { border-top: 1px solid #26374b; padding-top: 5px; }
.mm-replay-values span { display: block; font-size: 11px; }
.mm-replay-values strong { font-size: 16px; font-weight: 500; color: #e7eef8; }
@media (max-height: 850px) and (min-width: 951px) {
  .mantine-Paper-root:has(.mm-replay-arm) img { height: 160px !important; }
  .mm-replay-chart svg { height: 145px; }
}
body:has(.mm-replay-layout[data-scene="on"]) div:has(> div > .mm-replay-layout)
  > .mantine-Paper-root { grid-column: 1 / -1; }
body:has(.mm-replay-layout) .mantine-Paper-root[style*="position: fixed"]:has(.mm-replay-layout) {
  height: 100dvh !important; max-height: 100% !important; background: #0c121c;
}
body:has(.mm-replay-layout[data-scene="on"])
  .mantine-Paper-root[style*="position: fixed"]:has(.mm-replay-layout) {
  height: 65dvh !important;
}
@container replay (max-width: 950px) {
  div:has(> div > .mm-replay-layout) > .mantine-Paper-root {
    grid-column: 1 / -1;
  }
  div:has(> div > .mm-replay-layout) > div:nth-child(5) {
    grid-column: 1 / -1; position: static;
  }
  div:has(> div > .mm-replay-layout) > div:nth-child(4),
  div:has(> div > .mm-replay-layout) > div:nth-child(7),
  div:has(> div > .mm-replay-layout) > div:nth-child(8),
  div:has(> div > .mm-replay-layout) > div:nth-child(9) { grid-column: 1 / -1; }
  div:has(> div > .mm-replay-layout) > div:nth-child(2),
  div:has(> div > .mm-replay-layout) > div:nth-child(3) { grid-column: span 6; }
  div:has(> div > .mm-replay-layout) > div:nth-child(4) { top: 56px; }
}
@media (max-width: 950px) {
  div:has(> div > .mm-replay-layout) { gap: 8px; padding: 10px !important; }
  body:has(.mm-replay-layout[data-scene="on"]) div[style*="width: 24em"] {
    width: 85vw !important;
  }
}
</style>
<div class="mm-replay-layout" data-scene="SCENE"></div>
""".replace("SCENE", "on" if show_scene else "off")
