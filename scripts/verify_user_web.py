"""Browser user-style regenerate check (v11): erase moon/swirl/stroke, run long,
capture wounded/final canvas, measure steady-state flicker. Local only."""
from __future__ import annotations
import functools, http.server, json, threading, time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
OUT = ROOT / "checkpoints" / "user_web_v11"
OUT.mkdir(parents=True, exist_ok=True)
TARGET_STEPS = 6000

class Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass

@contextmanager
def server():
    h = functools.partial(Q, directory=str(WEB))
    s = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h)
    t = threading.Thread(target=s.serve_forever, daemon=True); t.start()
    try: yield f"http://127.0.0.1:{s.server_port}/"
    finally: s.shutdown(); t.join(timeout=3)

# grid coords (512x288) -> normalized (u, v) with v flipped
def norm(x, y): return x / 512.0, 1.0 - y / 288.0
WOUNDS = [(306, 34, 16), (350, 96, 30), (426, 84, 20), (392, 148, 14)]
STROKE = ((250, 60), (330, 120), 8)

def main():
    with server() as url, sync_playwright() as pw:
        b = pw.chromium.launch(headless=True)
        pg = b.new_page(viewport={"width": 1280, "height": 800})
        pg.set_default_timeout(120000)
        pg.goto(url + "?autostart=1&mode=regenerate", wait_until="domcontentloaded")
        pg.wait_for_function("() => window.nca4plot && window.nca4plot.entered && window.nca4plot.steps > 0", timeout=60000)
        pg.wait_for_function("(s) => window.nca4plot.steps > s + 20", arg=pg.evaluate("window.nca4plot.steps"), timeout=30000)
        backend = pg.evaluate("window.nca4plot.backend")
        # UI screenshot (regen-only controls)
        pg.screenshot(path=str(OUT / "ui_regen_only.png"))
        # pause the RAF loop so manual stepping is exclusive
        pg.evaluate("window.nca4plot.pause()")
        pg.wait_for_timeout(200)
        # erase wounds
        for (x, y, r) in WOUNDS:
            u, v = norm(x, y)
            pg.evaluate(f"window.nca4plot.eraseNormalized({u}, {v}, {r})")
        (x0, y0), (x1, y1), r = STROKE
        n = 14
        for i in range(n + 1):
            t = i / n
            u, v = norm(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
            pg.evaluate(f"window.nca4plot.eraseNormalized({u}, {v}, {r})")
        pg.wait_for_timeout(150)
        pg.locator("#sky").screenshot(path=str(OUT / "canvas_wounded.png"))
        steps0 = pg.evaluate("window.nca4plot.steps")
        # run long
        res = pg.evaluate("""async (target) => {
            const e = window.nca4plot.engine;
            const t0 = performance.now();
            while (e.steps < target && performance.now() - t0 < 240000) {
                const ok = await e.run();
                if (!ok) break;
            }
            return {steps: e.steps, sat: e.satFraction, guard: e.guardEvents || 0,
                    ms: Math.round(performance.now() - t0)};
        }""", steps0 + TARGET_STEPS)
        pg.locator("#sky").screenshot(path=str(OUT / "canvas_final.png"))
        # steady-state flicker: two shots 700ms apart
        pg.locator("#sky").screenshot(path=str(OUT / "flicker_a.png"))
        pg.wait_for_timeout(700)
        pg.locator("#sky").screenshot(path=str(OUT / "flicker_b.png"))
        a = np.asarray(Image.open(OUT / "flicker_a.png").convert("RGB"), np.float32)
        bb = np.asarray(Image.open(OUT / "flicker_b.png").convert("RGB"), np.float32)
        flick = float(np.abs(a - bb).mean())
        report = {"backend": backend, "steps0": steps0, "run": res,
                  "flicker_mean_abs": round(flick, 3),
                  "mode": pg.evaluate("window.nca4plot.mode"),
                  "lockWounds": pg.evaluate("window.nca4plot.engine.lockWounds")}
        (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report), flush=True)
        b.close()
    print("WEB VERIFY DONE", OUT, flush=True)

if __name__ == "__main__":
    main()
