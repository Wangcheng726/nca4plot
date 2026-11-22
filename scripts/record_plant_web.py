"""Record a short web-demo video for the Pre (run after training, GPU serial).

The app's main loop is paced at 40 CA steps/s, so a free-running recording
shows every stage in real time: growth -> small damage -> regrowth ->
big damage -> regrowth -> manual mouse erase -> regrowth (~30 s).
"""
from __future__ import annotations

import functools
import http.server
import shutil
import threading
from contextlib import contextmanager
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
OUT = ROOT / "media" / "plant_web"
OUT.mkdir(parents=True, exist_ok=True)


class Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


@contextmanager
def server():
    h = functools.partial(Q, directory=str(WEB))
    s = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h)
    t = threading.Thread(target=s.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{s.server_port}/"
    finally:
        s.shutdown()
        t.join(timeout=3)


def find_chrome() -> str:
    candidates = [
        shutil.which("chrome"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    raise RuntimeError("Chrome/Edge not found")


def main():
    with server() as url, sync_playwright() as pw:
        b = pw.chromium.launch(headless=True, executable_path=find_chrome(),
                               args=["--enable-unsafe-webgpu"])
        ctx = b.new_context(viewport={"width": 760, "height": 900},
                            record_video_dir=str(OUT / "raw"), record_video_size={"width": 760, "height": 900})
        pg = ctx.new_page()
        pg.set_default_timeout(120000)
        pg.goto(url, wait_until="domcontentloaded")
        pg.wait_for_function("() => window.plant && window.plant.engine.steps >= 160", timeout=90000)
        pg.wait_for_timeout(1200)
        pg.click("#btnDmgSmall")
        pg.wait_for_function("() => window.plant.engine.steps >= 420", timeout=60000)
        pg.wait_for_timeout(1200)
        pg.click("#btnDmgBig")
        pg.wait_for_function("() => window.plant.engine.steps >= 700", timeout=60000)
        pg.wait_for_timeout(1500)
        box = pg.locator("#view").bounding_box()
        x0 = box["x"] + box["width"] * 0.42
        y0 = box["y"] + box["height"] * 0.55
        pg.mouse.move(x0, y0)
        pg.mouse.down()
        pg.mouse.move(x0 + box["width"] * 0.18, y0 - box["height"] * 0.12, steps=12)
        pg.mouse.up()
        pg.wait_for_function("() => window.plant.engine.steps >= 1000", timeout=60000)
        pg.wait_for_timeout(2000)
        ctx.close()
        b.close()
    vids = sorted((OUT / "raw").glob("*.webm"), key=lambda p: p.stat().st_mtime)
    if vids:
        dst = OUT / "plant_demo.webm"
        if dst.exists():
            dst.unlink()
        shutil.move(str(vids[-1]), dst)
        print("video:", dst, flush=True)
    for old in (OUT / "raw").glob("*.webm"):
        old.unlink()
    print("RECORD DONE", flush=True)


if __name__ == "__main__":
    main()
