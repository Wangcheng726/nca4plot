"""Headless user-style verification of the minimal plant UI (local only).

Checks: load + auto-growth, real mouse erase position accuracy, small/big
damage buttons + regrowth, steady-state flicker. Screenshots + report.json.
Run AFTER training finishes (GPU tasks must stay serial).
"""
from __future__ import annotations

import functools
import http.server
import json
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
OUT = ROOT / "checkpoints" / "plant_web_verify"
OUT.mkdir(parents=True, exist_ok=True)


class Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


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


def state(pg):
    return np.array(pg.evaluate("Array.from(window.plant.engine.stateData)"), dtype=np.float32)


def manual_run(pg, n):
    return pg.evaluate("""async (n) => {
        const e = window.plant.engine;
        for (let i = 0; i < n; i += 1) { if (!await e.run()) break; }
        window.plant.updateRecovery(); window.plant.render();
        return { steps: e.steps, recovery: window.plant.recovery,
                 sat: e.satFraction, guard: e.guardEvents || 0 };
    }""", n)


def shot(pg, name):
    pg.locator("#view").screenshot(path=str(OUT / name))


def find_chrome() -> str:
    import shutil
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
    report = {}
    with server() as url, sync_playwright() as pw:
        b = pw.chromium.launch(headless=True, executable_path=find_chrome(), args=["--enable-unsafe-webgpu"])
        pg = b.new_page(viewport={"width": 760, "height": 900})
        pg.set_default_timeout(120000)
        pg.goto(url, wait_until="domcontentloaded")
        pg.wait_for_function("() => window.plant && window.plant.engine.steps > 0", timeout=90000)
        report["backend"] = pg.evaluate("window.plant.engine.backend")
        report["params"] = pg.evaluate("window.plant.engine.parameterCount")
        pg.screenshot(path=str(OUT / "ui.png"))

        # --- growth from the single seed ---
        pg.wait_for_function("() => window.plant.engine.steps >= 160", timeout=90000)
        pg.evaluate("window.plant.updateRecovery()")
        report["growth"] = {"steps": pg.evaluate("window.plant.engine.steps"),
                            "recovery": pg.evaluate("window.plant.recovery")}
        shot(pg, "grown.png")

        # --- real mouse erase: position accuracy ---
        pg.evaluate("window.plant.setPaused(true)")
        pg.wait_for_timeout(150)
        before = state(pg)
        box = pg.locator("#view").bounding_box()
        gx, gy = 40.0, 40.0
        w = pg.evaluate("window.plant.engine.width")
        h = pg.evaluate("window.plant.engine.height")
        cx = box["x"] + (gx + 0.5) / w * box["width"]
        cy = box["y"] + (gy + 0.5) / h * box["height"]
        pg.mouse.move(cx, cy)
        pg.mouse.down()
        pg.mouse.up()
        pg.wait_for_timeout(120)
        after = state(pg)
        chn = pg.evaluate("window.plant.engine.chn")
        diff = np.abs(after - before).reshape(chn, h, w).max(axis=0)
        ys, xs = np.nonzero(diff > 0.05)
        if len(xs):
            wts = diff[ys, xs]
            cent = (float((xs * wts).sum() / wts.sum()), float((ys * wts).sum() / wts.sum()))
        else:
            cent = (float("nan"), float("nan"))
        err = float(np.hypot(cent[0] - gx, cent[1] - gy))
        report["mouse_mapping"] = {"target_grid": [gx, gy], "erase_centroid": cent,
                                   "error_cells": round(err, 3), "cells_erased": int(len(xs))}
        assert len(xs) > 0, "mouse erase touched nothing"
        assert err < 2.0, f"erase centroid off by {err:.2f} cells"
        shot(pg, "mouse_erase.png")

        # --- small damage button + regrowth ---
        pg.evaluate("window.plant.reset()")
        manual_run(pg, 24)  # 96 steps
        rec0 = pg.evaluate("window.plant.recovery")
        pg.click("#btnDmgSmall")
        rec_dmg = pg.evaluate("window.plant.recovery")
        shot(pg, "small_erased.png")
        curve = {}
        for mark in (16, 64, 128, 256):
            res = manual_run(pg, (mark - (0 if mark == 16 else prev)) // 4)
            prev = mark
            curve[str(mark)] = round(res["recovery"], 4)
            if mark in (64, 256):
                shot(pg, f"small_plus{mark}.png")
        report["small_damage"] = {"recovery_before": round(rec0, 4),
                                  "recovery_erased": round(rec_dmg, 4), "curve": curve}

        # --- big damage button + regrowth ---
        pg.evaluate("window.plant.reset()")
        manual_run(pg, 24)
        rec0 = pg.evaluate("window.plant.recovery")
        pg.click("#btnDmgBig")
        rec_dmg = pg.evaluate("window.plant.recovery")
        shot(pg, "big_erased.png")
        curve = {}
        prev = 0
        for mark in (16, 64, 128, 256):
            res = manual_run(pg, (mark - prev) // 4)
            prev = mark
            curve[str(mark)] = round(res["recovery"], 4)
            if mark in (64, 256):
                shot(pg, f"big_plus{mark}.png")
        report["big_damage"] = {"recovery_before": round(rec0, 4),
                                "recovery_erased": round(rec_dmg, 4), "curve": curve}

        # --- steady-state flicker (running, no damage) ---
        pg.evaluate("window.plant.reset()")
        manual_run(pg, 64)  # 256 steps
        pg.evaluate("window.plant.setPaused(false)")
        pg.wait_for_timeout(400)
        shot(pg, "flicker_a.png")
        pg.wait_for_timeout(700)
        shot(pg, "flicker_b.png")
        a = np.asarray(Image.open(OUT / "flicker_a.png").convert("RGB"), np.float32)
        bb = np.asarray(Image.open(OUT / "flicker_b.png").convert("RGB"), np.float32)
        report["flicker_mean_abs"] = round(float(np.abs(a - bb).mean()), 3)
        report["final_steps"] = pg.evaluate("window.plant.engine.steps")
        b.close()
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print("PLANT WEB VERIFY DONE", OUT, flush=True)


if __name__ == "__main__":
    main()
