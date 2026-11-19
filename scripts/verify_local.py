"""Local visual/interaction verification: flicker metric, brush alignment, screenshots."""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "media" / "iterate"
OUT.mkdir(parents=True, exist_ok=True)
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/"
QUALITY = sys.argv[2] if len(sys.argv) > 2 else "256"
TAG = sys.argv[3] if len(sys.argv) > 3 else "run"


def shot(page, name):
    data = page.screenshot()
    (OUT / f"{TAG}_{name}.png").write_bytes(data)
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.int16)


def mean_diff(a, b):
    return float(np.abs(a - b).mean())


def main():
    report = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True,
            executable_path=next(
                c for c in [
                    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                ] if Path(c).exists()
            ),
            args=[
                "--enable-unsafe-webgpu",
                "--ignore-gpu-blocklist",
                "--disable-background-timer-throttling",
            ],
        )
        context = browser.new_context(viewport={"width": 1600, "height": 900})
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"{URL}?autostart=1&quality={QUALITY}", wait_until="domcontentloaded")
        page.wait_for_function(
            "() => window.nca4plot && window.nca4plot.entered && window.nca4plot.steps > 40",
            timeout=120_000,
        )
        report["backend"] = page.evaluate("window.nca4plot.backend")
        shot(page, "growth_early")
        page.wait_for_function("() => window.nca4plot.steps > 320", timeout=120_000)
        page.wait_for_timeout(1200)
        flow_a = shot(page, "flow_a")
        page.wait_for_timeout(400)
        flow_b = shot(page, "flow_b")
        report["flow_motion_diff"] = mean_diff(flow_a, flow_b)

        # --- flicker: pause CA, let toast expire, sample frames ---
        page.evaluate("window.nca4plot.pause()")
        page.wait_for_timeout(2300)
        frames = [shot(page, f"pause_{i}") for i in range(6)]
        for i in range(5):
            page.wait_for_timeout(160)
            frames.append(shot(page, f"pause_{i+6}"))
        diffs = [mean_diff(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]
        report["paused_frame_diffs"] = [round(d, 4) for d in diffs]
        report["paused_flicker_max"] = round(max(diffs), 4)

        # --- brush alignment: real mouse stroke at a known screen point ---
        # keep CA paused so the only screen change is the wound itself
        page.wait_for_timeout(600)
        before = shot(page, "pre_stroke")
        cx, cy = 1150, 320
        page.mouse.move(cx, cy)
        page.mouse.down()
        page.mouse.move(cx + 60, cy + 40, steps=8)
        page.mouse.up()
        page.wait_for_timeout(350)
        after = shot(page, "wounded")
        diff = np.abs(after - before).mean(axis=2)
        ys, xs = np.nonzero(diff > 24)
        if len(xs):
            report["stroke_screen_centroid"] = [round(float(xs.mean()), 1), round(float(ys.mean()), 1)]
        report["stroke_target"] = [cx + 30, cy + 20]
        # state-space centroid of erased cells
        info = page.evaluate(
            """() => {
              const e = window.nca4plot.engine;
              const d = e.stateData, n = e.size, n2 = n * n;
              let sx = 0, sy = 0, cnt = 0;
              for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
                const i = y * n + x;
                let m = 0;
                for (let c = 0; c < 12; c++) m = Math.max(m, Math.abs(d[c * n2 + i]));
                if (m < 0.08) { sx += x; sy += y; cnt++; }
              }
              const rect = document.querySelector('#sky').getBoundingClientRect();
              return { cx: cnt ? sx / cnt : -1, cy: cnt ? sy / cnt : -1, cnt,
                       size: n, w: rect.width, h: rect.height };
            }"""
        )
        n, w, h = info["size"], info["w"], info["h"]
        aspect = w / h
        pts = [(cx, cy), (cx + 60, cy + 40)]
        ex, ey = [], []
        for px, py in pts:
            sx = px / w
            sy = 1 - py / h
            ux = sx if aspect > 1 else 0.5 + (sx - 0.5) * aspect
            uy = 0.5 + (sy - 0.5) / aspect if aspect > 1 else sy
            ex.append(ux * n)
            ey.append((1 - uy) * n)
        report["erased_state_centroid"] = [round(info["cx"], 1), round(info["cy"], 1)]
        report["expected_state_centroid"] = [round(sum(ex) / 2, 1), round(sum(ey) / 2, 1)]
        report["erased_cells"] = info["cnt"]
        if info["cnt"]:
            report["alignment_error_cells"] = round(
                float(np.hypot(info["cx"] - sum(ex) / 2, info["cy"] - sum(ey) / 2)), 2
            )
        page.evaluate("window.nca4plot.pause()")  # resume for healing
        page.wait_for_timeout(4000)
        shot(page, "healed")
        report["page_errors"] = errors
        browser.close()
    (OUT / f"{TAG}_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
