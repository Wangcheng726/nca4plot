"""Measure growing-boot dynamics atomically (steps + state in one JS turn)."""
import io, json
from pathlib import Path
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
tgt = np.asarray(Image.open(ROOT / "data" / "target_sky_256x144.png").convert("RGBA"), dtype=np.float32)[::2, ::2] / 255.0

with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True, executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        args=["--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--disable-background-timer-throttling"])
    page = browser.new_context(viewport={"width": 1600, "height": 900}).new_page()
    page.goto("http://127.0.0.1:8765/?autostart=1&mode=growing", wait_until="domcontentloaded")
    page.wait_for_function("() => window.nca4plot && window.nca4plot.entered", timeout=120000)
    rows = []
    for s in (2, 6, 10, 16, 24, 40, 60, 120, 240):
        page.wait_for_function("(s) => window.nca4plot.steps >= s", arg=s, timeout=120000)
        info = page.evaluate("""() => {
          const e = window.nca4plot.engine;
          const w = e.width, h = e.height, n2 = w * h, d = e.stateData;
          const out = [];
          for (let y = 0; y < h; y += 2) for (let x = 0; x < w; x += 2) {
            const i = y * w + x;
            out.push(d[i], d[n2 + i], d[2 * n2 + i], d[3 * n2 + i]);
          }
          return { steps: e.steps, data: out };
        }""")
        st = np.asarray(info["data"], dtype=np.float32).reshape(tgt.shape)
        per = ((st - tgt) ** 2).mean(axis=2)
        rows.append({"steps": info["steps"],
                     "mse": round(float(per.mean()), 5),
                     "close_frac": round(float((per < 0.01).mean()), 4),
                     "alpha_cov": round(float((st[:, :, 3] > 0.1).mean()), 4)})
    browser.close()
print(json.dumps(rows, indent=1))
