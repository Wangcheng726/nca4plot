"""Local visual/interaction verification for all three modes.

Metrics: flicker while paused, brush alignment via the renderer's own
screen->CA mapping, motion while running, and (target modes) RGBA MSE against
the whole-sky training target so "looks like the painting" is a number.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "media" / "iterate"
OUT.mkdir(parents=True, exist_ok=True)
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/"
MODE = sys.argv[2] if len(sys.argv) > 2 else "regenerate"
TAG = sys.argv[3] if len(sys.argv) > 3 else MODE
assert MODE in {"regenerate", "growing", "persistent"}

BOOT_STEPS = {"regenerate": 440, "growing": 2, "persistent": 320}[MODE]


def shot(page, name):
    data = page.screenshot()
    (OUT / f"{TAG}_{name}.png").write_bytes(data)
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.int16)


def mean_diff(a, b):
    return float(np.abs(a - b).mean())


def pull_rgba(page):
    info = page.evaluate(
        """() => {
          const e = window.nca4plot.engine;
          const w = e.width, h = e.height, n2 = w * h, d = e.stateData;
          const out = [];
          for (let y = 0; y < h; y += 2) for (let x = 0; x < w; x += 2) {
            const i = y * w + x;
            out.push(d[i], d[n2 + i], d[2 * n2 + i], d[3 * n2 + i]);
          }
          return { w, h, kind: e.kind, data: out };
        }"""
    )
    arr = np.asarray(info["data"], dtype=np.float32).reshape(info["h"] // 2, info["w"] // 2, 4)
    return arr, info


def target_rgba(width=256, height=144):
    img = Image.open(ROOT / "data" / f"target_sky_{width}x{height}.png").convert("RGBA")
    return np.asarray(img, dtype=np.float32)[::2, ::2] / 255.0


def main():
    report = {"mode": MODE}
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
        quality = "&quality=256" if MODE == "persistent" else ""
        page.goto(f"{URL}?autostart=1&mode={MODE}{quality}", wait_until="domcontentloaded")
        page.wait_for_function(
            "(min) => window.nca4plot && window.nca4plot.entered && window.nca4plot.steps >= min",
            arg=BOOT_STEPS, timeout=180_000,
        )
        report["backend"] = page.evaluate("window.nca4plot.backend")
        report["grid"] = page.evaluate(
            "() => [window.nca4plot.engine.width, window.nca4plot.engine.height]")
        grid_w, grid_h = report["grid"]

        if MODE == "growing":
            # Freeze the slow clock at the constellation phase so the boot
            # shot shows scattered memory stars without the intro overlay.
            page.evaluate("window.nca4plot.pause()")
            page.wait_for_timeout(1800)
            shot(page, "growth_boot")
            page.evaluate("window.nca4plot.pause()")
            for steps, name in ((40, "growth_mid"), (240, "growth_late")):
                page.wait_for_function(
                    "(s) => window.nca4plot.steps >= s", arg=steps, timeout=120_000)
                shot(page, name)
        if MODE == "regenerate":
            page.wait_for_timeout(900)
            shot(page, "initial_painting")
            state, _ = pull_rgba(page)
            report["initial_rgba_mse_vs_target"] = round(
                float(((state - target_rgba()) ** 2).mean()), 5)

        page.wait_for_function("() => window.nca4plot.steps > 320", timeout=120_000)
        page.wait_for_timeout(1200)
        flow_a = shot(page, "flow_a")
        page.wait_for_timeout(400)
        flow_b = shot(page, "flow_b")
        report["flow_motion_diff"] = round(mean_diff(flow_a, flow_b), 4)

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

        # --- brush alignment: real mouse stroke, mapped by the renderer itself ---
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
        info = page.evaluate(
            """(pts) => {
              const e = window.nca4plot.engine;
              const r = window.nca4plot.renderer;
              const d = e.stateData, w = e.width, h = e.height, n2 = w * h;
              let sx = 0, sy = 0, cnt = 0;
              for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
                const i = y * w + x;
                const dead = e.kind === 'target'
                  ? d[3 * n2 + i] < 0.08
                  : (() => { let m = 0; for (let c = 0; c < 12; c++) m = Math.max(m, Math.abs(d[c * n2 + i])); return m < 0.08; })();
                if (dead) { sx += x; sy += y; cnt++; }
              }
              const mapped = pts.map(([px, py]) => r.screenToCa(px, py));
              return { cx: cnt ? sx / cnt : -1, cy: cnt ? sy / cnt : -1, cnt, mapped };
            }""",
            [[cx, cy], [cx + 60, cy + 40]],
        )
        report["erased_state_centroid"] = [round(info["cx"], 1), round(info["cy"], 1)]
        mapped = [m for m in info["mapped"] if m]
        if mapped:
            ex = sum(m["x"] for m in mapped) / len(mapped)
            ey = sum(m["y"] for m in mapped) / len(mapped)
            report["expected_state_centroid"] = [round(ex, 1), round(ey, 1)]
            report["erased_cells"] = info["cnt"]
            if info["cnt"]:
                report["alignment_error_cells"] = round(float(np.hypot(info["cx"] - ex, info["cy"] - ey)), 2)
        else:
            ex = ey = None
        page.evaluate("window.nca4plot.pause()")  # resume for healing
        page.wait_for_timeout(5000)
        shot(page, "healed")
        if MODE == "persistent":
            # Distill persistent semantics: the wound core is damage-locked and
            # must still be exactly zero after the healing window.
            retention = page.evaluate(
                """() => {
                  const e = window.nca4plot.engine;
                  const n2 = e.n2, d = e.stateData;
                  let maxAbs = 0;
                  for (const p of e.lockedPixels) {
                    for (let c = 0; c < 12; c += 1) maxAbs = Math.max(maxAbs, Math.abs(d[c * n2 + p]));
                  }
                  return { locked: e.lockedPixels.length, maxAbs };
                }"""
            )
            report["wound_locked_cells"] = retention["locked"]
            report["wound_retention_max_abs"] = round(float(retention["maxAbs"]), 6)
        if MODE != "persistent":
            state, _ = pull_rgba(page)
            tgt = target_rgba(grid_w, grid_h)
            report["healed_rgba_mse_vs_target"] = round(float(((state - tgt) ** 2).mean()), 5)
            if ex is not None:
                hh, ww = state.shape[:2]
                yy, xx = np.mgrid[0:hh, 0:ww]
                disc = (xx - ex / 2) ** 2 + (yy - ey / 2) ** 2 <= (0.12 * min(ww, hh)) ** 2
                report["wound_rgba_mse_vs_target"] = round(
                    float(((state[disc] - tgt[disc]) ** 2).mean()), 5)
        report["page_errors"] = errors
        browser.close()
    (OUT / f"{TAG}_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()